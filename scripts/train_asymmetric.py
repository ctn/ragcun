#!/usr/bin/env python3
"""
Training script for Asymmetric Projection Model.

Uses different projection heads for queries vs documents.
No predictor - just clean contrastive learning.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.asymmetric_model import AsymmetricDualEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    def __init__(self, data_path: str):
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} training examples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'positive': item['positive'],
            'negative': item.get('negative', None)
        }


class AsymmetricLoss(nn.Module):
    """
    Loss for asymmetric projections.
    
    Components:
    1. Contrastive loss: InfoNCE between queries and docs
    2. Isotropy loss: For both query and doc spaces
    """
    
    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_isotropy: float = 1.0,
        temperature: float = 0.05
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_isotropy = lambda_isotropy
        self.temperature = temperature
        
    def forward(self, query_emb, pos_emb, neg_emb=None):
        """
        Compute loss.
        
        Args:
            query_emb: Query embeddings [batch, dim]
            pos_emb: Positive doc embeddings [batch, dim]
            neg_emb: Optional negative doc embeddings [batch, dim]
        """
        batch_size = query_emb.shape[0]
        
        # 1. Contrastive Loss (InfoNCE with COSINE similarity)
        # Normalize embeddings for cosine similarity
        query_emb_norm = F.normalize(query_emb, p=2, dim=1)
        pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)
        
        # Similarity matrix: [batch, batch] using COSINE similarity
        sim_matrix = query_emb_norm @ pos_emb_norm.T / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=query_emb.device)
        
        # Cross-entropy loss
        contrastive_loss = F.cross_entropy(sim_matrix, labels)
        
        # 2. Isotropy Loss (for both query and doc spaces)
        # Query space isotropy
        query_mean = query_emb.mean(dim=0, keepdim=True)
        query_centered = query_emb - query_mean
        query_cov = (query_centered.T @ query_centered) / (batch_size - 1)
        query_variance = torch.var(query_emb)
        query_target_cov = torch.eye(query_cov.shape[0], device=query_emb.device) * query_variance
        query_isotropy_loss = torch.norm(query_cov - query_target_cov, p='fro') / query_cov.shape[0]
        
        # Doc space isotropy
        doc_mean = pos_emb.mean(dim=0, keepdim=True)
        doc_centered = pos_emb - doc_mean
        doc_cov = (doc_centered.T @ doc_centered) / (batch_size - 1)
        doc_variance = torch.var(pos_emb)
        doc_target_cov = torch.eye(doc_cov.shape[0], device=pos_emb.device) * doc_variance
        doc_isotropy_loss = torch.norm(doc_cov - doc_target_cov, p='fro') / doc_cov.shape[0]
        
        isotropy_loss = (query_isotropy_loss + doc_isotropy_loss) / 2
        
        # Total loss
        total_loss = (
            self.lambda_contrastive * contrastive_loss +
            self.lambda_isotropy * isotropy_loss
        )
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy: correct predictions on diagonal
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
            
            # Average similarity
            pos_sim = torch.diag(sim_matrix).mean()
            neg_sim = (sim_matrix.sum() - torch.diag(sim_matrix).sum()) / (batch_size * (batch_size - 1))
        
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'isotropy': isotropy_loss.item(),
            'query_isotropy': query_isotropy_loss.item(),
            'doc_isotropy': doc_isotropy_loss.item(),
            'accuracy': accuracy.item(),
            'pos_sim': pos_sim.item(),
            'neg_sim': neg_sim.item(),
            'query_std': torch.std(query_emb).item(),
            'doc_std': torch.std(pos_emb).item()
        }
        
        return total_loss, loss_dict


def collate_fn(batch):
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negative'] for item in batch if item['negative'] is not None]
    
    return {
        'queries': queries,
        'positives': positives,
        'negatives': negatives if negatives else None
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Encode with appropriate projections
        query_emb = model(batch['queries'], is_query=True)
        pos_emb = model(batch['positives'], is_query=False)
        neg_emb = model(batch['negatives'], is_query=False) if batch['negatives'] else None
        
        # Compute loss
        loss, loss_dict = criterion(query_emb, pos_emb, neg_emb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'contr': f"{loss_dict['contrastive']:.4f}",
            'iso': f"{loss_dict['isotropy']:.4f}",
            'acc': f"{loss_dict['accuracy']:.3f}"
        })
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            query_emb = model(batch['queries'], is_query=True)
            pos_emb = model(batch['positives'], is_query=False)
            neg_emb = model(batch['negatives'], is_query=False) if batch['negatives'] else None
            
            loss, loss_dict = criterion(query_emb, pos_emb, neg_emb)
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', default=None)
    
    # Model
    parser.add_argument('--base_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--output_dim', type=int, default=768)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Loss weights
    parser.add_argument('--lambda_contrastive', type=float, default=1.0)
    parser.add_argument('--lambda_isotropy', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.05)
    
    # Output
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--log_file', default=None)
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    
    # Print config
    logger.info("="*80)
    logger.info("Asymmetric Projection Training")
    logger.info("="*80)
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output dim: {args.output_dim}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")
    logger.info("Loss weights:")
    logger.info(f"  λ_contrastive: {args.lambda_contrastive}")
    logger.info(f"  λ_isotropy: {args.lambda_isotropy}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info("="*80)
    logger.info("")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating asymmetric projection model...")
    model = AsymmetricDualEncoder(
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=True,
        normalize_embeddings=False
    )
    model = model.to(device)
    
    # Create loss
    criterion = AsymmetricLoss(
        lambda_contrastive=args.lambda_contrastive,
        lambda_isotropy=args.lambda_isotropy,
        temperature=args.temperature
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Load data
    logger.info("Loading training data...")
    train_dataset = ContrastiveDataset(args.train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = None
    if args.val_data:
        logger.info("Loading validation data...")
        val_dataset = ContrastiveDataset(args.val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    logger.info("\nStarting training...")
    logger.info("="*80)
    
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        logger.info(f"\nTraining Loss: {train_loss:.6f}")
        logger.info(f"  Contrastive: {train_components['contrastive']:.6f}")
        logger.info(f"  Isotropy: {train_components['isotropy']:.6f}")
        logger.info(f"  Accuracy: {train_components['accuracy']:.4f}")
        logger.info(f"  Pos similarity: {train_components['pos_sim']:.4f}")
        logger.info(f"  Neg similarity: {train_components['neg_sim']:.4f}")
        logger.info(f"  Query std: {train_components['query_std']:.4f}")
        logger.info(f"  Doc std: {train_components['doc_std']:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_components = evaluate(model, val_loader, criterion, device)
            logger.info(f"\nValidation Loss: {val_loss:.6f}")
            logger.info(f"  Contrastive: {val_components['contrastive']:.6f}")
            logger.info(f"  Isotropy: {val_components['isotropy']:.6f}")
            logger.info(f"  Accuracy: {val_components['accuracy']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"  ✅ New best validation loss!")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': vars(args)
                }, f"{args.output_dir}/best_model.pt")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'args': vars(args)
        }, f"{args.output_dir}/checkpoint_epoch{epoch}.pt")
        
        # Record stats
        stats = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_components': train_components,
        }
        if val_loader:
            stats['val_loss'] = val_loss
            stats['val_components'] = val_components
        training_stats.append(stats)
    
    # Save training stats
    with open(f"{args.output_dir}/training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

