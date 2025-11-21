#!/usr/bin/env python3
"""
Training script for Asymmetric Projection Model WITH Predictor.

Architecture: (1, 0, 1)
- Shared frozen encoder
- Separate query/doc projections
- Predictor for query→doc transformation

Combines benefits of:
- AsymmetricDualEncoder: Explicit query/doc spaces
- IsotropicGaussianEncoder: Predictive learning signal
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragcun.asymmetric_predictor_model import AsymmetricWithPredictor

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


class AsymmetricPredictorLoss(nn.Module):
    """
    Combined loss for Asymmetric + Predictor model.
    
    Three components:
    1. Contrastive: Pull query-doc pairs together, push negatives apart
    2. Isotropy: Encourage uniform distribution in both query and doc spaces
    3. Predictive: JEPA-style prediction of doc from query
    """
    
    def __init__(
        self,
        lambda_contrastive=1.0,
        lambda_isotropy=1.0,
        lambda_predictive=1.0,
        temperature=0.05,
        use_stopgrad=True
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_isotropy = lambda_isotropy
        self.lambda_predictive = lambda_predictive
        self.temperature = temperature
        self.use_stopgrad = use_stopgrad
    
    def compute_isotropy_loss(self, embeddings):
        """Compute SIGReg isotropy loss."""
        # Center embeddings
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = (centered.T @ centered) / (embeddings.shape[0] - 1)
        
        # Target: identity matrix scaled by variance
        variance = torch.var(embeddings)
        target_cov = torch.eye(cov.shape[0], device=cov.device) * variance
        
        # Frobenius norm of difference
        isotropy_loss = torch.norm(cov - target_cov, p='fro') / cov.shape[0]
        
        return isotropy_loss
    
    def forward(self, query_emb, pos_emb, neg_emb=None, predicted_pos=None):
        """
        Compute combined loss.
        
        Args:
            query_emb: Query embeddings [batch, dim]
            pos_emb: Positive doc embeddings [batch, dim]
            neg_emb: Negative doc embeddings [batch, dim] or None
            predicted_pos: Predicted doc from query [batch, dim]
        
        Returns:
            total_loss, loss_dict
        """
        batch_size = query_emb.shape[0]
        device = query_emb.device
        
        # 1. Contrastive Loss (InfoNCE with temperature scaling)
        if self.lambda_contrastive > 0:
            # Normalize for cosine similarity
            query_norm = F.normalize(query_emb, p=2, dim=1)
            pos_norm = F.normalize(pos_emb, p=2, dim=1)
            
            # Positive similarities
            pos_sim = torch.sum(query_norm * pos_norm, dim=1) / self.temperature
            
            # Negative similarities (in-batch negatives)
            if neg_emb is not None:
                neg_norm = F.normalize(neg_emb, p=2, dim=1)
                neg_sim = torch.sum(query_norm * neg_norm, dim=1) / self.temperature
                
                # Binary classification: positive vs negative
                logits = torch.stack([pos_sim, neg_sim], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                contrastive_loss = F.cross_entropy(logits, labels)
                
                # Accuracy for monitoring
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == labels).float().mean()
            else:
                # In-batch negatives: all other docs in batch
                # Compute all query-doc similarities
                all_sim = (query_norm @ pos_norm.T) / self.temperature
                
                # Labels: diagonal elements are positives
                labels = torch.arange(batch_size, device=device)
                contrastive_loss = F.cross_entropy(all_sim, labels)
                
                # Accuracy
                with torch.no_grad():
                    preds = torch.argmax(all_sim, dim=1)
                    accuracy = (preds == labels).float().mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            accuracy = 0.0
        
        # 2. Isotropy Loss (on both query and doc spaces)
        if self.lambda_isotropy > 0:
            query_iso = self.compute_isotropy_loss(query_emb)
            doc_iso = self.compute_isotropy_loss(pos_emb)
            isotropy_loss = (query_iso + doc_iso) / 2
            
            # Track separately for monitoring
            query_iso_val = query_iso.item()
            doc_iso_val = doc_iso.item()
        else:
            isotropy_loss = torch.tensor(0.0, device=device)
            query_iso_val = 0.0
            doc_iso_val = 0.0
        
        # 3. Predictive Loss (JEPA-style)
        if predicted_pos is not None and self.lambda_predictive > 0:
            # Stop-gradient on target to prevent collapse
            target = pos_emb.detach() if self.use_stopgrad else pos_emb
            predictive_loss = F.mse_loss(predicted_pos, target)
        else:
            predictive_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = (
            self.lambda_contrastive * contrastive_loss +
            self.lambda_isotropy * isotropy_loss +
            self.lambda_predictive * predictive_loss
        )
        
        # Loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
            'isotropy': isotropy_loss.item() if isinstance(isotropy_loss, torch.Tensor) else isotropy_loss,
            'query_iso': query_iso_val,
            'doc_iso': doc_iso_val,
            'predictive': predictive_loss.item() if isinstance(predictive_loss, torch.Tensor) else predictive_loss,
            'accuracy': accuracy if isinstance(accuracy, float) else accuracy.item(),
        }
        
        return total_loss, loss_dict


def collate_fn(batch):
    """Collate function for DataLoader."""
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
        
        # Predict doc from query (JEPA-style)
        predicted_pos = model.predict_doc_from_query(query_emb)
        
        # Compute loss
        loss, loss_dict = criterion(query_emb, pos_emb, neg_emb, predicted_pos)
        
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
            'pred': f"{loss_dict['predictive']:.4f}",
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
        for batch in tqdm(dataloader, desc="Validation"):
            query_emb = model(batch['queries'], is_query=True)
            pos_emb = model(batch['positives'], is_query=False)
            neg_emb = model(batch['negatives'], is_query=False) if batch['negatives'] else None
            predicted_pos = model.predict_doc_from_query(query_emb)
            
            loss, loss_dict = criterion(query_emb, pos_emb, neg_emb, predicted_pos)
            
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
    parser.add_argument('--lambda_predictive', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--no_stopgrad', action='store_true', help='Disable stop-gradient on predictive target')
    
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
    logger.info("Asymmetric + Predictor Training (1, 0, 1)")
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
    logger.info(f"  λ_predictive: {args.lambda_predictive}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Stop-gradient: {not args.no_stopgrad}")
    logger.info("")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating asymmetric + predictor model...")
    model = AsymmetricWithPredictor(
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=True,
        normalize_embeddings=False
    )
    model = model.to(device)
    
    # Create loss
    criterion = AsymmetricPredictorLoss(
        lambda_contrastive=args.lambda_contrastive,
        lambda_isotropy=args.lambda_isotropy,
        lambda_predictive=args.lambda_predictive,
        temperature=args.temperature,
        use_stopgrad=not args.no_stopgrad
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
    
    # Save config
    config = vars(args)
    config_path = os.path.join(args.output_dir, 'train_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Training loop
    logger.info("\nStarting training...")
    logger.info("="*80)
    
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_components = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"  Contrastive: {train_components['contrastive']:.4f}")
        logger.info(f"  Isotropy: {train_components['isotropy']:.4f}")
        logger.info(f"    - Query: {train_components['query_iso']:.4f}")
        logger.info(f"    - Doc: {train_components['doc_iso']:.4f}")
        logger.info(f"  Predictive: {train_components['predictive']:.4f}")
        logger.info(f"  Accuracy: {train_components['accuracy']:.3f}")
        
        # Validate
        if val_loader:
            val_loss, val_components = evaluate(model, val_loader, criterion, device)
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"  Contrastive: {val_components['contrastive']:.4f}")
            logger.info(f"  Isotropy: {val_components['isotropy']:.4f}")
            logger.info(f"  Predictive: {val_components['predictive']:.4f}")
            logger.info(f"  Accuracy: {val_components['accuracy']:.3f}")
        else:
            val_loss = train_loss
            val_components = train_components
        
        # Save stats
        training_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_components': train_components,
            'val_loss': val_loss,
            'val_components': val_components
        })
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
            logger.info(f"✅ New best model! Saved to {best_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model: {final_path}")
    
    # Save training stats
    stats_path = os.path.join(args.output_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Saved training stats: {stats_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Training complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()


