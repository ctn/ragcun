#!/usr/bin/env python3
"""
Training script for ResPred: Residual Predictor for Dense Retrieval

Key differences from standard training:
1. Uses ResidualPredictor with bounded outputs (Tanh)
2. Learnable scale factor (alpha)
3. Adds residual regularization loss
4. Returns both predicted_doc and delta for loss computation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.respred_model import ResidualGaussianEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning."""
    
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


class ResPredLoss(nn.Module):
    """
    Loss function for ResPred with residual regularization.
    
    Components:
    1. Isotropy loss: Isotropic covariance in embedding space
    2. Predictive loss: MSE(predicted_doc, doc_emb)
    3. Residual loss: L2 regularization on delta (NEW!)
    4. Contrastive loss: Optional alignment loss
    """
    
    def __init__(
        self,
        lambda_isotropy: float = 1.5,
        lambda_predictive: float = 1.2,
        lambda_residual: float = 0.01,
        lambda_contrastive: float = 0.0,
        margin: float = 1.0,
        use_stopgrad: bool = True
    ):
        super().__init__()
        self.lambda_isotropy = lambda_isotropy
        self.lambda_predictive = lambda_predictive
        self.lambda_residual = lambda_residual
        self.lambda_contrastive = lambda_contrastive
        self.margin = margin
        self.use_stopgrad = use_stopgrad
        
    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        predicted_pos: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
        neg_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ResPred loss.
        
        Args:
            query_emb: Query embeddings (Z-space) [batch, dim]
            pos_emb: Positive embeddings (Z-space) [batch, dim]
            predicted_pos: query_emb + alpha*delta [batch, dim]
            delta: Raw residual from predictor [batch, dim]
            neg_emb: Optional negative embeddings [batch, dim]
        """
        batch_size = query_emb.shape[0]
        
        # 1. Isotropy Loss
        all_emb = torch.cat([query_emb, pos_emb], dim=0)
        mean = all_emb.mean(dim=0, keepdim=True)
        centered = all_emb - mean
        cov = (centered.T @ centered) / (all_emb.shape[0] - 1)
        
        # Scale-invariant isotropy
        variance = torch.var(all_emb)
        target_cov = torch.eye(cov.shape[0], device=cov.device) * variance
        isotropy_loss = torch.norm(cov - target_cov, p='fro') / cov.shape[0]
        
        # 2. Predictive Loss (JEPA)
        if predicted_pos is not None and self.lambda_predictive > 0:
            target = pos_emb.detach() if self.use_stopgrad else pos_emb
            predictive_loss = F.mse_loss(predicted_pos, target)
        else:
            predictive_loss = torch.tensor(0.0, device=query_emb.device)
        
        # 3. Residual Regularization Loss (NEW!)
        if delta is not None and self.lambda_residual > 0:
            # L2 norm of residual (encourage small corrections)
            residual_loss = torch.mean(torch.sum(delta ** 2, dim=1))
        else:
            residual_loss = torch.tensor(0.0, device=query_emb.device)
        
        # 4. Contrastive Loss (optional)
        if self.lambda_contrastive > 0:
            pos_dist = torch.norm(query_emb - pos_emb, p=2, dim=1)
            
            if neg_emb is not None:
                neg_dist = torch.norm(query_emb - neg_emb, p=2, dim=1)
                contrastive_loss = torch.mean(
                    F.relu(pos_dist - neg_dist + self.margin)
                )
            else:
                # In-batch negatives
                all_emb_contrast = torch.cat([query_emb, pos_emb], dim=0)
                dist_matrix = torch.cdist(all_emb_contrast, all_emb_contrast, p=2)
                
                pos_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
                for i in range(batch_size):
                    pos_mask[i, batch_size + i] = True
                
                neg_mask = ~pos_mask
                neg_mask.fill_diagonal_(False)
                
                pos_distances = dist_matrix[pos_mask][:batch_size]
                neg_distances = dist_matrix[neg_mask].view(2 * batch_size, -1)
                hard_neg_distances, _ = neg_distances.min(dim=1)[:batch_size]
                
                contrastive_loss = torch.mean(
                    F.relu(pos_distances - hard_neg_distances + self.margin)
                )
        else:
            contrastive_loss = torch.tensor(0.0, device=query_emb.device)
            pos_dist = torch.norm(query_emb - pos_emb, p=2, dim=1)
        
        # Total loss
        total_loss = (
            self.lambda_isotropy * isotropy_loss +
            self.lambda_predictive * predictive_loss +
            self.lambda_residual * residual_loss +
            self.lambda_contrastive * contrastive_loss
        )
        
        # Logging dict
        loss_dict = {
            'total': total_loss.item(),
            'isotropy': isotropy_loss.item(),
            'predictive': predictive_loss.item() if isinstance(predictive_loss, torch.Tensor) else 0.0,
            'residual': residual_loss.item() if isinstance(residual_loss, torch.Tensor) else 0.0,
            'contrastive': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0,
            'pos_dist_mean': pos_dist.mean().item(),
            'embedding_std': torch.std(all_emb).item(),
        }
        
        # Add delta stats if available
        if delta is not None:
            loss_dict['delta_mean'] = torch.mean(torch.abs(delta)).item()
            loss_dict['delta_max'] = torch.max(torch.abs(delta)).item()
        
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
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Encode
        query_emb = model(batch['queries'])
        pos_emb = model(batch['positives'])
        neg_emb = model(batch['negatives']) if batch['negatives'] else None
        
        # ResPred prediction with residual
        if model.predictor is not None:
            predicted_pos, delta = model.predict_with_residual(query_emb)
        else:
            predicted_pos, delta = None, None
        
        # Compute loss
        loss, loss_dict = criterion(
            query_emb, pos_emb,
            predicted_pos=predicted_pos,
            delta=delta,
            neg_emb=neg_emb
        )
        
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
            'iso': f"{loss_dict['isotropy']:.4f}",
            'pred': f"{loss_dict['predictive']:.4f}",
            'res': f"{loss_dict['residual']:.4f}"
        })
    
    # Average over batches
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    loss_components = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            query_emb = model(batch['queries'])
            pos_emb = model(batch['positives'])
            neg_emb = model(batch['negatives']) if batch['negatives'] else None
            
            if model.predictor is not None:
                predicted_pos, delta = model.predict_with_residual(query_emb)
            else:
                predicted_pos, delta = None, None
            
            loss, loss_dict = criterion(
                query_emb, pos_emb,
                predicted_pos=predicted_pos,
                delta=delta,
                neg_emb=neg_emb
            )
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description='Train ResPred model')
    
    # Data
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', default=None)
    
    # Model
    parser.add_argument('--base_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--output_dim', type=int, default=768)
    parser.add_argument('--freeze_base', action='store_true', default=True)
    parser.add_argument('--residual_scale_init', type=float, default=0.1)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Loss weights
    parser.add_argument('--lambda_isotropy', type=float, default=1.5)
    parser.add_argument('--lambda_predictive', type=float, default=1.2)
    parser.add_argument('--lambda_residual', type=float, default=0.01)
    parser.add_argument('--lambda_contrastive', type=float, default=0.0)
    
    # Output
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--log_file', default=None)
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    
    # Print configuration
    logger.info("="*80)
    logger.info("ResPred Training Configuration")
    logger.info("="*80)
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output dim: {args.output_dim}")
    logger.info(f"Freeze base: {args.freeze_base}")
    logger.info(f"Residual scale (α): {args.residual_scale_init}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")
    logger.info("Loss weights:")
    logger.info(f"  λ_isotropy: {args.lambda_isotropy}")
    logger.info(f"  λ_predictive: {args.lambda_predictive}")
    logger.info(f"  λ_residual: {args.lambda_residual}")
    logger.info(f"  λ_contrastive: {args.lambda_contrastive}")
    logger.info("="*80)
    logger.info("")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating ResPred model...")
    model = ResidualGaussianEncoder(
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=args.freeze_base,
        normalize_embeddings=False,
        use_predictor=True,
        residual_scale_init=args.residual_scale_init
    )
    model = model.to(device)
    
    # Create loss
    criterion = ResPredLoss(
        lambda_isotropy=args.lambda_isotropy,
        lambda_predictive=args.lambda_predictive,
        lambda_residual=args.lambda_residual,
        lambda_contrastive=args.lambda_contrastive,
        margin=1.0,
        use_stopgrad=True
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
        logger.info(f"  Isotropy: {train_components['isotropy']:.6f}")
        logger.info(f"  Predictive: {train_components['predictive']:.6f}")
        logger.info(f"  Residual: {train_components['residual']:.6f}")
        logger.info(f"  Embedding std: {train_components['embedding_std']:.4f}")
        if 'delta_mean' in train_components:
            logger.info(f"  Delta mean: {train_components['delta_mean']:.6f}")
            logger.info(f"  Delta max: {train_components['delta_max']:.6f}")
            logger.info(f"  Alpha: {model.predictor.get_alpha():.6f}")
        
        # Validate
        if val_loader:
            val_loss, val_components = evaluate(model, val_loader, criterion, device)
            logger.info(f"\nValidation Loss: {val_loss:.6f}")
            logger.info(f"  Isotropy: {val_components['isotropy']:.6f}")
            logger.info(f"  Predictive: {val_components['predictive']:.6f}")
            logger.info(f"  Residual: {val_components['residual']:.6f}")
            
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

