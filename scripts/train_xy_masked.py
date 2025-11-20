#!/usr/bin/env python3
"""
Self-supervised JEPA training using X/Y masked text pairs.

This script trains JEPA using X/Y pairs where:
- X: Original text
- Y: Masked version of the same text
- Predictor learns: P(embed(X)) ≈ embed(Y)

The base model is frozen, so only projection and predictor are trained.

Usage:
    python scripts/train_xy_masked.py \
        --input_xy_pairs data/processed/xy_masked_documents.json \
        --batch_size 32 \
        --epochs 3 \
        --lambda_predictive 1.2 \
        --lambda_isotropy 1.5 \
        --output_dir checkpoints/jepa_xy_masked
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import IsotropicGaussianEncoder
from scripts.train import SIGRegLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_xy_masked.log')
    ]
)
logger = logging.getLogger(__name__)


class XYMaskedDataset(Dataset):
    """Dataset for X/Y masked text pairs."""
    
    def __init__(self, xy_pairs: List[Dict[str, str]]):
        """
        Initialize X/Y masked dataset.
        
        Args:
            xy_pairs: List of dicts with 'x' (original) and 'y' (masked) keys
        """
        self.xy_pairs = xy_pairs
        logger.info(f"Loaded {len(self.xy_pairs):,} X/Y pairs")
    
    def __len__(self):
        return len(self.xy_pairs)
    
    def __getitem__(self, idx):
        pair = self.xy_pairs[idx]
        return {
            'x': pair['x'],  # Original text
            'y': pair['y']   # Masked text
        }


def collate_fn_xy_masked(batch):
    """Collate function for X/Y masked DataLoader."""
    x_texts = [item['x'] for item in batch]
    y_texts = [item['y'] for item in batch]
    
    return {
        'x': x_texts,
        'y': y_texts
    }


def train_epoch(
    model: IsotropicGaussianEncoder,
    dataloader: DataLoader,
    criterion: SIGRegLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    loss_components = {
        'predictive': 0,
        'isotropy': 0,
        'regularization': 0,
        'embedding_std': 0
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # Encode X (original) and Y (masked)
        x_emb = model(batch['x'])
        y_emb = model(batch['y'])
        
        # Predict Y from X (JEPA-style)
        predicted_y = model.predictor(x_emb) if model.predictor is not None else None
        
        # Compute loss (no contrastive, no negatives)
        loss, loss_dict = criterion(
            query_emb=x_emb,
            pos_emb=y_emb,
            neg_emb=None,  # No negatives in self-supervised
            predicted_pos=predicted_y
        )
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        
        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            # Use scientific notation for very small values, regular for larger ones
            def format_loss(val):
                if val < 0.0001:
                    return f"{val:.2e}"
                else:
                    return f"{val:.4f}"
            
            pbar.set_postfix({
                'loss': format_loss(loss.item()),
                'pred': format_loss(loss_dict.get('predictive', 0)),
                'iso': format_loss(loss_dict.get('isotropy', 0)),
                'std': f"{loss_dict.get('embedding_std', 0):.3f}"
            })
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return {'total': avg_loss, **loss_components}


def main():
    parser = argparse.ArgumentParser(description="Self-supervised JEPA training with X/Y masked pairs")
    
    # Data arguments
    parser.add_argument('--input_xy_pairs', type=str, required=True,
                        help='Path to JSON file with X/Y pairs [{x: text, y: masked_text}]')
    
    # Model arguments
    parser.add_argument('--base_model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='Base encoder model (default: all-mpnet-base-v2)')
    parser.add_argument('--output_dim', type=int, default=768,
                        help='Output embedding dimension (default: 768)')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                        help='Freeze base model (default: True)')
    parser.add_argument('--use_predictor', action='store_true', default=True,
                        help='Use predictor network (default: True)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--base_learning_rate', type=float, default=2e-5,
                        help='Learning rate for base model (if not frozen)')
    parser.add_argument('--projection_learning_rate', type=float, default=5e-4,
                        help='Learning rate for projection and predictor (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval (default: 10)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers (default: 0)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    
    # Loss arguments
    parser.add_argument('--lambda_contrastive', type=float, default=0.0,
                        help='Weight for contrastive loss (default: 0.0, no contrastive)')
    parser.add_argument('--lambda_isotropy', type=float, default=1.5,
                        help='Weight for isotropy loss (default: 1.5)')
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help='Weight for regularization loss (default: 0.0)')
    parser.add_argument('--lambda_predictive', type=float, default=1.2,
                        help='Weight for predictive loss (default: 1.2)')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin for contrastive loss (default: 0.1)')
    parser.add_argument('--use_stopgrad', action='store_true', default=True,
                        help='Use stop-gradient on target (default: True)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load X/Y pairs
    logger.info(f"Loading X/Y pairs from {args.input_xy_pairs}")
    with open(args.input_xy_pairs, 'r', encoding='utf-8') as f:
        xy_pairs = json.load(f)
    
    if not isinstance(xy_pairs, list):
        raise ValueError("Input must be a JSON array of X/Y pairs")
    
    if len(xy_pairs) == 0:
        raise ValueError("No X/Y pairs found in input file")
    
    # Validate format
    if not all('x' in pair and 'y' in pair for pair in xy_pairs[:10]):
        raise ValueError("Each pair must have 'x' and 'y' keys")
    
    logger.info(f"Loaded {len(xy_pairs):,} X/Y pairs")
    
    # Create dataset
    dataset = XYMaskedDataset(xy_pairs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_xy_masked,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = IsotropicGaussianEncoder(
        output_dim=args.output_dim,
        base_model=args.base_model,
        freeze_base=args.freeze_base,
        use_predictor=args.use_predictor
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize loss
    criterion = SIGRegLoss(
        lambda_contrastive=args.lambda_contrastive,
        lambda_isotropy=args.lambda_isotropy,
        lambda_reg=args.lambda_reg,
        lambda_predictive=args.lambda_predictive,
        margin=args.margin,
        use_stopgrad=args.use_stopgrad
    )
    
    # Initialize optimizer (only trainable params: projection + predictor)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=args.projection_learning_rate,
        weight_decay=args.weight_decay
    )
    
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Data-to-parameter ratio: {len(xy_pairs) / trainable_params:.4f}")
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop
    best_loss = float('inf')
    
    logger.info("Starting training...")
    logger.info(f"Training on {len(xy_pairs):,} X/Y pairs")
    logger.info(f"Loss weights: Iso={args.lambda_isotropy}, Pred={args.lambda_predictive}, Cont={args.lambda_contrastive}, Reg={args.lambda_reg}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*80}")
        
        # Train
        train_losses = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch,
            args.log_interval, scaler
        )
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_losses['total']:.4f}")
        logger.info(f"  Predictive: {train_losses.get('predictive', 0):.4f}")
        logger.info(f"  Isotropy: {train_losses.get('isotropy', 0):.4f}")
        logger.info(f"  Regularization: {train_losses.get('regularization', 0):.4f}")
        logger.info(f"  Embedding std: {train_losses.get('embedding_std', 0):.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses['total'],
                'config': vars(args)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if train_losses['total'] < best_loss:
            best_loss = train_losses['total']
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': vars(args)
            }, best_model_path)
            logger.info(f"Saved best model (loss: {best_loss:.4f}) to {best_model_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

