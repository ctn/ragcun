#!/usr/bin/env python3
"""
Training script for MPNet-LeJEPA architecture.

Pipeline:
1. Frozen base encoder (MPNet)
2. Trainable projection heads (online + target via EMA)
3. Trainable predictor (online only)
4. LeJEPA loss (predictive + SIGReg isotropy)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.mpnet_lejepa import MPNetLeJEPA
from ragcun.lejepa_loss import LeJEPALoss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryDocumentDataset(Dataset):
    """Dataset for query-document pairs."""
    
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'positive': item['positive'],
            'negative': item.get('negative', None)
        }


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


def train_epoch(
    model: MPNetLeJEPA,
    dataloader: DataLoader,
    criterion: LeJEPALoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    ema_decay: float = 0.999,
    log_interval: int = 10,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_predictive_loss = 0
    total_sigreg_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # Forward pass
        # Online branch: queries
        # Target branch: positives
        output = model(batch['queries'], batch['positives'])
        p_online = output['p_online']  # Predictor output
        z_target = output['z_target']  # Target embedding (stop-gradient)
        
        # Compute LeJEPA loss (predictive + SIGReg)
        loss, loss_dict = criterion(
            p_online=p_online,
            z_target=z_target,
            embeddings=p_online  # Use predictor output for SIGReg
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
        
        # Update target network via EMA
        model.update_target_network()
        
        # Accumulate losses
        total_loss += loss.item()
        total_predictive_loss += loss_dict['predictive']
        total_sigreg_loss += loss_dict['sigreg']
        num_batches += 1
        
        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pred': f"{loss_dict['predictive']:.4f}",
                'sigreg': f"{loss_dict['sigreg']:.4f}"
            })
    
    # Average losses
    avg_loss = total_loss / num_batches
    avg_predictive = total_predictive_loss / num_batches
    avg_sigreg = total_sigreg_loss / num_batches
    
    return {
        'total': avg_loss,
        'predictive': avg_predictive,
        'sigreg': avg_sigreg
    }


def validate(
    model: MPNetLeJEPA,
    dataloader: DataLoader,
    criterion: LeJEPALoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_predictive_loss = 0
    total_sigreg_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Forward pass
            output = model(batch['queries'], batch['positives'])
            p_online = output['p_online']
            z_target = output['z_target']
            
            # Compute LeJEPA loss
            loss, loss_dict = criterion(
                p_online=p_online,
                z_target=z_target,
                embeddings=p_online
            )
            
            total_loss += loss.item()
            total_predictive_loss += loss_dict['predictive']
            total_sigreg_loss += loss_dict['sigreg']
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_predictive = total_predictive_loss / num_batches
    avg_sigreg = total_sigreg_loss / num_batches
    
    return {
        'total': avg_loss,
        'predictive': avg_predictive,
        'sigreg': avg_sigreg
    }


def main():
    parser = argparse.ArgumentParser(description="Train MPNet-LeJEPA model")
    
    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data JSON file')
    
    # Model arguments
    parser.add_argument('--base_model', type=str,
                        default='sentence-transformers/all-mpnet-base-v2',
                        help='Base model name')
    parser.add_argument('--proj_dim', type=int, default=256,
                        help='Projection dimension')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate for target network')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                        help='Freeze base encoder (default: True)')
    
    # Loss arguments
    parser.add_argument('--lambda_predictive', type=float, default=1.0,
                        help='Weight for predictive loss')
    parser.add_argument('--lambda_sigreg', type=float, default=1.0,
                        help='Weight for SIGReg isotropy loss')
    parser.add_argument('--num_slices', type=int, default=1000,
                        help='Number of random slices for SIGReg')
    parser.add_argument('--t_max', type=float, default=5.0,
                        help='Maximum integration point for Epps-Pulley')
    parser.add_argument('--n_points', type=int, default=21,
                        help='Number of integration points for Epps-Pulley')
    parser.add_argument('--use_stopgrad', action='store_true', default=True,
                        help='Use stop-gradient on target (default: True)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints/mpnet_lejepa',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (FP16)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = QueryDocumentDataset(args.train_data)
    val_dataset = QueryDocumentDataset(args.val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = MPNetLeJEPA(
        base_model=args.base_model,
        proj_dim=args.proj_dim,
        ema_decay=args.ema_decay,
        freeze_base=args.freeze_base
    ).to(device)
    
    # Loss function (LeJEPA)
    criterion = LeJEPALoss(
        lambda_predictive=args.lambda_predictive,
        lambda_sigreg=args.lambda_sigreg,
        num_slices=args.num_slices,
        t_max=args.t_max,
        n_points=args.n_points,
        use_stopgrad=args.use_stopgrad
    )
    
    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            ema_decay=args.ema_decay,
            log_interval=args.log_interval,
            scaler=scaler
        )
        
        logger.info(f"Train Loss: {train_metrics['total']:.4f}")
        logger.info(f"  Predictive: {train_metrics['predictive']:.4f}")
        logger.info(f"  SIGReg: {train_metrics['sigreg']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['total']:.4f}")
        logger.info(f"  Predictive: {val_metrics['predictive']:.4f}")
        logger.info(f"  SIGReg: {val_metrics['sigreg']:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total'],
                'config': {
                    'base_model': args.base_model,
                    'proj_dim': args.proj_dim,
                    'ema_decay': args.ema_decay,
                    'freeze_base': args.freeze_base
                }
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'base_model': args.base_model,
                    'proj_dim': args.proj_dim,
                    'ema_decay': args.ema_decay,
                    'freeze_base': args.freeze_base
                }
            }, best_model_path)
            logger.info(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
    
    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()

