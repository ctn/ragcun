#!/usr/bin/env python3
"""
Training script for IsotropicGaussianEncoder with LeJEPA SIGReg loss.

This script trains the model using isotropic Gaussian embeddings with LeJEPA's
SIGReg loss for superior retrieval performance.

Usage:
    python scripts/train/isotropic.py --config config/train_config.yaml
    python scripts/train/isotropic.py --data_path data/processed/train.json --epochs 3
    python scripts/train/isotropic.py --help
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import IsotropicGaussianEncoder


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning.

    Expected format: List of dicts with 'query', 'positive', and optionally 'negative'.
    """

    def __init__(self, data_path: str):
        """
        Load training data from JSON file.

        Args:
            data_path: Path to JSON file with format:
                [{"query": "...", "positive": "...", "negative": "..."}, ...]
        """
        self.data_path = data_path
        logger.info(f"Loading data from {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} training examples")

        # Validate format
        for i, item in enumerate(self.data[:5]):
            if 'query' not in item or 'positive' not in item:
                raise ValueError(
                    f"Item {i} missing 'query' or 'positive' field. "
                    f"Expected format: {{'query': '...', 'positive': '...', 'negative': '...'}}"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'positive': item['positive'],
            'negative': item.get('negative', None)
        }


class SIGRegLoss(nn.Module):
    """
    LeJEPA SIGReg Loss for isotropic Gaussian embeddings.

    Combines three components:
    1. Contrastive loss: Pull positives closer, push negatives apart
    2. Isotropy loss: Encourage uniform distribution in embedding space
    3. Regularization loss: Prevent collapse and maintain variance

    Args:
        lambda_contrastive: Weight for contrastive loss (default: 1.0, set to 0.0 for pure isotropy)
        lambda_isotropy: Weight for isotropy loss (default: 1.0)
        lambda_reg: Weight for regularization loss (default: 0.1)
        margin: Margin for contrastive loss (default: 1.0)
        target_std: Target standard deviation for embeddings (default: 1.0)
    """

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_isotropy: float = 1.0,
        lambda_reg: float = 0.1,
        lambda_predictive: float = 0.0,
        margin: float = 1.0,
        target_std: float = 1.0,
        use_stopgrad: bool = True
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_isotropy = lambda_isotropy
        self.lambda_reg = lambda_reg
        self.lambda_predictive = lambda_predictive
        self.margin = margin
        self.target_std = target_std
        self.use_stopgrad = use_stopgrad

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: Optional[torch.Tensor] = None,
        predicted_pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SIGReg loss.

        Args:
            query_emb: Query embeddings (batch_size, dim)
            pos_emb: Positive embeddings (batch_size, dim)
            neg_emb: Negative embeddings (batch_size, dim) or None

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size = query_emb.shape[0]

        # 1. Contrastive Loss (Euclidean distance based)
        pos_dist = torch.norm(query_emb - pos_emb, p=2, dim=1)

        if neg_emb is not None:
            # With hard negatives
            neg_dist = torch.norm(query_emb - neg_emb, p=2, dim=1)
            contrastive_loss = torch.mean(
                F.relu(pos_dist - neg_dist + self.margin)
            )
        else:
            # Without hard negatives - use in-batch negatives
            # Compute pairwise distances
            all_emb = torch.cat([query_emb, pos_emb], dim=0)  # (2*batch_size, dim)

            # Distance matrix
            dist_matrix = torch.cdist(all_emb, all_emb, p=2)  # (2*batch, 2*batch)

            # Positive pairs: query[i] <-> pos[i]
            pos_mask = torch.zeros_like(dist_matrix, dtype=torch.bool)
            for i in range(batch_size):
                pos_mask[i, batch_size + i] = True
                pos_mask[batch_size + i, i] = True

            # Negative pairs: all other pairs
            neg_mask = ~pos_mask
            neg_mask.fill_diagonal_(False)  # Exclude self-comparison

            pos_distances = dist_matrix[pos_mask]
            neg_distances = dist_matrix[neg_mask].view(2 * batch_size, -1)

            # For each query, find hardest negative
            hard_neg_distances, _ = neg_distances.min(dim=1)

            # pos_distances contains both query->pos and pos->query pairs
            # We only need query->pos pairs (first batch_size elements)
            query_pos_distances = pos_distances[:batch_size]
            query_hard_neg_distances = hard_neg_distances[:batch_size]

            contrastive_loss = torch.mean(
                F.relu(query_pos_distances - query_hard_neg_distances + self.margin)
            )

        # 2. Isotropy Loss: Encourage uniform distribution
        # Measure deviation from isotropic Gaussian
        all_emb = torch.cat([query_emb, pos_emb], dim=0)

        # Covariance matrix
        mean = all_emb.mean(dim=0, keepdim=True)
        centered = all_emb - mean
        cov = (centered.T @ centered) / (all_emb.shape[0] - 1)

        # SIGReg: Isotropic covariance at CURRENT scale (scale-invariant)
        # This enforces isotropy without constraining variance
        variance = torch.var(all_emb)
        target_cov = torch.eye(cov.shape[0], device=cov.device) * variance

        # Frobenius norm of difference (normalized by dimension)
        isotropy_loss = torch.norm(cov - target_cov, p='fro') / cov.shape[0]

        # 3. Regularization Loss: Optional variance constraint
        std = torch.std(all_emb)
        reg_loss = (std - self.target_std) ** 2

        # 4. Predictive Loss (JEPA-style): Predict document from query
        if predicted_pos is not None and self.lambda_predictive > 0:
            # Stop-gradient on target to prevent collapse (JEPA standard)
            target = pos_emb.detach() if self.use_stopgrad else pos_emb
            predictive_loss = F.mse_loss(predicted_pos, target)
        else:
            predictive_loss = torch.tensor(0.0, device=query_emb.device)

        # Total loss
        total_loss = (
            self.lambda_contrastive * contrastive_loss +
            self.lambda_isotropy * isotropy_loss +
            self.lambda_reg * reg_loss +
            self.lambda_predictive * predictive_loss
        )

        # Return loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'contrastive': contrastive_loss.item(),
            'isotropy': isotropy_loss.item(),
            'regularization': reg_loss.item(),
            'predictive': predictive_loss.item() if isinstance(predictive_loss, torch.Tensor) else predictive_loss,
            'pos_dist_mean': pos_dist.mean().item(),
            'embedding_std': std.item(),
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
        'contrastive': 0,
        'isotropy': 0,
        'regularization': 0,
        'pos_dist_mean': 0,
        'embedding_std': 0
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                # Encode texts
                query_emb = model(batch['queries'])
                pos_emb = model(batch['positives'])
                neg_emb = model(batch['negatives']) if batch['negatives'] else None

                # Predict document from query (JEPA-style)
                predicted_pos = model.predictor(query_emb) if model.predictor is not None else None

                # Compute loss
                loss, loss_dict = criterion(query_emb, pos_emb, neg_emb, predicted_pos=predicted_pos)
        else:
            # Encode texts
            query_emb = model(batch['queries'])
            pos_emb = model(batch['positives'])
            neg_emb = model(batch['negatives']) if batch['negatives'] else None

            # Predict document from query (JEPA-style)
            predicted_pos = model.predictor(query_emb) if model.predictor is not None else None

            # Compute loss
            loss, loss_dict = criterion(query_emb, pos_emb, neg_emb, predicted_pos=predicted_pos)

        # Backward pass with optional mixed precision
        if use_amp:
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]

        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_dist': f"{loss_dict['pos_dist_mean']:.3f}",
                'std': f"{loss_dict['embedding_std']:.3f}"
            })

    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'total': avg_loss, **loss_components}


def validate(
    model: IsotropicGaussianEncoder,
    dataloader: DataLoader,
    criterion: SIGRegLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0
    loss_components = {
        'contrastive': 0,
        'isotropy': 0,
        'regularization': 0,
        'pos_dist_mean': 0,
        'embedding_std': 0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Encode texts
            query_emb = model(batch['queries'])
            pos_emb = model(batch['positives'])
            neg_emb = model(batch['negatives']) if batch['negatives'] else None

            # Compute loss
            loss, loss_dict = criterion(query_emb, pos_emb, neg_emb)

            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]

    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {'total': avg_loss, **loss_components}


def main():
    parser = argparse.ArgumentParser(description="Train IsotropicGaussianEncoder")

    # Data arguments
    parser.add_argument('--train_data', type=str, default='data/processed/train.json',
                        help='Path to training data JSON')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data JSON')

    # Model arguments
    parser.add_argument('--output_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Base model to use (default: google/embeddinggemma-300m)')
    parser.add_argument('--freeze_base', action='store_true',
                        help='Freeze entire base encoder (smart hybrid - train projection only)')
    parser.add_argument('--freeze_early_layers', action='store_true',
                        help='Freeze first 4 transformer layers')
    parser.add_argument('--no_normalize_embeddings', action='store_true',
                        help='Disable normalization of base model embeddings (use raw embeddings)')
    parser.add_argument('--base_learning_rate', type=float, default=None,
                        help='Learning rate for base encoder (default: same as --learning_rate)')
    parser.add_argument('--projection_learning_rate', type=float, default=None,
                        help='Learning rate for projection layer (default: same as --learning_rate)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')

    # Loss arguments
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='Weight for contrastive loss (set to 0.0 for pure isotropy)')
    parser.add_argument('--lambda_isotropy', type=float, default=1.0,
                        help='Weight for isotropy loss')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                        help='Weight for regularization loss')
    parser.add_argument('--lambda_predictive', type=float, default=0.0,
                        help='Weight for predictive loss (JEPA-style, set >0 to enable)')
    parser.add_argument('--use_predictor', action='store_true',
                        help='Use predictor network (JEPA-style: query → document)')
    parser.add_argument('--no_stopgrad', action='store_true',
                        help='Disable stop-gradient on target (default: enabled for JEPA)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for contrastive loss')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoint_epoch_N.pt)')
    parser.add_argument('--load_weights_only', action='store_true',
                        help='When resuming, only load model weights (not optimizer/scheduler state)')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (FP16)')

    args = parser.parse_args()

    # Validate input files exist (fail fast)
    train_data_path = Path(args.train_data)
    if not train_data_path.exists():
        logger.error(f"❌ Training data not found: {args.train_data}")
        logger.error(f"   Please prepare data first using: python scripts/prepare_data.py")
        sys.exit(1)
    
    if args.val_data:
        val_data_path = Path(args.val_data)
        if not val_data_path.exists():
            logger.error(f"❌ Validation data not found: {args.val_data}")
            logger.error(f"   Either remove --val_data or create the file")
            sys.exit(1)
    
    logger.info("✅ Input files validated")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config_path = output_dir / 'train_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data
    logger.info("Loading datasets...")
    train_dataset = ContrastiveDataset(args.train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues with transformers
    )

    val_loader = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = ContrastiveDataset(args.val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

    # Initialize model
    logger.info("Initializing model...")
    model = IsotropicGaussianEncoder(
        output_dim=args.output_dim,
        base_model=args.base_model,
        freeze_base=args.freeze_base,
        freeze_early_layers=args.freeze_early_layers,
        normalize_embeddings=not args.no_normalize_embeddings,
        use_predictor=args.use_predictor
    )
    model = model.to(device)

    # Initialize loss
    criterion = SIGRegLoss(
        lambda_contrastive=args.lambda_contrastive,
        lambda_isotropy=args.lambda_isotropy,
        lambda_reg=args.lambda_reg,
        lambda_predictive=args.lambda_predictive,
        margin=args.margin,
        use_stopgrad=not args.no_stopgrad
    )

    # Initialize optimizer with differential learning rates if specified
    param_groups_dict = model.get_trainable_parameters()
    
    base_lr = args.base_learning_rate if args.base_learning_rate is not None else args.learning_rate
    proj_lr = args.projection_learning_rate if args.projection_learning_rate is not None else args.learning_rate
    
    use_diff_lr = (
        base_lr != proj_lr and
        len(param_groups_dict['base']) > 0
    )
    
    if use_diff_lr:
        logger.info("Using differential learning rates:")
        logger.info(f"  Base encoder: {base_lr}")
        logger.info(f"  Projection: {proj_lr}")
        
        param_groups = [
            {'params': param_groups_dict['base'], 'lr': base_lr},
            {'params': param_groups_dict['projection'], 'lr': proj_lr}
        ]
        # Add predictor if it exists
        if len(param_groups_dict['predictor']) > 0:
            param_groups.append({'params': param_groups_dict['predictor'], 'lr': proj_lr})
            logger.info(f"  Predictor: {proj_lr}")
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_steps
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_steps]
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        logger.info("✅ Mixed precision training enabled (FP16)")

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            logger.info(f"🔄 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state (allow missing/extra keys for flexibility)
            model_state = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out keys that don't match (e.g., predictor if not using it)
            filtered_state = {k: v for k, v in model_state.items() if k in model_dict}
            missing_keys = set(model_dict.keys()) - set(filtered_state.keys())
            extra_keys = set(model_state.keys()) - set(filtered_state.keys())
            
            if missing_keys:
                logger.warning(f"⚠️  Missing keys (will use random init): {list(missing_keys)[:5]}...")
            if extra_keys:
                logger.info(f"ℹ️  Ignoring extra keys from checkpoint: {list(extra_keys)[:5]}...")
            
            model_dict.update(filtered_state)
            model.load_state_dict(model_dict)
            logger.info("✅ Loaded model weights")
            
            if args.load_weights_only:
                # Fine-tuning mode: only load weights, start fresh
                logger.info("   Fine-tuning mode: Starting from epoch 1 with loaded weights")
                start_epoch = 1
                best_val_loss = float('inf')
            else:
                # Full resume: load optimizer and scheduler state
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("✅ Loaded optimizer state")
                
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("✅ Loaded scheduler state")
                
                # Get starting epoch
                start_epoch = checkpoint.get('epoch', 1) + 1
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                
                logger.info(f"✅ Resumed from epoch {checkpoint.get('epoch', 1)}")
                logger.info(f"   Continuing from epoch {start_epoch} to {args.epochs}")
        else:
            logger.warning(f"⚠️  Checkpoint not found: {checkpoint_path}")
            logger.warning("   Starting training from scratch")

    # Training loop
    logger.info("Starting training...")
    if not args.resume_from:
        best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.log_interval, scaler
        )

        # Step learning rate scheduler
        scheduler.step()

        # Log training metrics
        logger.info(f"\nTraining metrics:")
        logger.info(f"  Total Loss: {train_losses['total']:.4f}")
        logger.info(f"  Contrastive Loss: {train_losses['contrastive']:.4f}")
        logger.info(f"  Isotropy Loss: {train_losses['isotropy']:.4f}")
        logger.info(f"  Regularization Loss: {train_losses['regularization']:.4f}")
        logger.info(f"  Pos Distance (mean): {train_losses['pos_dist_mean']:.3f}")
        logger.info(f"  Embedding Std: {train_losses['embedding_std']:.3f}")

        # Validate
        if val_loader:
            val_losses = validate(model, val_loader, criterion, device)

            logger.info(f"\nValidation metrics:")
            logger.info(f"  Total Loss: {val_losses['total']:.4f}")
            logger.info(f"  Contrastive Loss: {val_losses['contrastive']:.4f}")
            logger.info(f"  Isotropy Loss: {val_losses['isotropy']:.4f}")

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': vars(args)
                }, best_path)
                logger.info(f"  ✅ Saved best model to {best_path}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses['total'],
                'config': vars(args)
            }, checkpoint_path)
            logger.info(f"  💾 Saved checkpoint to {checkpoint_path}")

        # Clear GPU cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': vars(args)
    }, final_path)
    logger.info(f"\n✅ Training complete! Final model saved to {final_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Training Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Final training loss: {train_losses['total']:.4f}")
    if val_loader:
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved in: {output_dir}")


if __name__ == '__main__':
    main()
