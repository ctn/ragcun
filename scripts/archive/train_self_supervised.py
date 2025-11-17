#!/usr/bin/env python3
"""
Self-supervised JEPA training on documents only (no query-document pairs).

This script trains JEPA using only documents by splitting them into parts
and learning to predict one part from another. The base model is frozen,
so it already provides semantic understanding.

Usage:
    python scripts/train_self_supervised.py \
        --input_documents data/raw/msmarco_documents.json \
        --document_split_strategy half \
        --batch_size 32 \
        --epochs 2 \
        --lambda_predictive 1.2 \
        --lambda_isotropy 1.5
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import GaussianEmbeddingGemma
from scripts.train import SIGRegLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_self_supervised.log')
    ]
)
logger = logging.getLogger(__name__)


def split_document_half(doc: str, split_ratio: float = 0.5) -> Tuple[str, str]:
    """Split document in half."""
    words = doc.split()
    if len(words) <= 1:
        return doc, doc
    
    split_idx = int(len(words) * split_ratio)
    part1 = ' '.join(words[:split_idx])
    part2 = ' '.join(words[split_idx:])
    return part1, part2


def split_document_random(doc: str, split_ratio: float = 0.5) -> Tuple[str, str]:
    """Split document at random point."""
    words = doc.split()
    if len(words) <= 1:
        return doc, doc
    
    # Random split between 0.3 and 0.7
    min_ratio = max(0.3, split_ratio - 0.2)
    max_ratio = min(0.7, split_ratio + 0.2)
    actual_ratio = random.uniform(min_ratio, max_ratio)
    
    split_idx = int(len(words) * actual_ratio)
    part1 = ' '.join(words[:split_idx])
    part2 = ' '.join(words[split_idx:])
    return part1, part2


def split_document_prefix_suffix(doc: str, prefix_ratio: float = 0.3) -> Tuple[str, str]:
    """Split document into prefix and suffix."""
    words = doc.split()
    if len(words) <= 1:
        return doc, doc
    
    prefix_idx = int(len(words) * prefix_ratio)
    part1 = ' '.join(words[:prefix_idx])
    part2 = ' '.join(words[prefix_idx:])
    return part1, part2


class SelfSupervisedDataset(Dataset):
    """Dataset for self-supervised JEPA training on documents only."""
    
    def __init__(
        self,
        documents: List[str],
        split_strategy: str = 'half',
        split_ratio: float = 0.5,
        min_length: int = 50,
        max_length: int = 2000
    ):
        """
        Initialize self-supervised dataset.
        
        Args:
            documents: List of document strings
            split_strategy: 'half', 'random', or 'prefix_suffix'
            split_ratio: Ratio for splitting (0.5 = 50/50)
            min_length: Minimum document length
            max_length: Maximum document length (truncate if exceeded)
        """
        self.documents = []
        self.split_strategy = split_strategy
        self.split_ratio = split_ratio
        
        # Filter and process documents
        for doc in documents:
            # Filter by length
            if len(doc) < min_length:
                continue
            
            # Truncate if too long
            if len(doc) > max_length:
                doc = doc[:max_length]
            
            self.documents.append(doc)
        
        logger.info(f"Loaded {len(self.documents)} documents (after filtering)")
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Split document based on strategy
        if self.split_strategy == 'half':
            part1, part2 = split_document_half(doc, self.split_ratio)
        elif self.split_strategy == 'random':
            part1, part2 = split_document_random(doc, self.split_ratio)
        elif self.split_strategy == 'prefix_suffix':
            part1, part2 = split_document_prefix_suffix(doc, self.split_ratio)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        
        return {
            'part1': part1,
            'part2': part2
        }


def collate_fn_self_supervised(batch):
    """Collate function for self-supervised DataLoader."""
    part1s = [item['part1'] for item in batch]
    part2s = [item['part2'] for item in batch]
    
    return {
        'part1': part1s,
        'part2': part2s
    }


def train_epoch(
    model: GaussianEmbeddingGemma,
    dataloader: DataLoader,
    criterion: SIGRegLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
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
        
        # Encode both parts
        part1_emb = model(batch['part1'])
        part2_emb = model(batch['part2'])
        
        # Predict part2 from part1 (JEPA-style)
        predicted_part2 = model.predictor(part1_emb) if model.predictor is not None else None
        
        # Compute loss (no contrastive, no negatives)
        loss, loss_dict = criterion(
            query_emb=part1_emb,
            pos_emb=part2_emb,
            neg_emb=None,  # No negatives in self-supervised
            predicted_pos=predicted_part2
        )
        
        # Backward pass
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
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pred': f"{loss_dict.get('predictive', 0):.4f}",
                'iso': f"{loss_dict.get('isotropy', 0):.4f}",
                'std': f"{loss_dict.get('embedding_std', 0):.3f}"
            })
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return {'total': avg_loss, **loss_components}


def main():
    parser = argparse.ArgumentParser(description="Self-supervised JEPA training")
    
    # Data arguments
    parser.add_argument('--input_documents', type=str, required=True,
                        help='Path to JSON file with array of documents')
    parser.add_argument('--document_split_strategy', type=str, default='half',
                        choices=['half', 'random', 'prefix_suffix'],
                        help='How to split documents (default: half)')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='Split ratio for first part (default: 0.5)')
    parser.add_argument('--min_document_length', type=int, default=50,
                        help='Minimum document length (default: 50)')
    parser.add_argument('--max_document_length', type=int, default=2000,
                        help='Maximum document length (default: 2000)')
    
    # Model arguments
    parser.add_argument('--base_model', type=str,
                        default='sentence-transformers/all-mpnet-base-v2',
                        help='Base model (default: sentence-transformers/all-mpnet-base-v2)')
    parser.add_argument('--output_dim', type=int, default=768,
                        help='Output embedding dimension (default: 768)')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                        help='Freeze base encoder (default: True, required for self-supervised)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs (default: 2)')
    parser.add_argument('--projection_learning_rate', type=float, default=5e-4,
                        help='Learning rate for projection (default: 5e-4)')
    parser.add_argument('--predictor_learning_rate', type=float, default=5e-4,
                        help='Learning rate for predictor (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    
    # Loss arguments
    parser.add_argument('--lambda_predictive', type=float, default=1.2,
                        help='Weight for predictive loss (default: 1.2)')
    parser.add_argument('--lambda_isotropy', type=float, default=1.5,
                        help='Weight for isotropy loss (default: 1.5)')
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help='Weight for regularization loss (default: 0.0)')
    parser.add_argument('--use_stopgrad', action='store_true', default=True,
                        help='Use stop-gradient on target (default: True)')
    
    # Validation (optional)
    parser.add_argument('--val_data', type=str, default=None,
                        help='Optional: Path to query-doc pairs for validation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/jepa_self_supervised',
                        help='Output directory (default: checkpoints/jepa_self_supervised)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Validate input
    input_path = Path(args.input_documents)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {args.input_documents}")
        sys.exit(1)
    
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
    
    # Load documents
    logger.info(f"Loading documents from {args.input_documents}")
    with open(args.input_documents, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"Loaded {len(documents):,} documents")
    
    # Create dataset
    dataset = SelfSupervisedDataset(
        documents=documents,
        split_strategy=args.document_split_strategy,
        split_ratio=args.split_ratio,
        min_length=args.min_document_length,
        max_length=args.max_document_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_self_supervised,
        num_workers=0
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = GaussianEmbeddingGemma(
        output_dim=args.output_dim,
        base_model=args.base_model,
        freeze_base=args.freeze_base,
        use_predictor=True
    )
    model = model.to(device)
    
    # Initialize loss
    criterion = SIGRegLoss(
        lambda_contrastive=0.0,  # No contrastive in self-supervised
        lambda_isotropy=args.lambda_isotropy,
        lambda_reg=args.lambda_reg,
        lambda_predictive=args.lambda_predictive,
        use_stopgrad=args.use_stopgrad
    )
    
    # Initialize optimizer (only trainable params: projection + predictor)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.projection_learning_rate,
        weight_decay=args.weight_decay
    )
    
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        train_metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, args.log_interval
        )
        
        logger.info(f"Epoch {epoch} - Loss: {train_metrics['total']:.4f}")
        logger.info(f"  Predictive: {train_metrics['predictive']:.4f}")
        logger.info(f"  Isotropy: {train_metrics['isotropy']:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_metrics['total'],
            'config': vars(args)
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if train_metrics['total'] < best_loss:
            best_loss = train_metrics['total']
            best_model_path = output_dir / 'best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model (loss: {best_loss:.4f})")
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"\n✅ Training complete!")
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Best model saved to {output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()


