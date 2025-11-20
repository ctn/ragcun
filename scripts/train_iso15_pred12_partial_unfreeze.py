#!/usr/bin/env python3
"""
Train ISO15_PRED12 with PARTIAL UNFREEZING of top encoder layers.

Strategy:
- Freeze layers 0-7 (keep general language understanding)
- Unfreeze layers 8-11 (adapt to query-document matching)
- Train projection + predictor as before
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import IsotropicGaussianEncoder
from scripts.archive.train import train_model


def unfreeze_top_layers(model, num_layers_to_unfreeze=4):
    """
    Unfreeze the top N transformer layers in MPNet.
    
    MPNet has 12 layers total (0-11).
    To unfreeze top 4: unfreeze layers 8-11, keep 0-7 frozen.
    """
    print(f"\n{'='*60}")
    print(f"PARTIAL UNFREEZING: Top {num_layers_to_unfreeze} layers")
    print(f"{'='*60}\n")
    
    # Access the transformer model inside SentenceTransformer
    # SentenceTransformer wraps: tokenizer, transformer, pooling
    # The actual transformer is in model.base[0] or model.base._first_module()
    
    transformer_model = None
    if hasattr(model.base, '_modules') and '0' in model.base._modules:
        transformer_model = model.base._modules['0'].auto_model
    elif hasattr(model.base, '_first_module'):
        transformer_model = model.base._first_module().auto_model
    
    if transformer_model is None:
        print("❌ Could not find transformer model!")
        return
    
    # Get encoder layers
    if hasattr(transformer_model, 'encoder') and hasattr(transformer_model.encoder, 'layer'):
        encoder_layers = transformer_model.encoder.layer
        total_layers = len(encoder_layers)
        
        print(f"Found {total_layers} encoder layers")
        
        # First, ensure everything is frozen
        for param in model.base.parameters():
            param.requires_grad = False
        
        # Unfreeze top N layers
        layers_to_unfreeze = list(range(total_layers - num_layers_to_unfreeze, total_layers))
        
        trainable_params = 0
        frozen_params = 0
        
        for layer_idx in layers_to_unfreeze:
            for param in encoder_layers[layer_idx].parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        
        for param in model.base.parameters():
            if not param.requires_grad:
                frozen_params += param.numel()
        
        print(f"\n✅ Unfroze layers: {layers_to_unfreeze}")
        print(f"✅ Kept frozen: layers 0-{total_layers - num_layers_to_unfreeze - 1}")
        print(f"\nParameter counts:")
        print(f"  Encoder trainable: {trainable_params:,}")
        print(f"  Encoder frozen: {frozen_params:,}")
        
    else:
        print("❌ Could not access encoder layers!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='data/processed/msmarco_full')
    parser.add_argument('--val_data', default='data/processed/msmarco_smoke')
    parser.add_argument('--output_dir', default='checkpoints/iso15_pred12_partial_unfreeze')
    parser.add_argument('--base_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--output_dim', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr_encoder', type=float, default=5e-6, help='Learning rate for unfrozen encoder layers')
    parser.add_argument('--lr_heads', type=float, default=1e-3, help='Learning rate for projection/predictor')
    parser.add_argument('--num_layers_unfreeze', type=int, default=4, help='Number of top layers to unfreeze')
    parser.add_argument('--lambda_isotropy', type=float, default=1.5)
    parser.add_argument('--lambda_predictive', type=float, default=1.2)
    parser.add_argument('--lambda_contrastive', type=float, default=0.0)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("ISO15_PRED12 with PARTIAL UNFREEZING")
    print(f"{'='*80}\n")
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Base model: {args.base_model}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR (encoder): {args.lr_encoder}")
    print(f"LR (heads): {args.lr_heads}")
    print(f"Layers to unfreeze: Top {args.num_layers_unfreeze}")
    print(f"\nLoss weights:")
    print(f"  Isotropy: {args.lambda_isotropy}")
    print(f"  Predictive: {args.lambda_predictive}")
    print(f"  Contrastive: {args.lambda_contrastive}")
    print()
    
    # Create model (initially frozen)
    print("Creating model...")
    model = IsotropicGaussianEncoder(
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=True,  # Start frozen
        normalize_embeddings=False,
        use_predictor=True
    )
    
    # Partially unfreeze
    unfreeze_top_layers(model, args.num_layers_unfreeze)
    
    # Create parameter groups with different learning rates
    encoder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'base' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
    
    param_groups = [
        {'params': encoder_params, 'lr': args.lr_encoder},
        {'params': head_params, 'lr': args.lr_heads}
    ]
    
    print(f"\nOptimizer setup:")
    print(f"  Encoder params: {sum(p.numel() for p in encoder_params):,} @ {args.lr_encoder}")
    print(f"  Head params: {sum(p.numel() for p in head_params):,} @ {args.lr_heads}")
    print(f"  Ratio: {args.lr_heads / args.lr_encoder:.0f}x higher LR for heads")
    
    # Train using existing training loop
    # (We'll need to modify this to use param_groups)
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    # Note: This requires modifying train.py to accept param_groups
    # For now, just show the setup
    print("⚠️  Training loop needs to be adapted for differential learning rates")
    print("    Use the param_groups defined above in the optimizer")
    

if __name__ == '__main__':
    main()

