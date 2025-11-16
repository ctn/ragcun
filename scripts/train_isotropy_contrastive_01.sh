#!/bin/bash
# Train with Small Contrastive Component + Isotropy
# This tests if small contrastive loss (0.1) preserves semantics while still regularizing

set -e

echo "============================================="
echo "Training: Isotropy + Small Contrastive (λ_contrastive=0.1)"
echo "============================================="
echo ""
echo "Strategy: λ_contrastive=0.1, λ_isotropy=1.0, λ_reg=0.0"
echo "  • Base Model: sentence-transformers/all-mpnet-base-v2 (frozen)"
echo "  • Training Data: data/processed/msmarco_smoke/train.json (10K examples)"
echo "  • Validation Data: data/processed/msmarco_smoke/dev.json (1K examples)"
echo "  • Output Dim: 512"
echo "  • Batch size: 16"
echo "  • Epochs: 2"
echo "  • Margin: 0.1"
echo "  • Projection LR: 5e-4 (lower for stability)"
echo "  • Expected time: ~15-20 minutes"
echo ""

python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 512 \
    --batch_size 16 \
    --epochs 2 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_contrastive 0.1 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --margin 0.1 \
    --output_dir "checkpoints/isotropy_contrastive_01" \
    --freeze_base \
    > logs/isotropy_contrastive_01_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  - Evaluate on BEIR: ./scripts/eval_beir_isotropy_contrastive_01.sh"
echo "  - Compare with baseline and pure isotropy"

