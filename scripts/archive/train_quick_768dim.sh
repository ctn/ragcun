#!/bin/bash
# Quick test: 768-dim vs 512-dim embeddings

set -e

echo "============================================="
echo "Quick Test: 768-dim Embeddings"
echo "============================================="
echo ""
echo "Testing if output_dim=768 (matching base encoder) improves performance"
echo "  • Base Model: sentence-transformers/all-mpnet-base-v2 (frozen)"
echo "  • Output Dim: 768 (vs 512 in current models)"
echo "  • Training Data: data/processed/msmarco_smoke/train.json (10K examples)"
echo "  • Epochs: 1 (quick test)"
echo "  • λ_contrastive=0.1, λ_isotropy=1.0"
echo ""

python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --batch_size 16 \
    --epochs 1 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_contrastive 0.1 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --margin 0.1 \
    --output_dir "checkpoints/quick_test_768dim" \
    --freeze_base \
    > logs/quick_test_768dim.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Next: Evaluate on BEIR to compare with 512-dim model"

