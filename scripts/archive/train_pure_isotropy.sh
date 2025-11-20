#!/bin/bash
# Train with ONLY isotropy loss (no contrastive learning)
# This tests if the base model's semantic knowledge + isotropy regularization
# is sufficient for retrieval

set -e

echo "============================================="
echo "Training: Pure Isotropy (No Contrastive Loss)"
echo "============================================="
echo ""
echo "Strategy: λ_contrastive=0.0, λ_isotropy=1.0, λ_reg=0.0"
echo "  • Base Model: sentence-transformers/all-mpnet-base-v2 (frozen)"
echo "  • Training Data: data/processed/msmarco_smoke/train.json (10K examples)"
echo "  • Validation Data: data/processed/msmarco_smoke/dev.json (1K examples)"
echo "  • Output Dim: 512"
echo "  • Batch size: 16"
echo "  • Epochs: 1"
echo "  • Expected time: ~7-10 minutes"
echo ""

python scripts/train/isotropic.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 512 \
    --batch_size 16 \
    --epochs 1 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 1e-3 \
    --lambda_contrastive 0.0 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --margin 0.1 \
    --output_dir "checkpoints/pure_isotropy_only" \
    --freeze_base \
    > logs/pure_isotropy_only_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  - Evaluate on BEIR: python scripts/eval/beir.py --model_path checkpoints/pure_isotropy_only/best_model.pt --base_model sentence-transformers/all-mpnet-base-v2 --freeze_base --datasets scifact nfcorpus arguana fiqa trec-covid"
echo "  - Compare with: contrastive-only baseline and contrastive+isotropy"

