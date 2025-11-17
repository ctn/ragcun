#!/bin/bash
# Train JEPA with self-supervised X/Y masked pairs
# 
# This script trains JEPA using only self-supervised data:
#   • X: Original text
#   • Y: Masked version of the same text
#   • Predictor learns: P(embed(X)) ≈ embed(Y)
#
# Dataset: 237,420 X/Y pairs (ratio: 0.0503 - Good for self-supervised)
# Loss: Only isotropy + predictive (no contrastive, no regularization)

set -e

echo "============================================="
echo "Self-Supervised JEPA Training (X/Y Masked)"
echo "============================================="
echo ""
echo "Dataset: data/processed/xy_masked_documents.json"
echo "  • 237,420 X/Y pairs"
echo "  • Ratio: 0.0503 (Good for self-supervised)"
echo ""
echo "Training Strategy:"
echo "  • X: Original text → encode to embedding"
echo "  • Y: Masked text → encode to embedding (target)"
echo "  • Predictor learns: P(embed(X)) ≈ embed(Y)"
echo "  • Base model: Frozen (sentence-transformers/all-mpnet-base-v2)"
echo "  • Only projection + predictor train"
echo ""
echo "Loss Configuration:"
echo "  • lambda_isotropy: 1.5"
echo "  • lambda_predictive: 1.2"
echo "  • lambda_contrastive: 0.0 (self-supervised, no negatives)"
echo "  • lambda_reg: 0.0"
echo ""
echo "Training:"
echo "  • Epochs: 3"
echo "  • Batch size: 32"
echo "  • Learning rate: 5e-4 (projection + predictor)"
echo ""

# Create logs directory
mkdir -p logs

echo "🚀 Starting training..."
echo ""

python scripts/train_xy_masked.py \
    --input_xy_pairs data/processed/xy_masked_documents.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --batch_size 32 \
    --epochs 3 \
    --projection_learning_rate 5e-4 \
    --lambda_contrastive 0.0 \
    --lambda_isotropy 1.5 \
    --lambda_reg 0.0 \
    --lambda_predictive 1.2 \
    --margin 0.1 \
    --use_predictor \
    --freeze_base \
    --output_dir "checkpoints/jepa_xy_masked" \
    > logs/jepa_xy_masked_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Model saved to: checkpoints/jepa_xy_masked/best_model.pt"
echo "Logs: logs/jepa_xy_masked_training.log"
echo ""
echo "Next steps:"
echo "  1. Evaluate on BEIR: ./scripts/eval_beir_jepa_xy_masked.sh"
echo "  2. Compare with other models"
echo ""

