#!/bin/bash
# Continue training JEPA-like model from epoch 1 checkpoint
# Trains for 2 more epochs (total 3 epochs)

set -e

echo "============================================="
echo "Continuing JEPA Training: Epochs 2-3"
echo "============================================="
echo ""
echo "Loading from: checkpoints/jepa_10k/checkpoint_epoch_1.pt"
echo "Training for 2 more epochs (total: 3 epochs)"
echo "  • Base Model: sentence-transformers/all-mpnet-base-v2 (frozen)"
echo "  • Training Data: data/processed/msmarco_smoke/train.json (10K examples)"
echo "  • Validation Data: data/processed/msmarco_smoke/dev.json (1K examples)"
echo "  • Output Dim: 768"
echo "  • Batch size: 16"
echo "  • Epochs: 3 (continuing from epoch 1)"
echo "  • Loss: λ_contrastive=0.1, λ_isotropy=1.0, λ_predictive=1.0"
echo ""

# Check if checkpoint exists
if [ ! -f "checkpoints/jepa_10k/checkpoint_epoch_1.pt" ]; then
    echo "❌ Checkpoint not found: checkpoints/jepa_10k/checkpoint_epoch_1.pt"
    echo "   Run initial training first: ./scripts/train_jepa_10k.sh"
    exit 1
fi

python scripts/train/isotropic.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --batch_size 16 \
    --epochs 3 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_contrastive 0.1 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --lambda_predictive 1.0 \
    --margin 0.1 \
    --use_predictor \
    --freeze_base \
    --output_dir "checkpoints/jepa_10k" \
    --resume_from "checkpoints/jepa_10k/checkpoint_epoch_1.pt" \
    > logs/jepa_10k_continue_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  - Evaluate on BEIR: ./scripts/eval_beir_jepa_10k.sh (update to use best_model.pt)"
echo "  - Compare with 1-epoch and baseline models"

