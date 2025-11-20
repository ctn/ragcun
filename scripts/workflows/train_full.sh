#!/bin/bash
# Full training script - recommended settings for production

set -e

# Load HF_TOKEN from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "============================================"
echo "Full Training (3 epochs, production settings)"
echo "============================================"
echo ""

# Default parameters
TRAIN_DATA="${1:-data/processed/train.json}"
VAL_DATA="${2:-data/processed/val.json}"
OUTPUT_DIR="${3:-checkpoints/full}"
BATCH_SIZE="${4:-8}"
EPOCHS="${5:-3}"

echo "Configuration:"
echo "  Training data: $TRAIN_DATA"
echo "  Validation data: $VAL_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo ""

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo "Run: ./scripts/prepare_data_full.sh first"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "⚠️  No GPU detected"
echo ""

# Train with full settings
echo "Starting full training ($EPOCHS epochs)..."
echo "This will take approximately $((EPOCHS * 30)) - $((EPOCHS * 60)) minutes on T4 GPU"
echo ""

python scripts/train/isotropic.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dim 512 \
    --freeze_early_layers \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --margin 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --save_interval 1 \
    --log_interval 10

echo ""
echo "============================================"
echo "✅ Full training complete!"
echo "============================================"
echo ""
echo "Best model saved to: $OUTPUT_DIR/best_model.pt"
echo "Final model saved to: $OUTPUT_DIR/final_model.pt"
echo ""
echo "View training log: cat training.log"
echo ""
echo "Next steps:"
echo "  1. Evaluate: ./scripts/eval.sh $OUTPUT_DIR/best_model.pt"
echo "  2. Compare checkpoints: ./scripts/eval_all.sh $OUTPUT_DIR"
