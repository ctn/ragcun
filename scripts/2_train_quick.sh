#!/bin/bash
# Quick training script - trains for 1 epoch for testing

set -e

echo "============================================"
echo "Quick Training (1 epoch for testing)"
echo "============================================"
echo ""

# Default parameters
TRAIN_DATA="${1:-data/processed/train.json}"
VAL_DATA="${2:-data/processed/val.json}"
OUTPUT_DIR="${3:-checkpoints/quick}"
BATCH_SIZE="${4:-8}"

echo "Configuration:"
echo "  Training data: $TRAIN_DATA"
echo "  Validation data: $VAL_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo "Run: ./scripts/prepare_data_quick.sh first"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "⚠️  No GPU detected - training will be slow"
echo ""

# Train
echo "Starting quick training (1 epoch)..."
python scripts/train.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --epochs 1 \
    --batch_size "$BATCH_SIZE" \
    --output_dim 512 \
    --learning_rate 2e-5 \
    --output_dir "$OUTPUT_DIR" \
    --log_interval 5

echo ""
echo "============================================"
echo "✅ Quick training complete!"
echo "============================================"
echo ""
echo "Model saved to: $OUTPUT_DIR/final_model.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate: ./scripts/eval.sh $OUTPUT_DIR/final_model.pt"
echo "  2. Full training: ./scripts/train_full.sh"
