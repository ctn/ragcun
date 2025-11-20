#!/bin/bash
# Fine-tune self-supervised model on supervised MS MARCO data

set -e

# Configuration
CHECKPOINT="checkpoints/jepa_xy_masked/best_model.pt"
TRAIN_DATA="data/processed/msmarco/train.json"
VAL_DATA="data/processed/msmarco/dev.json"
OUTPUT_DIR="checkpoints/jepa_supervised_finetuned"
BASE_MODEL="sentence-transformers/all-mpnet-base-v2"

echo "============================================="
echo "Fine-Tuning Self-Supervised Model (WITH Predictor)"
echo "============================================="
echo ""
echo "Source checkpoint: $CHECKPOINT"
echo "Training data: $TRAIN_DATA"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ ERROR: Training data not found: $TRAIN_DATA"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Fine-tune with supervised data
# Key differences from self-supervised:
# - Enable contrastive loss (lambda_contrastive > 0) - main supervised signal
# - Keep isotropy loss (maintains learned structure)
# - USE predictor - prevents embedding collapse (key finding!)
# - Lower learning rate (fine-tuning)
python scripts/train/isotropic.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --base_model "$BASE_MODEL" \
    --output_dim 768 \
    --freeze_base \
    --batch_size 32 \
    --epochs 3 \
    --projection_learning_rate 1e-4 \
    --lambda_contrastive 1.0 \
    --lambda_isotropy 1.5 \
    --lambda_reg 0.0 \
    --lambda_predictive 1.0 \
    --use_predictor \
    --margin 0.1 \
    --output_dir "$OUTPUT_DIR" \
    --mixed_precision \
    --resume_from "$CHECKPOINT" \
    --load_weights_only \
    > logs/supervised_finetuning.log 2>&1 &

TRAIN_PID=$!
echo "✅ Fine-tuning started (PID: $TRAIN_PID)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/supervised_finetuning.log"
echo ""
echo "Check status:"
echo "  ps aux | grep train.py"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"

