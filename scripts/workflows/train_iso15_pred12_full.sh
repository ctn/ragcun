#!/bin/bash
#
# Train ISO15_PRED12 on FULL msmarco dataset (48K examples)
# This is the winning architecture that beat baseline by +2.82%
#

set -e

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/iso15_pred12_full_${TIMESTAMP}"
LOG_FILE="logs/iso15_pred12_full_${TIMESTAMP}.log"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_DIR"

echo "Starting ISO15_PRED12 training on FULL dataset (48K examples)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

python scripts/train/isotropic.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/val.json \
    --output_dir "$OUTPUT_DIR" \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --lambda_isotropy 1.5 \
    --lambda_predictive 1.0 \
    --lambda_contrastive 1.2 \
    --log_interval 50 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Training completed!"
echo "Best model saved to: $OUTPUT_DIR/best_model.pt"
echo "Full log: $LOG_FILE"

