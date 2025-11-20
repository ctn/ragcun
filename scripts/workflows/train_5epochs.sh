#!/bin/bash
# ISO15_PRED12: Train 5 epochs straight (NO evaluation)

set -e

# Configuration
TRAIN_DATA="data/processed/msmarco/train.json"
VAL_DATA="data/processed/msmarco/dev.json"
BASE_MODEL="sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIM=768
BATCH_SIZE=16
EPOCHS=5
LEARNING_RATE=2e-5
LAMBDA_ISOTROPY=1.5
LAMBDA_PREDICTIVE=1.2
LAMBDA_CONTRASTIVE=0.0
LAMBDA_REG=0.0
LOG_INTERVAL=100

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/iso15_pred12_5epochs_${TIMESTAMP}"
TRAIN_LOG="logs/train_5epochs_${TIMESTAMP}.log"

# Setup
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$TRAIN_LOG")"

echo "=============================================" | tee "$TRAIN_LOG"
echo "ISO15_PRED12 Training: 5 Epochs on 48K Data" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "Output: $OUTPUT_DIR" | tee -a "$TRAIN_LOG"
echo "Epochs: $EPOCHS" | tee -a "$TRAIN_LOG"
echo "Dataset: 48,433 examples" | tee -a "$TRAIN_LOG"
echo "Started: $(date)" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

# Train all 5 epochs in one shot
python scripts/train/isotropic.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --output_dim "$OUTPUT_DIM" \
    --freeze_base \
    --use_predictor \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --lambda_isotropy "$LAMBDA_ISOTROPY" \
    --lambda_predictive "$LAMBDA_PREDICTIVE" \
    --lambda_contrastive "$LAMBDA_CONTRASTIVE" \
    --lambda_reg "$LAMBDA_REG" \
    --log_interval "$LOG_INTERVAL" \
    2>&1 | tee -a "$TRAIN_LOG"

echo "" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "Training complete!" | tee -a "$TRAIN_LOG"
echo "Finished: $(date)" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Checkpoints saved in: $OUTPUT_DIR" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch_1.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch_2.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch_3.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch_4.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch_5.pt" | tee -a "$TRAIN_LOG"
echo "  - best_model.pt" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Run evaluation when ready with:" | tee -a "$TRAIN_LOG"
echo "  python scripts/eval/beir.py \\" | tee -a "$TRAIN_LOG"
echo "    --model_path $OUTPUT_DIR/checkpoint_epoch_5.pt \\" | tee -a "$TRAIN_LOG"
echo "    --base_model $BASE_MODEL \\" | tee -a "$TRAIN_LOG"
echo "    --output_dim $OUTPUT_DIM \\" | tee -a "$TRAIN_LOG"
echo "    --freeze_base \\" | tee -a "$TRAIN_LOG"
echo "    --use_predictor \\" | tee -a "$TRAIN_LOG"
echo "    --datasets scifact nfcorpus arguana \\" | tee -a "$TRAIN_LOG"
echo "    --output_file results/beir_standard/iso15_5epochs.json" | tee -a "$TRAIN_LOG"

