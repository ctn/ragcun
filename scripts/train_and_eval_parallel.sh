#!/bin/bash
# ISO15_PRED12 training on 48K with parallel evaluation after each epoch

set -e

# Configuration
TRAIN_DATA="data/processed/msmarco/train.json"
VAL_DATA="data/processed/msmarco/dev.json"
BASE_MODEL="sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIM=768
BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE=2e-5
LAMBDA_ISOTROPY=1.5
LAMBDA_PREDICTIVE=1.2
LAMBDA_CONTRASTIVE=0.0
LOG_INTERVAL=100

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="checkpoints/iso15_pred12_48k_${TIMESTAMP}"
TRAIN_LOG="logs/train_48k_${TIMESTAMP}.log"
EVAL_LOG_DIR="logs/evals_48k_${TIMESTAMP}"
RESULTS_DIR="results/beir_standard"

# Quick eval datasets
EVAL_DATASETS="scifact nfcorpus arguana"

# Setup
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$(dirname "$TRAIN_LOG")"

echo "=============================================" | tee "$TRAIN_LOG"
echo "ISO15_PRED12 Training: 48K Examples" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "Output: $OUTPUT_DIR" | tee -a "$TRAIN_LOG"
echo "Epochs: $EPOCHS" | tee -a "$TRAIN_LOG"
echo "Dataset: 48,433 examples" | tee -a "$TRAIN_LOG"
echo "Started: $(date)" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

# Train one epoch at a time and evaluate in background
for epoch in $(seq 1 $EPOCHS); do
    echo "=============================================" | tee -a "$TRAIN_LOG"
    echo "EPOCH $epoch/$EPOCHS" | tee -a "$TRAIN_LOG"
    echo "Started: $(date)" | tee -a "$TRAIN_LOG"
    echo "=============================================" | tee -a "$TRAIN_LOG"
    echo "" | tee -a "$TRAIN_LOG"
    
    # Determine resume parameters
    if [ $epoch -eq 1 ]; then
        RESUME_ARGS=""
    else
        PREV_EPOCH=$((epoch - 1))
        RESUME_ARGS="--resume_from ${OUTPUT_DIR}/checkpoint_epoch_${PREV_EPOCH}.pt"
    fi
    
    # Train this epoch
    python scripts/train.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --base_model "$BASE_MODEL" \
        --output_dim "$OUTPUT_DIM" \
        --freeze_base \
        --use_predictor \
        --epochs 1 \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --lambda_isotropy "$LAMBDA_ISOTROPY" \
        --lambda_predictive "$LAMBDA_PREDICTIVE" \
        --lambda_contrastive "$LAMBDA_CONTRASTIVE" \
        --lambda_reg 0.0 \
        --log_interval "$LOG_INTERVAL" \
        $RESUME_ARGS \
        2>&1 | tee -a "$TRAIN_LOG"
    
    echo "" | tee -a "$TRAIN_LOG"
    echo "✅ Epoch $epoch training complete: $(date)" | tee -a "$TRAIN_LOG"
    echo "" | tee -a "$TRAIN_LOG"
    
    # Check if checkpoint exists
    CHECKPOINT="${OUTPUT_DIR}/checkpoint_epoch_${epoch}.pt"
    if [ -f "$CHECKPOINT" ]; then
        echo "=============================================" | tee -a "$TRAIN_LOG"
        echo "Launching evaluation for Epoch $epoch (background)" | tee -a "$TRAIN_LOG"
        echo "=============================================" | tee -a "$TRAIN_LOG"
        
        EVAL_LOG="${EVAL_LOG_DIR}/epoch${epoch}.log"
        EVAL_RESULTS="${RESULTS_DIR}/iso15_48k_epoch${epoch}.json"
        
        # Launch eval in background
        nohup python scripts/evaluate_beir.py \
            --model_path "$CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --output_dim "$OUTPUT_DIM" \
            --freeze_base \
            --use_predictor \
            --lambda_reg 0.0 \
            --datasets $EVAL_DATASETS \
            --output_file "$EVAL_RESULTS" \
            > "$EVAL_LOG" 2>&1 &
        
        EVAL_PID=$!
        echo "Eval for Epoch $epoch started (PID: $EVAL_PID)" | tee -a "$TRAIN_LOG"
        echo "  Log: $EVAL_LOG" | tee -a "$TRAIN_LOG"
        echo "  Results: $EVAL_RESULTS" | tee -a "$TRAIN_LOG"
        echo "" | tee -a "$TRAIN_LOG"
    else
        echo "⚠️  Checkpoint not found: $CHECKPOINT" | tee -a "$TRAIN_LOG"
        echo "   Skipping evaluation for epoch $epoch" | tee -a "$TRAIN_LOG"
        echo "" | tee -a "$TRAIN_LOG"
    fi
    
    # Brief pause before next epoch
    sleep 3
done

echo "=============================================" | tee -a "$TRAIN_LOG"
echo "All training complete!" | tee -a "$TRAIN_LOG"
echo "Evaluations running in background." | tee -a "$TRAIN_LOG"
echo "Finished: $(date)" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Monitor evaluations with:" | tee -a "$TRAIN_LOG"
echo "  tail -f ${EVAL_LOG_DIR}/epoch*.log" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Check results in: $RESULTS_DIR" | tee -a "$TRAIN_LOG"

