#!/bin/bash
#
# Incremental training + evaluation for ISO15_PRED12 on full dataset
# 
# Strategy:
#   1. Train epoch 1
#   2. Launch eval epoch 1 (background)
#   3. Train epoch 2
#   4. Launch eval epoch 2 (background)
#   5. Train epoch 3
#   6. Launch eval epoch 3 (background)
#

set -e

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/iso15_pred12_full_${TIMESTAMP}"
TRAIN_LOG="logs/iso15_full_train_${TIMESTAMP}.log"
EVAL_LOG_DIR="logs/iso15_full_evals_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_LOG_DIR"
mkdir -p "$(dirname "$TRAIN_LOG")"

echo "=============================================" | tee "$TRAIN_LOG"
echo "ISO15_PRED12 Incremental Training + Eval" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "Training on: 48,433 examples (full dataset)" | tee -a "$TRAIN_LOG"
echo "Output dir: $OUTPUT_DIR" | tee -a "$TRAIN_LOG"
echo "Train log: $TRAIN_LOG" | tee -a "$TRAIN_LOG"
echo "Eval logs: $EVAL_LOG_DIR/" | tee -a "$TRAIN_LOG"
echo "Started: $(date -u)" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

# Function to train a single epoch
train_epoch() {
    local epoch=$1
    local start_epoch=$epoch
    local resume_from=""
    
    echo "" | tee -a "$TRAIN_LOG"
    echo "=============================================" | tee -a "$TRAIN_LOG"
    echo "Training Epoch $epoch" | tee -a "$TRAIN_LOG"
    echo "Started: $(date -u)" | tee -a "$TRAIN_LOG"
    echo "=============================================" | tee -a "$TRAIN_LOG"
    
    if [ $epoch -gt 1 ]; then
        # Resume from previous epoch
        prev_epoch=$((epoch - 1))
        resume_from="$OUTPUT_DIR/checkpoint_epoch${prev_epoch}.pt"
        echo "Resuming from: $resume_from" | tee -a "$TRAIN_LOG"
    fi
    
    if [ -n "$resume_from" ]; then
        python scripts/train/isotropic.py \
            --train_data data/processed/msmarco/train.json \
            --val_data data/processed/msmarco/dev.json \
            --output_dir "$OUTPUT_DIR" \
            --base_model sentence-transformers/all-mpnet-base-v2 \
            --output_dim 768 \
            --freeze_base \
            --use_predictor \
            --epochs $epoch \
            --batch_size 32 \
            --learning_rate 2e-4 \
            --lambda_isotropy 1.5 \
            --lambda_predictive 1.0 \
            --lambda_contrastive 1.2 \
            --log_interval 100 \
            --resume_from "$resume_from" \
            2>&1 | tee -a "$TRAIN_LOG"
    else
        python scripts/train/isotropic.py \
            --train_data data/processed/msmarco/train.json \
            --val_data data/processed/msmarco/dev.json \
            --output_dir "$OUTPUT_DIR" \
            --base_model sentence-transformers/all-mpnet-base-v2 \
            --output_dim 768 \
            --freeze_base \
            --use_predictor \
            --epochs $epoch \
            --batch_size 32 \
            --learning_rate 2e-4 \
            --lambda_isotropy 1.5 \
            --lambda_predictive 1.0 \
            --lambda_contrastive 1.2 \
            --log_interval 100 \
            2>&1 | tee -a "$TRAIN_LOG"
    fi
    
    echo "" | tee -a "$TRAIN_LOG"
    echo "✅ Epoch $epoch complete: $(date -u)" | tee -a "$TRAIN_LOG"
}

# Function to evaluate a specific epoch (in background)
eval_epoch() {
    local epoch=$1
    local checkpoint="$OUTPUT_DIR/checkpoint_epoch${epoch}.pt"
    local eval_log="$EVAL_LOG_DIR/epoch${epoch}_eval.log"
    local results_file="results/beir_standard/iso15_full_epoch${epoch}_quick.json"
    
    mkdir -p "$(dirname "$results_file")"
    
    echo "" | tee -a "$TRAIN_LOG"
    echo "Launching evaluation for epoch $epoch (background)" | tee -a "$TRAIN_LOG"
    echo "  Checkpoint: $checkpoint" | tee -a "$TRAIN_LOG"
    echo "  Results: $results_file" | tee -a "$TRAIN_LOG"
    echo "  Log: $eval_log" | tee -a "$TRAIN_LOG"
    
    # Launch eval in background
    nohup python scripts/eval/beir.py \
        --model_path "$checkpoint" \
        --base_model sentence-transformers/all-mpnet-base-v2 \
        --output_dim 768 \
        --freeze_base \
        --use_predictor \
        --datasets scifact nfcorpus arguana \
        --output_file "$results_file" \
        > "$eval_log" 2>&1 &
    
    local eval_pid=$!
    echo "  Eval PID: $eval_pid" | tee -a "$TRAIN_LOG"
}

# Main training loop
for epoch in 1 2 3; do
    train_epoch $epoch
    
    # Check if checkpoint was saved
    checkpoint="$OUTPUT_DIR/checkpoint_epoch${epoch}.pt"
    if [ -f "$checkpoint" ]; then
        eval_epoch $epoch
    else
        echo "⚠️  Warning: Checkpoint not found: $checkpoint" | tee -a "$TRAIN_LOG"
        echo "   Skipping evaluation for epoch $epoch" | tee -a "$TRAIN_LOG"
    fi
    
    # Small delay between epochs
    sleep 5
done

echo "" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "Training Complete!" | tee -a "$TRAIN_LOG"
echo "Finished: $(date -u)" | tee -a "$TRAIN_LOG"
echo "=============================================" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Models saved to: $OUTPUT_DIR/" | tee -a "$TRAIN_LOG"
echo "  - best_model.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch1.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch2.pt" | tee -a "$TRAIN_LOG"
echo "  - checkpoint_epoch3.pt" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Evaluation results:" | tee -a "$TRAIN_LOG"
echo "  - results/beir_standard/iso15_full_epoch1_quick.json" | tee -a "$TRAIN_LOG"
echo "  - results/beir_standard/iso15_full_epoch2_quick.json" | tee -a "$TRAIN_LOG"
echo "  - results/beir_standard/iso15_full_epoch3_quick.json" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"
echo "Check eval logs in: $EVAL_LOG_DIR/" | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

# Wait a moment for background evals to start
sleep 10

# Show running eval processes
echo "Background evaluation processes:" | tee -a "$TRAIN_LOG"
ps aux | grep "evaluate_beir.py" | grep -v grep | tee -a "$TRAIN_LOG"
echo "" | tee -a "$TRAIN_LOG"

