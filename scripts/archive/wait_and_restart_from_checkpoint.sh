#!/bin/bash
# Wait for epoch 1 to complete, then start evaluation on checkpoint

set -e

LOG_FILE="logs/jepa_xy_masked_training.log"
CHECKPOINT_DIR="checkpoints/jepa_xy_masked"
EPOCH1_CHECKPOINT="$CHECKPOINT_DIR/checkpoint_epoch_1.pt"
RESULTS_DIR="results/beir_standard"
RESULTS_FILE="$RESULTS_DIR/jepa_xy_masked_epoch1.json"

echo "============================================="
echo "Waiting for Epoch 1 Checkpoint"
echo "============================================="
echo ""
echo "Monitoring: $LOG_FILE"
echo "Waiting for: $EPOCH1_CHECKPOINT"
echo "Will evaluate on: scifact, nfcorpus, arguana, fiqa, trec-covid"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to check if epoch 1 is complete
check_epoch1_complete() {
    # Check if checkpoint exists
    if [ -f "$EPOCH1_CHECKPOINT" ]; then
        return 0
    fi
    
    # Check if log shows epoch 1 completion
    if grep -q "Epoch 1.*Train Loss:" "$LOG_FILE" 2>/dev/null; then
        return 0
    fi
    
    return 1
}

# Function to get training PID
get_training_pid() {
    ps aux | grep "python.*train_xy_masked.py" | grep -v grep | awk '{print $2}' | head -1
}

# Monitor until epoch 1 completes
echo "Monitoring training progress..."
while true; do
    PID=$(get_training_pid)
    
    if [ -z "$PID" ]; then
        echo "⚠️  Training process not found. It may have already finished or crashed."
        break
    fi
    
    if check_epoch1_complete; then
        echo ""
        echo "✅ Epoch 1 checkpoint found!"
        echo ""
        break
    fi
    
    # Show current progress
    LATEST_BATCH=$(grep -oE '[0-9]+/7420' "$LOG_FILE" 2>/dev/null | tail -1 | cut -d'/' -f1 || echo "0")
    if [ -n "$LATEST_BATCH" ] && [ "$LATEST_BATCH" != "0" ]; then
        PROGRESS=$(echo "scale=1; $LATEST_BATCH * 100 / 7420" | bc)
        echo -ne "\r⏳ Progress: Batch $LATEST_BATCH/7420 ($PROGRESS%) - Waiting for epoch 1 checkpoint..."
    else
        echo -ne "\r⏳ Waiting for epoch 1 checkpoint..."
    fi
    
    sleep 5
done

# Verify checkpoint exists
if [ ! -f "$EPOCH1_CHECKPOINT" ]; then
    echo ""
    echo "❌ ERROR: Epoch 1 checkpoint not found at $EPOCH1_CHECKPOINT"
    echo "Please check the training logs to see if epoch 1 completed."
    exit 1
fi

echo ""
echo "✅ Epoch 1 checkpoint confirmed: $EPOCH1_CHECKPOINT"
echo ""

# Extract model from checkpoint (evaluate_beir.py needs the model state dict)
# We'll create a temporary best_model.pt from the checkpoint
TEMP_MODEL="$CHECKPOINT_DIR/best_model_epoch1.pt"
echo "Extracting model from checkpoint..."
python3 << PYEOF
import torch

checkpoint = torch.load("$EPOCH1_CHECKPOINT", map_location='cpu')
model_state = checkpoint['model_state_dict']

# Save as best_model format for evaluation
torch.save({
    'model_state_dict': model_state,
    'epoch': checkpoint.get('epoch', 1),
    'loss': checkpoint.get('loss', 0.0)
}, "$TEMP_MODEL")
print(f"✅ Model extracted to $TEMP_MODEL")
PYEOF

echo ""
echo "============================================="
echo "Starting Evaluation on Epoch 1 Checkpoint"
echo "============================================="
echo ""
echo "Model: $TEMP_MODEL"
echo "Base: sentence-transformers/all-mpnet-base-v2"
echo "Output Dim: 768"
echo "Datasets: scifact, nfcorpus, arguana, fiqa, trec-covid"
echo ""

# Start evaluation in background
python scripts/eval/beir.py \
    --model_path "$TEMP_MODEL" \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file "$RESULTS_FILE" \
    > logs/jepa_xy_masked_epoch1_eval.log 2>&1 &

EVAL_PID=$!
echo "✅ Evaluation started (PID: $EVAL_PID)"
echo ""
echo "Evaluation is running in the background."
echo ""
echo "Monitor evaluation progress:"
echo "  tail -f logs/jepa_xy_masked_epoch1_eval.log"
echo ""
echo "Check evaluation status:"
echo "  ps aux | grep evaluate_beir"
echo ""
echo "Results will be saved to: $RESULTS_FILE"
echo ""
echo "Note: Training continues in the background. You can:"
echo "  - Let it continue to epochs 2-3"
echo "  - Stop it manually if needed: ps aux | grep train_xy_masked"
echo ""

