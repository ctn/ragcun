#!/bin/bash
# Evaluate all 5 epochs on scifact and nfcorpus (first two datasets)

set -e

BASE_MODEL="sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIM=768
DATASETS="scifact nfcorpus"
CHECKPOINT_DIR="checkpoints/iso15_pred12_5epochs_20251119_075704"
RESULTS_DIR="results/beir_standard"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"
mkdir -p "logs/eval_5epochs"

echo "========================================"
echo "Launching evaluations for all 5 epochs"
echo "========================================"
echo "Datasets: $DATASETS"
echo "Timestamp: $TIMESTAMP"
echo ""

# Evaluate each epoch in parallel
for epoch in 1 2 3 4 5; do
    CHECKPOINT="$CHECKPOINT_DIR/checkpoint_epoch_${epoch}.pt"
    OUTPUT_FILE="$RESULTS_DIR/iso15_epoch${epoch}_${TIMESTAMP}.json"
    LOG_FILE="logs/eval_5epochs/epoch${epoch}_${TIMESTAMP}.log"
    
    echo "Starting epoch $epoch evaluation..."
    nohup python scripts/eval/beir.py \
        --model_path "$CHECKPOINT" \
        --base_model "$BASE_MODEL" \
        --output_dim "$OUTPUT_DIM" \
        --freeze_base \
        --use_predictor \
        --datasets $DATASETS \
        --output_file "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "  • Epoch $epoch: PID $PID (log: $LOG_FILE)"
done

echo ""
echo "✅ All 5 evaluations launched in background!"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/eval_5epochs/epoch*_${TIMESTAMP}.log"
echo ""
echo "Results will be saved to:"
echo "  $RESULTS_DIR/iso15_epoch[1-5]_${TIMESTAMP}.json"

