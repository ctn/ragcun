#!/bin/bash
# Evaluate all checkpoints in a directory

set -e

CHECKPOINT_DIR="${1:-checkpoints}"
TEST_DATA="${2:-data/processed/test_eval.json}"
OUTPUT_DIR="${3:-results/all_checkpoints}"

echo "============================================"
echo "Evaluate All Checkpoints"
echo "============================================"
echo ""

echo "Configuration:"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Test data: $TEST_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "❌ Error: Test data not found at $TEST_DATA"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all .pt files
MODELS=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f)
NUM_MODELS=$(echo "$MODELS" | wc -l)

if [ -z "$MODELS" ]; then
    echo "❌ No model files found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found $NUM_MODELS model(s) to evaluate"
echo ""

# Evaluate each model
COUNT=0
for MODEL_PATH in $MODELS; do
    COUNT=$((COUNT + 1))
    MODEL_NAME=$(basename "$MODEL_PATH" .pt)
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_results.json"

    echo "[$COUNT/$NUM_MODELS] Evaluating: $MODEL_NAME"
    echo "----------------------------------------"

    python scripts/evaluate.py \
        --model_path "$MODEL_PATH" \
        --test_data "$TEST_DATA" \
        --batch_size 32 \
        --output_file "$OUTPUT_FILE" 2>&1 | grep -E "(Recall|MRR|NDCG|MAP)" || true

    echo ""
done

echo "============================================"
echo "✅ All evaluations complete!"
echo "============================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Create summary if jq is available
if command -v jq &> /dev/null; then
    echo "Summary:"
    echo "----------------------------------------"
    for RESULT_FILE in "$OUTPUT_DIR"/*.json; do
        MODEL_NAME=$(basename "$RESULT_FILE" _results.json)
        MRR=$(jq -r '.metrics.MRR' "$RESULT_FILE" 2>/dev/null || echo "N/A")
        RECALL10=$(jq -r '.metrics."Recall@10"' "$RESULT_FILE" 2>/dev/null || echo "N/A")
        printf "%-30s MRR: %.4f  Recall@10: %.4f\n" "$MODEL_NAME" "$MRR" "$RECALL10" 2>/dev/null || echo "$MODEL_NAME: Error reading metrics"
    done
fi
