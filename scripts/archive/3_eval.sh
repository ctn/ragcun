#!/bin/bash
# Evaluation script - evaluate a trained model

set -e

# Default parameters
MODEL_PATH="${1:-checkpoints/best_model.pt}"
TEST_DATA="${2:-data/processed/test_eval.json}"
OUTPUT_FILE="${3:-results/eval_results.json}"
BATCH_SIZE="${4:-32}"

echo "============================================"
echo "Model Evaluation"
echo "============================================"
echo ""

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test data: $TEST_DATA"
echo "  Output file: $OUTPUT_FILE"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo ""
    echo "Available models:"
    find checkpoints -name "*.pt" 2>/dev/null | head -10 || echo "  No models found in checkpoints/"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo "❌ Error: Test data not found at $TEST_DATA"
    echo "Run: ./scripts/prepare_data_full.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "⚠️  No GPU detected"
echo ""

# Evaluate
echo "Starting evaluation..."
python scripts/evaluate.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --batch_size "$BATCH_SIZE" \
    --top_k 100 \
    --output_file "$OUTPUT_FILE"

echo ""
echo "============================================"
echo "✅ Evaluation complete!"
echo "============================================"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Display results if jq is available
if command -v jq &> /dev/null; then
    echo "Key Metrics:"
    jq -r '.metrics | to_entries | .[] | "  \(.key): \(.value)"' "$OUTPUT_FILE" 2>/dev/null || cat "$OUTPUT_FILE"
else
    echo "Results:"
    cat "$OUTPUT_FILE"
fi
