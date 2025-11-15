#!/bin/bash
# Quick data preparation - small dataset for testing

set -e

echo "============================================"
echo "Quick Data Preparation (50 pairs)"
echo "============================================"
echo ""

INPUT="${1:-data/raw/sample_docs.txt}"
OUTPUT_DIR="${2:-data/processed}"
NUM_PAIRS="${3:-50}"

echo "Configuration:"
echo "  Input: $INPUT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of pairs: $NUM_PAIRS"
echo ""

# Check if input exists
if [ ! -f "$INPUT" ]; then
    echo "❌ Error: Input file not found: $INPUT"
    exit 1
fi

# Prepare data
echo "Preparing data..."
python scripts/prepare_data.py \
    --documents "$INPUT" \
    --generate_pairs \
    --num_pairs "$NUM_PAIRS" \
    --split 0.7 0.15 0.15 \
    --output "$OUTPUT_DIR/train.json" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "✅ Quick data preparation complete!"
echo "============================================"
echo ""
echo "Files created:"
echo "  - $OUTPUT_DIR/train.json"
echo "  - $OUTPUT_DIR/val.json"
echo "  - $OUTPUT_DIR/test_eval.json"
echo ""
echo "Next step:"
echo "  ./scripts/train_quick.sh"
