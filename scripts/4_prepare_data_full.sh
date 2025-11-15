#!/bin/bash
# Full data preparation - combines tech and science documents

set -e

echo "============================================"
echo "Full Data Preparation (1000 pairs)"
echo "============================================"
echo ""

OUTPUT_DIR="${1:-data/processed}"
NUM_PAIRS="${2:-1000}"

echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of pairs: $NUM_PAIRS"
echo ""

# Combine documents
echo "Combining tech and science documents..."
if [ -f data/raw/tech_docs.txt ] && [ -f data/raw/science_docs.txt ]; then
    cat data/raw/tech_docs.txt data/raw/science_docs.txt > data/raw/combined_docs.txt
    echo "✅ Combined 61 documents"
else
    echo "⚠️  Warning: Some documents not found, using available files"
    cat data/raw/*.txt 2>/dev/null > data/raw/combined_docs.txt || true
fi

echo ""

# Check if combined file was created
if [ ! -f data/raw/combined_docs.txt ] || [ ! -s data/raw/combined_docs.txt ]; then
    echo "❌ Error: No documents found in data/raw/"
    exit 1
fi

# Prepare data
echo "Generating training pairs from combined documents..."
python scripts/prepare_data.py \
    --documents data/raw/combined_docs.txt \
    --generate_pairs \
    --num_pairs "$NUM_PAIRS" \
    --split 0.8 0.1 0.1 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "✅ Full data preparation complete!"
echo "============================================"
echo ""
echo "Files created:"
echo "  - $OUTPUT_DIR/train.json (800 pairs)"
echo "  - $OUTPUT_DIR/val.json (100 pairs)"
echo "  - $OUTPUT_DIR/test_eval.json (100 queries)"
echo ""
echo "Next step:"
echo "  ./scripts/train_full.sh"
