#!/bin/bash
# Quick pipeline for testing - runs in ~5 minutes

set -e

echo "============================================"
echo "Quick Test Pipeline (~5 minutes)"
echo "============================================"
echo ""
echo "This script will:"
echo "  1. Prepare small dataset (50 pairs)"
echo "  2. Train for 1 epoch (~3 minutes)"
echo "  3. Evaluate model"
echo ""

OUTPUT_DIR="${1:-checkpoints/quick}"
DATA_DIR="data/processed/quick"

echo "Step 1/3: Preparing Data"
echo "============================================"
./scripts/1_prepare_data_quick.sh "data/raw/sample_docs.txt" "$DATA_DIR" 50

echo ""
echo "Step 2/3: Training Model"
echo "============================================"
./scripts/2_train_quick.sh "$DATA_DIR/train.json" "$DATA_DIR/val.json" "$OUTPUT_DIR" 8

echo ""
echo "Step 3/3: Evaluating Model"
echo "============================================"
./scripts/3_eval.sh "$OUTPUT_DIR/final_model.pt" "$DATA_DIR/test_eval.json" "results/quick_results.json"

echo ""
echo "============================================"
echo "✅ Quick pipeline complete!"
echo "============================================"
echo ""
echo "Model: $OUTPUT_DIR/final_model.pt"
echo "Results: results/quick_results.json"
echo ""
echo "For full training, run: ./scripts/pipeline_full.sh"
