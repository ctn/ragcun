#!/bin/bash
# Complete training pipeline - data prep, training, and evaluation

set -e

echo "============================================"
echo "Complete RAG Training Pipeline"
echo "============================================"
echo ""
echo "This script will:"
echo "  1. Prepare training data (1000 pairs)"
echo "  2. Train model (3 epochs, ~1-2 hours)"
echo "  3. Evaluate model"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

OUTPUT_DIR="${1:-checkpoints/pipeline}"
DATA_DIR="data/processed"

echo ""
echo "Step 1/3: Preparing Data"
echo "============================================"
./scripts/prepare_data_full.sh "$DATA_DIR" 1000

echo ""
echo "Step 2/3: Training Model"
echo "============================================"
./scripts/train_full.sh "$DATA_DIR/train.json" "$DATA_DIR/val.json" "$OUTPUT_DIR" 8 3

echo ""
echo "Step 3/3: Evaluating Model"
echo "============================================"
./scripts/eval.sh "$OUTPUT_DIR/best_model.pt" "$DATA_DIR/test_eval.json" "results/pipeline_results.json"

echo ""
echo "============================================"
echo "🎉 Pipeline Complete!"
echo "============================================"
echo ""
echo "Model: $OUTPUT_DIR/best_model.pt"
echo "Results: results/pipeline_results.json"
echo ""
echo "View training log: cat training.log"
echo "View results: cat results/pipeline_results.json"
