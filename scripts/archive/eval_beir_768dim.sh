#!/bin/bash
# Evaluate 768-dim model on BEIR

set -e

echo "============================================="
echo "BEIR Evaluation: 768-dim Model"
echo "============================================="
echo ""

MODEL_PATH="checkpoints/quick_test_768dim/best_model.pt"
OUTPUT_FILE="results/beir_standard/768dim_test.json"
BASE_MODEL="sentence-transformers/all-mpnet-base-v2"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Training may still be in progress..."
    exit 1
fi

mkdir -p results/beir_standard

echo "Evaluating 768-dim model on 5 BEIR datasets..."
echo "  Model: MPNet Frozen+Iso (768-dim, λ_contrastive=0.1)"
echo "  Datasets: SciFact, NFCorpus, ArguAna, FiQA, TREC-COVID"
echo ""

python scripts/evaluate_beir.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --output_dim 768 \
    --freeze_base \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file "$OUTPUT_FILE" \
    > logs/768dim_beir_eval.log 2>&1

echo ""
echo "✅ BEIR evaluation complete!"
echo "Results saved to $OUTPUT_FILE"

