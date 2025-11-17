#!/bin/bash
# Evaluate Frozen+Iso 48K Epoch 2 checkpoint on BEIR

set -e

echo "============================================"
echo "BEIR Evaluation: Frozen+Iso 48K (Epoch 2)"
echo "============================================"
echo ""

MODEL_PATH="checkpoints/frozen_48k/mpnet_frozen_isotropy/checkpoint_epoch_2.pt"
OUTPUT_FILE="results/beir_standard/mpnet_frozen_48k_epoch2.json"
BASE_MODEL="sentence-transformers/all-mpnet-base-v2"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Training is still in progress or checkpoint was not saved."
    exit 1
fi

mkdir -p results/beir_standard

echo "Evaluating Epoch 2 checkpoint on 5 BEIR datasets..."
echo "  Model: MPNet Frozen+Iso (48K examples, Epoch 2)"
echo "  Datasets: SciFact, NFCorpus, ArguAna, FiQA, TREC-COVID"
echo ""

python scripts/evaluate_beir.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --freeze_base \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file "$OUTPUT_FILE" \
    > logs/frozen_48k_epoch2_beir_eval.log 2>&1

echo ""
echo "✅ BEIR evaluation complete for Epoch 2!"
echo "Results saved to $OUTPUT_FILE"
echo ""

