#!/bin/bash
# Evaluate Pure JEPA Epoch 2 on BEIR

set -e

echo "============================================="
echo "Evaluating: Pure JEPA Epoch 2 (No Contrastive)"
echo "============================================="
echo ""
echo "Model: checkpoints/jepa_pure/checkpoint_epoch_2.pt"
echo "Base: sentence-transformers/all-mpnet-base-v2"
echo "Output Dim: 768"
echo ""

python scripts/eval/beir.py \
    --model_path checkpoints/jepa_pure/checkpoint_epoch_2.pt \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file results/beir_standard/jepa_pure_epoch2.json \
    > logs/jepa_pure_epoch2_eval.log 2>&1

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: results/beir_standard/jepa_pure_epoch2.json"

