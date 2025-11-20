#!/bin/bash
# Evaluate JEPA Contrastive 0.5 Epoch 2 on BEIR

set -e

echo "============================================="
echo "Evaluating: JEPA Contrastive 0.5 Epoch 2"
echo "============================================="
echo ""
echo "Model: checkpoints/jepa_contrastive_05/checkpoint_epoch_2.pt"
echo "Base: sentence-transformers/all-mpnet-base-v2"
echo "Output Dim: 768"
echo ""

python scripts/eval/beir.py \
    --model_path checkpoints/jepa_contrastive_05/checkpoint_epoch_2.pt \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file results/beir_standard/jepa_contrastive_05_epoch2.json \
    > logs/jepa_contrastive_05_epoch2_eval.log 2>&1

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: results/beir_standard/jepa_contrastive_05_epoch2.json"
echo ""
echo "Compare with:"
echo "  - Baseline: results/beir_standard/mpnet_frozen.json"
echo "  - JEPA Pure E2: results/beir_standard/jepa_pure_epoch2.json"
echo "  - JEPA + Cont (λ=0.1): results/beir_standard/jepa_10k.json"

