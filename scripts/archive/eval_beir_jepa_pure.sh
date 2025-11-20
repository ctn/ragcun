#!/bin/bash
# Evaluate Pure JEPA model (no contrastive loss) on BEIR

set -e

echo "============================================="
echo "Evaluating: Pure JEPA (No Contrastive)"
echo "============================================="
echo ""
echo "Model: checkpoints/jepa_pure/best_model.pt"
echo "Base: sentence-transformers/all-mpnet-base-v2"
echo "Output Dim: 768"
echo ""

python scripts/eval/beir.py \
    --model_path checkpoints/jepa_pure/best_model.pt \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file results/beir_standard/jepa_pure.json \
    > logs/jepa_pure_eval.log 2>&1

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: results/beir_standard/jepa_pure.json"
echo ""
echo "Compare with:"
echo "  - jepa_10k (λ_contrastive=0.1): results/beir_standard/jepa_10k.json"
echo "  - Vanilla baseline: results/beir_standard/mpnet_frozen.json"

