#!/bin/bash
# Evaluate JEPA model with Iso=1.5, Pred=1.2 on BEIR

set -e

echo "============================================="
echo "Evaluating: JEPA (Iso=1.5, Pred=1.2)"
echo "============================================="
echo ""
echo "Model: checkpoints/jepa_iso15_pred12/best_model.pt"
echo "Base: sentence-transformers/all-mpnet-base-v2"
echo "Output Dim: 768"
echo ""

python scripts/eval/beir.py \
    --model_path checkpoints/jepa_iso15_pred12/best_model.pt \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file results/beir_standard/jepa_iso15_pred12.json \
    > logs/jepa_iso15_pred12_eval.log 2>&1

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: results/beir_standard/jepa_iso15_pred12.json"
echo ""
echo "Compare with:"
echo "  - Cont 0.0 E3 (Iso=1.0, Pred=1.0): results/beir_standard/jepa_pure_epoch3.json"
echo "  - Cont 0.1 E3 (Iso=1.0, Pred=1.0): results/beir_standard/jepa_10k.json"

