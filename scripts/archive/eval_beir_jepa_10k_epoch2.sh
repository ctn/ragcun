#!/bin/bash
# Evaluate JEPA-like model (10K training, Epoch 2 checkpoint) on BEIR benchmark

set -e

echo "============================================="
echo "Evaluating JEPA-like Model (10K, Epoch 2) on BEIR"
echo "============================================="
echo ""
echo "Model: checkpoints/jepa_10k/checkpoint_epoch_2.pt"
echo "  • Output Dim: 768"
echo "  • Uses predictor network (training-only, not used in inference)"
echo "  • Base encoder: frozen"
echo ""

python scripts/evaluate_beir.py \
    --model_path "checkpoints/jepa_10k/checkpoint_epoch_2.pt" \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --freeze_base \
    --use_predictor \
    --output_file "results/beir_standard/jepa_10k_epoch2.json" \
    > logs/jepa_10k_epoch2_beir_eval.log 2>&1

echo ""
echo "✅ Evaluation complete!"
echo ""
echo "Results saved to: results/beir_standard/jepa_10k_epoch2.json"
echo "Compare with: python scripts/compare_all_models.py"

