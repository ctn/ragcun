#!/bin/bash
# JEPA with optimized Iso + Pred only (no contrastive, no reg)
# Recommended: Iso=1.5, Pred=1.2

set -e

echo "============================================="
echo "Training: JEPA (Iso=1.5, Pred=1.2, Cont=0, Reg=0)"
echo "============================================="
echo ""
echo "Architecture:"
echo "  • Embed both X (queries) and Y (documents) with F = encoder+projection"
echo "  • Predictor P learns: P(F(X)) ≈ F(Y)"
echo "  • Base Model: sentence-transformers/all-mpnet-base-v2 (frozen)"
echo "  • Training Data: data/processed/msmarco_smoke/train.json (10K examples)"
echo "  • Validation Data: data/processed/msmarco_smoke/dev.json (1K examples)"
echo "  • Output Dim: 768 (matching base encoder, no reduction)"
echo "  • Batch size: 16"
echo "  • Epochs: 3"
echo "  • Loss: λ_contrastive=0.0, λ_isotropy=1.5, λ_reg=0.0, λ_predictive=1.2"
echo "  • Stop-gradient: enabled (JEPA standard)"
echo "  • Expected time: ~10-15 minutes per epoch"
echo ""

python scripts/train/isotropic.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --batch_size 16 \
    --epochs 3 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_contrastive 0.0 \
    --lambda_isotropy 1.5 \
    --lambda_reg 0.0 \
    --lambda_predictive 1.2 \
    --margin 0.1 \
    --use_predictor \
    --freeze_base \
    --output_dir "checkpoints/jepa_iso15_pred12" \
    > logs/jepa_iso15_pred12_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "🔄 Starting evaluation on BEIR..."
echo ""

# Run evaluation immediately after training
./scripts/eval_beir_jepa_iso15_pred12.sh

echo ""
echo "📊 Training and evaluation complete!"
echo ""
echo "Results saved to: results/beir_standard/jepa_iso15_pred12.json"
echo ""
echo "Compare with:"
echo "  - Cont 0.0 E3 (Iso=1.0, Pred=1.0): results/beir_standard/jepa_pure_epoch3.json"
echo "  - Cont 0.1 E3 (Iso=1.0, Pred=1.0): results/beir_standard/jepa_10k.json"

