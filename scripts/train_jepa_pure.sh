#!/bin/bash
# Pure JEPA: Predictive loss + Isotropy (NO contrastive loss)
# Ablation study to test if contrastive is necessary

set -e

echo "============================================="
echo "Training: Pure JEPA (No Contrastive Loss)"
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
echo "  • Epochs: 3 (matching jepa_10k)"
echo "  • Loss: λ_contrastive=0.0, λ_isotropy=1.0, λ_predictive=1.0"
echo "  • Stop-gradient: enabled (JEPA standard)"
echo "  • Expected time: ~30-45 minutes"
echo ""

python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 768 \
    --batch_size 16 \
    --epochs 3 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_contrastive 0.0 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --lambda_predictive 1.0 \
    --margin 0.1 \
    --use_predictor \
    --freeze_base \
    --output_dir "checkpoints/jepa_pure" \
    > logs/jepa_pure_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  - Evaluate on BEIR: ./scripts/eval_beir_jepa_pure.sh"
echo "  - Compare with jepa_10k (λ_contrastive=0.1)"

