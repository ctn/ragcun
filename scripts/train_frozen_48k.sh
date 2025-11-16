#!/bin/bash
# Train Frozen+Iso with Full 48K MS MARCO Dataset
# Goal: Beat vanilla baseline (0.49 NDCG@10)

set -e

echo "============================================"
echo "Training Frozen+Iso with 48K Examples"
echo "============================================"
echo ""
echo "Goal: Exceed vanilla MPNet baseline (0.49 NDCG@10)"
echo "Strategy: More data + frozen base + isotropy"
echo ""

# Create output directory
mkdir -p logs
mkdir -p checkpoints/frozen_48k

# Train MPNet with Frozen Base + Isotropy
echo "Training MPNet (Frozen Base + Isotropy)..."
echo "  • Training examples: 48,433"
echo "  • Base model: sentence-transformers/all-mpnet-base-v2 (FROZEN)"
echo "  • Isotropy: λ_iso = 1.0"
echo "  • Batch size: 16"
echo "  • Epochs: 3"
echo "  • Expected time: ~15-20 minutes"
echo ""

python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 512 \
    --batch_size 16 \
    --epochs 3 \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 1e-3 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir "checkpoints/frozen_48k/mpnet_frozen_isotropy" \
    --freeze_base \
    > logs/frozen_48k_mpnet_training.log 2>&1

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  1. Evaluate on BEIR: ./scripts/eval_beir_frozen_48k.sh"
echo "  2. Compare to vanilla baseline (0.49)"
echo ""

