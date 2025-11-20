#!/bin/bash
# Recommended training path: Full fine-tuning with ablations
# This implements the single recommended path from docs/RECOMMENDED_TRAINING_PATH.md

set -e

echo "============================================"
echo "Publication Training: Recommended Path"
echo "============================================"
echo ""
echo "This will train 3 models:"
echo "  1. Baseline (no isotropy) - 5-6 days"
echo "  2. With isotropy (YOUR METHOD) - 5-6 days"
echo "  3. Frozen base (efficiency) - 2-3 days"
echo ""
echo "Total time: ~15 days sequential, ~6 days parallel"
echo ""

# Confirmation
read -p "Continue with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check data
if [ ! -f "data/processed/msmarco/train.json" ]; then
    echo ""
    echo "❌ MS MARCO not found. Downloading..."
    python scripts/download_msmarco.py --output_dir data/processed/msmarco
else
    echo "✅ MS MARCO data found"
fi

# Create output directories
mkdir -p checkpoints results logs

# Common arguments
COMMON_ARGS="--train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --mixed_precision \
    --save_interval 1 \
    --log_interval 100"

# Experiment 1: Baseline (no isotropy)
echo ""
echo "============================================"
echo "Experiment 1: Baseline (no isotropy)"
echo "Started: $(date)"
echo "============================================"
python scripts/train/isotropic.py \
    $COMMON_ARGS \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --output_dir checkpoints/baseline_no_isotropy \
    2>&1 | tee logs/baseline_no_isotropy.log

echo "✅ Experiment 1 complete: $(date)"

# Experiment 2: With isotropy (YOUR METHOD)
echo ""
echo "============================================"
echo "Experiment 2: With Isotropy (YOUR METHOD)"
echo "Started: $(date)"
echo "============================================"
python scripts/train/isotropic.py \
    $COMMON_ARGS \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/with_isotropy \
    2>&1 | tee logs/with_isotropy.log

echo "✅ Experiment 2 complete: $(date)"

# Experiment 3: Frozen base (efficiency)
echo ""
echo "============================================"
echo "Experiment 3: Frozen Base (Efficiency)"
echo "Started: $(date)"
echo "============================================"
python scripts/train/isotropic.py \
    $COMMON_ARGS \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/frozen_efficient \
    2>&1 | tee logs/frozen_efficient.log

echo "✅ Experiment 3 complete: $(date)"

# Summary
echo ""
echo "============================================"
echo "✅ All Training Complete!"
echo "============================================"
echo ""
echo "Models trained:"
echo "  1. checkpoints/baseline_no_isotropy/best_model.pt"
echo "  2. checkpoints/with_isotropy/best_model.pt"
echo "  3. checkpoints/frozen_efficient/best_model.pt"
echo ""
echo "Training logs saved in logs/"
echo ""
echo "Next steps:"
echo "  1. Evaluate all models on BEIR:"
echo "     ./scripts/evaluate_all_beir.sh"
echo ""
echo "  2. Generate comparison table for paper:"
echo "     python scripts/generate_comparison_table.py"
echo ""

