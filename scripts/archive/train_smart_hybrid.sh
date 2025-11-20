#!/bin/bash
# Smart Hybrid Training Script
# Trains Gaussian projection on frozen pre-trained base model
#
# This is the recommended approach for publication:
# - Fast (2-3 hours on V100, 45 min on 8x A100)
# - Efficient (trains only 1M params vs 300M)
# - Effective (achieves competitive BEIR scores)

set -e

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "============================================"
echo "Smart Hybrid Training"
echo "============================================"
echo ""
echo "Strategy: Train Gaussian projection on frozen pre-trained base"
echo "Base model: sentence-transformers/all-mpnet-base-v2"
echo "Trainable: ~1.2M params (projection only)"
echo ""

# Default parameters
TRAIN_DATA="${1:-data/processed/msmarco/train.json}"
VAL_DATA="${2:-data/processed/msmarco/dev.json}"
OUTPUT_DIR="${3:-checkpoints/smart_hybrid}"
EPOCHS="${4:-3}"

echo "Configuration:"
echo "  Training data: $TRAIN_DATA"
echo "  Validation data: $VAL_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo ""

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Training data not found: $TRAIN_DATA"
    echo ""
    echo "Download MS MARCO first:"
    echo "  python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    exit 1
fi

# Detect if running on multiple GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "1")
fi

echo "Detected $NUM_GPUS GPU(s)"
echo ""

# Start training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 Starting multi-GPU training ($NUM_GPUS GPUs)..."
    echo ""
    
    torchrun --nproc_per_node=$NUM_GPUS scripts/train/isotropic.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --base_model sentence-transformers/all-mpnet-base-v2 \
        --freeze_base \
        --epochs "$EPOCHS" \
        --batch_size 32 \
        --projection_learning_rate 5e-4 \
        --output_dim 512 \
        --lambda_isotropy 1.0 \
        --lambda_reg 0.1 \
        --warmup_steps 1000 \
        --weight_decay 0.01 \
        --mixed_precision \
        --output_dir "$OUTPUT_DIR" \
        --save_interval 1 \
        --log_interval 100
else
    echo "🚀 Starting single-GPU training..."
    echo ""
    
    python scripts/train/isotropic.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --base_model sentence-transformers/all-mpnet-base-v2 \
        --freeze_base \
        --epochs "$EPOCHS" \
        --batch_size 16 \
        --projection_learning_rate 5e-4 \
        --output_dim 512 \
        --lambda_isotropy 1.0 \
        --lambda_reg 0.1 \
        --warmup_steps 1000 \
        --weight_decay 0.01 \
        --mixed_precision \
        --output_dir "$OUTPUT_DIR" \
        --save_interval 1 \
        --log_interval 100
fi

echo ""
echo "============================================"
echo "✅ Smart Hybrid Training Complete!"
echo "============================================"
echo ""
echo "Model saved to: $OUTPUT_DIR/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate on BEIR:"
echo "     python scripts/eval/beir.py \\"
echo "       --model_path $OUTPUT_DIR/best_model.pt \\"
echo "       --output_file results/beir_results.json"
echo ""
echo "  2. Generate paper results:"
echo "     python scripts/generate_paper_results.py \\"
echo "       --results results/beir_results.json"
echo ""

