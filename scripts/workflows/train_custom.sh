#!/bin/bash
# Custom training script with adjustable hyperparameters

set -e

# Parse arguments
TRAIN_DATA=""
VAL_DATA=""
OUTPUT_DIR="checkpoints/custom"
BATCH_SIZE=8
EPOCHS=3
LR=2e-5
LAMBDA_ISO=1.0
LAMBDA_REG=0.1
MARGIN=1.0
FREEZE_LAYERS=false

# Help message
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --train_data PATH       Path to training data (required)"
    echo "  --val_data PATH         Path to validation data"
    echo "  --output_dir PATH       Output directory (default: checkpoints/custom)"
    echo "  --batch_size N          Batch size (default: 8)"
    echo "  --epochs N              Number of epochs (default: 3)"
    echo "  --lr FLOAT              Learning rate (default: 2e-5)"
    echo "  --lambda_iso FLOAT      Isotropy loss weight (default: 1.0)"
    echo "  --lambda_reg FLOAT      Regularization weight (default: 0.1)"
    echo "  --margin FLOAT          Contrastive margin (default: 1.0)"
    echo "  --freeze_layers         Freeze early transformer layers"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --train_data data/processed/train.json --epochs 5 --batch_size 16 --freeze_layers"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --val_data)
            VAL_DATA="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --lambda_iso)
            LAMBDA_ISO="$2"
            shift 2
            ;;
        --lambda_reg)
            LAMBDA_REG="$2"
            shift 2
            ;;
        --margin)
            MARGIN="$2"
            shift 2
            ;;
        --freeze_layers)
            FREEZE_LAYERS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$TRAIN_DATA" ]; then
    echo "❌ Error: --train_data is required"
    usage
fi

echo "============================================"
echo "Custom Training"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Training data: $TRAIN_DATA"
echo "  Validation data: $VAL_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Lambda isotropy: $LAMBDA_ISO"
echo "  Lambda reg: $LAMBDA_REG"
echo "  Margin: $MARGIN"
echo "  Freeze layers: $FREEZE_LAYERS"
echo ""

# Build command
CMD="python scripts/train/isotropic.py \
    --train_data $TRAIN_DATA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --lambda_isotropy $LAMBDA_ISO \
    --lambda_reg $LAMBDA_REG \
    --margin $MARGIN \
    --output_dir $OUTPUT_DIR"

if [ -n "$VAL_DATA" ]; then
    CMD="$CMD --val_data $VAL_DATA"
fi

if [ "$FREEZE_LAYERS" = true ]; then
    CMD="$CMD --freeze_early_layers"
fi

# Run training
echo "Starting training..."
echo ""
eval $CMD

echo ""
echo "============================================"
echo "✅ Training complete!"
echo "============================================"
echo "Model saved to: $OUTPUT_DIR"
