#!/bin/bash
# Hyperparameter search - test different configurations

set -e

echo "============================================"
echo "Hyperparameter Search"
echo "============================================"
echo ""

TRAIN_DATA="${1:-data/processed/train.json}"
VAL_DATA="${2:-data/processed/val.json}"
TEST_DATA="${3:-data/processed/test_eval.json}"
BASE_DIR="checkpoints/hypersearch"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found: $TRAIN_DATA"
    echo "Run: ./scripts/prepare_data_full.sh first"
    exit 1
fi

echo "Training data: $TRAIN_DATA"
echo "Base directory: $BASE_DIR"
echo ""
echo "This will train multiple models with different hyperparameters."
echo "Estimated time: 3-6 hours on T4 GPU"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create results directory
mkdir -p results/hypersearch

# Experiment 1: Different learning rates
echo ""
echo "Experiment 1: Learning Rates"
echo "============================================"
for LR in 1e-5 2e-5 5e-5; do
    EXP_NAME="lr_${LR}"
    echo "Training with LR=$LR..."

    python scripts/train.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --epochs 2 \
        --batch_size 8 \
        --learning_rate "$LR" \
        --output_dir "$BASE_DIR/$EXP_NAME" \
        --log_interval 20

    echo "Evaluating..."
    python scripts/evaluate.py \
        --model_path "$BASE_DIR/$EXP_NAME/best_model.pt" \
        --test_data "$TEST_DATA" \
        --output_file "results/hypersearch/${EXP_NAME}.json"

    echo "✅ $EXP_NAME complete"
    echo ""
done

# Experiment 2: Different batch sizes
echo ""
echo "Experiment 2: Batch Sizes"
echo "============================================"
for BS in 4 8 16; do
    EXP_NAME="bs_${BS}"
    echo "Training with batch_size=$BS..."

    python scripts/train.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --epochs 2 \
        --batch_size "$BS" \
        --learning_rate 2e-5 \
        --output_dir "$BASE_DIR/$EXP_NAME" \
        --log_interval 20

    echo "Evaluating..."
    python scripts/evaluate.py \
        --model_path "$BASE_DIR/$EXP_NAME/best_model.pt" \
        --test_data "$TEST_DATA" \
        --output_file "results/hypersearch/${EXP_NAME}.json"

    echo "✅ $EXP_NAME complete"
    echo ""
done

# Experiment 3: Different isotropy weights
echo ""
echo "Experiment 3: Isotropy Loss Weights"
echo "============================================"
for LAMBDA in 0.5 1.0 1.5; do
    EXP_NAME="iso_${LAMBDA}"
    echo "Training with lambda_isotropy=$LAMBDA..."

    python scripts/train.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --epochs 2 \
        --batch_size 8 \
        --lambda_isotropy "$LAMBDA" \
        --output_dir "$BASE_DIR/$EXP_NAME" \
        --log_interval 20

    echo "Evaluating..."
    python scripts/evaluate.py \
        --model_path "$BASE_DIR/$EXP_NAME/best_model.pt" \
        --test_data "$TEST_DATA" \
        --output_file "results/hypersearch/${EXP_NAME}.json"

    echo "✅ $EXP_NAME complete"
    echo ""
done

echo ""
echo "============================================"
echo "🎉 Hyperparameter Search Complete!"
echo "============================================"
echo ""
echo "Results saved to: results/hypersearch/"
echo ""

# Generate summary
if command -v jq &> /dev/null; then
    echo "Summary of Results:"
    echo "============================================"
    printf "%-20s %-10s %-10s %-10s\n" "Experiment" "MRR" "Recall@10" "NDCG@10"
    echo "------------------------------------------------------------"

    for RESULT in results/hypersearch/*.json; do
        EXP=$(basename "$RESULT" .json)
        MRR=$(jq -r '.metrics.MRR' "$RESULT" 2>/dev/null || echo "N/A")
        R10=$(jq -r '.metrics."Recall@10"' "$RESULT" 2>/dev/null || echo "N/A")
        NDCG=$(jq -r '.metrics."NDCG@10"' "$RESULT" 2>/dev/null || echo "N/A")
        printf "%-20s %-10.4f %-10.4f %-10.4f\n" "$EXP" "$MRR" "$R10" "$NDCG" 2>/dev/null || echo "$EXP: Error"
    done
fi
