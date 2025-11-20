#!/bin/bash
# Train Asymmetric Projection Model on smoke dataset (10K examples)
# 
# Key differences from ResPred:
# - Different projections for queries vs documents
# - No predictor, no residual connections
# - Clean contrastive + isotropy losses
# - Expected time: ~15 minutes

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/asymmetric_smoke_${TIMESTAMP}"
LOG_FILE="logs/asymmetric_dual/asymmetric_smoke_${TIMESTAMP}.log"

# Create directories
mkdir -p "$(dirname ${LOG_FILE})"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================" | tee -a ${LOG_FILE}
echo "Asymmetric Projection Training: Smoke Dataset" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Configuration:" | tee -a ${LOG_FILE}
echo "  Dataset: msmarco_smoke (10K train, 1K val)" | tee -a ${LOG_FILE}
echo "  Architecture: Asymmetric projections" | tee -a ${LOG_FILE}
echo "  Query projection: 768→1536→768 (trainable)" | tee -a ${LOG_FILE}
echo "  Doc projection: 768→1536→768 (trainable)" | tee -a ${LOG_FILE}
echo "  Base encoder: MPNet (frozen)" | tee -a ${LOG_FILE}
echo "  Output: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "  Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Loss weights:" | tee -a ${LOG_FILE}
echo "  λ_contrastive: 1.0" | tee -a ${LOG_FILE}
echo "  λ_isotropy: 1.0" | tee -a ${LOG_FILE}
echo "  Temperature: 0.05" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Expected training time: ~15 minutes (3 epochs)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Comparison targets:" | tee -a ${LOG_FILE}
echo "  Baseline (frozen MPNet): 0.4628" | tee -a ${LOG_FILE}
echo "  ISO15_PRED12: 0.4759 (+2.83%)" | tee -a ${LOG_FILE}
echo "  ResPred (failed): 0.4416 (-4.59%)" | tee -a ${LOG_FILE}
echo "  Goal: > 0.48 (+3.5%)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Started at: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Run training
python scripts/train_asymmetric_dual.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --output_dim 768 \
    --batch_size 64 \
    --epochs 3 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --lambda_contrastive 1.0 \
    --lambda_isotropy 1.0 \
    --temperature 0.05 \
    --output_dir "${OUTPUT_DIR}" \
    --log_file "${LOG_FILE}" \
    2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "Training Complete!" | tee -a ${LOG_FILE}
echo "Finished at: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Results saved to: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "Log saved to: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Next steps:" | tee -a ${LOG_FILE}
echo "  1. Check training stats: tail ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "  2. Evaluate on BEIR: python scripts/evaluate_beir.py --model_path ${OUTPUT_DIR}/best_model.pt --model_type asymmetric" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

