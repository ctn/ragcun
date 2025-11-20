#!/bin/bash
# Train ResPred on smoke dataset (10K examples) for quick validation
# This will help us verify:
# 1. Residual learning is working
# 2. Alpha scale evolves properly
# 3. Delta magnitudes are reasonable
# 4. Loss components are balanced

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/respred_smoke_${TIMESTAMP}"
LOG_FILE="logs/residual_gaussian/respred_smoke_${TIMESTAMP}.log"

# Create directories
mkdir -p "$(dirname ${LOG_FILE})"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================" | tee -a ${LOG_FILE}
echo "ResPred Training: Smoke Dataset (10K)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Configuration:" | tee -a ${LOG_FILE}
echo "  Dataset: msmarco_smoke (10K train, 1K val)" | tee -a ${LOG_FILE}
echo "  Architecture: ResPred with residual connection" | tee -a ${LOG_FILE}
echo "  Base model: MPNet (frozen)" | tee -a ${LOG_FILE}
echo "  Output: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "  Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Loss weights:" | tee -a ${LOG_FILE}
echo "  λ_isotropy: 1.5" | tee -a ${LOG_FILE}
echo "  λ_predictive: 1.2" | tee -a ${LOG_FILE}
echo "  λ_residual: 0.01 (NEW!)" | tee -a ${LOG_FILE}
echo "  λ_contrastive: 0.0" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Expected training time: ~15 minutes (3 epochs)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Started at: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Run training
python scripts/train/residual_gaussian.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --output_dim 768 \
    --freeze_base \
    --residual_scale_init 0.1 \
    --batch_size 64 \
    --epochs 3 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --lambda_isotropy 1.5 \
    --lambda_predictive 1.2 \
    --lambda_residual 0.01 \
    --lambda_contrastive 0.0 \
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
echo "  1. Check training stats: cat ${LOG_FILE} | grep -A 5 'Epoch'" | tee -a ${LOG_FILE}
echo "  2. Monitor alpha evolution: cat ${LOG_FILE} | grep 'Alpha:'" | tee -a ${LOG_FILE}
echo "  3. Check delta magnitudes: cat ${LOG_FILE} | grep 'Delta'" | tee -a ${LOG_FILE}
echo "  4. Evaluate on BEIR: python scripts/eval/beir.py --model_path ${OUTPUT_DIR}/best_model.pt" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

