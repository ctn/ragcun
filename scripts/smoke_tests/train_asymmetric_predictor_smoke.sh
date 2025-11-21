#!/bin/bash
# Train Asymmetric + Predictor Model on smoke dataset (10K examples)
# 
# Architecture: (1, 0, 1)
# - Shared encoder (frozen)
# - Separate query/doc projections
# - Predictor for query→doc transformation
# 
# Expected time: ~15-20 minutes

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/asymmetric_pred_smoke_${TIMESTAMP}"
LOG_FILE="logs/asymmetric_predictor/asymmetric_pred_smoke_${TIMESTAMP}.log"

# Create directories
mkdir -p "$(dirname ${LOG_FILE})"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================" | tee -a ${LOG_FILE}
echo "Asymmetric + Predictor Training: Smoke Dataset" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Configuration:" | tee -a ${LOG_FILE}
echo "  Dataset: msmarco_smoke (10K train, 1K val)" | tee -a ${LOG_FILE}
echo "  Architecture: (1, 0, 1)" | tee -a ${LOG_FILE}
echo "    - Encoder: Shared (frozen MPNet)" | tee -a ${LOG_FILE}
echo "    - Projections: Separate (query + doc)" | tee -a ${LOG_FILE}
echo "    - Predictor: Yes (query→doc)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "  Query projection: 768→1536→768 (trainable)" | tee -a ${LOG_FILE}
echo "  Doc projection: 768→1536→768 (trainable)" | tee -a ${LOG_FILE}
echo "  Predictor: 768→1536→768 (trainable)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "  Output: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}
echo "  Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Loss weights:" | tee -a ${LOG_FILE}
echo "  λ_contrastive: 1.0" | tee -a ${LOG_FILE}
echo "  λ_isotropy: 1.0" | tee -a ${LOG_FILE}
echo "  λ_predictive: 1.0" | tee -a ${LOG_FILE}
echo "  Temperature: 0.05" | tee -a ${LOG_FILE}
echo "  Stop-gradient: enabled" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Expected training time: ~15-20 minutes (3 epochs)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Comparison targets:" | tee -a ${LOG_FILE}
echo "  (1,1,1) jepa_10k: 0.4779 (shared proj + predictor)" | tee -a ${LOG_FILE}
echo "  (1,0,0) asymmetric: ~0.47 (separate proj, no predictor)" | tee -a ${LOG_FILE}
echo "  (1,1,0) pure_isotropy: 0.4562 (shared proj, no predictor)" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Hypothesis: (1,0,1) could be the best of both worlds!" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Started at: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Run training
python scripts/train/asymmetric_predictor.py \
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
    --lambda_predictive 1.0 \
    --temperature 0.05 \
    --output_dir "${OUTPUT_DIR}" \
    --log_file "${LOG_FILE}" \
    2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "✅ Training complete!" | tee -a ${LOG_FILE}
echo "Finished at: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Next steps:" | tee -a ${LOG_FILE}
echo "  1. Evaluate on BEIR: python scripts/eval/asymmetric_predictor_quick.py --checkpoint ${OUTPUT_DIR}/best_model.pt" | tee -a ${LOG_FILE}
echo "  2. Compare with (1,1,1) and (1,0,0) models" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Model saved to: ${OUTPUT_DIR}/best_model.pt" | tee -a ${LOG_FILE}


