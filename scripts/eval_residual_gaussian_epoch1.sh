#!/bin/bash
# Quick evaluation of ResPred epoch 1 on scifact and nfcorpus

set -e

CHECKPOINT="checkpoints/respred_smoke_20251118_104914/checkpoint_epoch1.pt"
OUTPUT_FILE="results/beir_standard/respred_epoch1_quick.json"
LOG_FILE="logs/residual_gaussian/eval_residual_gaussian_epoch1.log"

echo "============================================="
echo "ResPred Epoch 1 - Quick Evaluation"
echo "============================================="
echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "Datasets: scifact, nfcorpus"
echo "Output: ${OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Started at: $(date)"
echo "============================================="
echo ""

python scripts/eval_residual_gaussian_quick.py \
    --checkpoint "${CHECKPOINT}" \
    --datasets scifact nfcorpus \
    --output_file "${OUTPUT_FILE}" \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --output_dim 768 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================="
echo "Evaluation Complete!"
echo "Finished at: $(date)"
echo "============================================="
echo ""
echo "Results saved to: ${OUTPUT_FILE}"
echo ""

