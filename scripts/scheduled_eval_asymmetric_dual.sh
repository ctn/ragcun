#!/bin/bash
# Scheduled evaluation script for Asymmetric Projection model
# Waits 30 minutes, then evaluates on scifact + nfcorpus

set -e

WAIT_TIME=1800  # 30 minutes in seconds
CHECKPOINT_DIR="checkpoints/asymmetric_smoke_20251118_111938"
OUTPUT_FILE="results/beir_standard/asymmetric_epoch3_quick.json"
LOG_FILE="logs/asymmetric_dual/eval_asymmetric_dual_scheduled.log"

echo "=============================================" | tee ${LOG_FILE}
echo "Scheduled Evaluation: Asymmetric Projections" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Current time: $(date)" | tee -a ${LOG_FILE}
echo "Waiting ${WAIT_TIME} seconds (30 minutes)..." | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Wait 30 minutes
sleep ${WAIT_TIME}

echo "" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "Wait complete! Starting evaluation..." | tee -a ${LOG_FILE}
echo "Time now: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Check if training completed
if [ ! -f "${CHECKPOINT_DIR}/best_model.pt" ]; then
    echo "⚠️  Warning: best_model.pt not found!" | tee -a ${LOG_FILE}
    echo "Training may not be complete yet." | tee -a ${LOG_FILE}
    echo "Using latest available checkpoint..." | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
    
    # Find most recent checkpoint
    LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/checkpoint_epoch*.pt 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "❌ Error: No checkpoints found in ${CHECKPOINT_DIR}" | tee -a ${LOG_FILE}
        exit 1
    fi
    CHECKPOINT="${LATEST_CKPT}"
    echo "Using: ${CHECKPOINT}" | tee -a ${LOG_FILE}
else
    CHECKPOINT="${CHECKPOINT_DIR}/best_model.pt"
    echo "✅ Found best model checkpoint" | tee -a ${LOG_FILE}
fi

echo "" | tee -a ${LOG_FILE}
echo "Checkpoint: ${CHECKPOINT}" | tee -a ${LOG_FILE}
echo "Datasets: scifact, nfcorpus" | tee -a ${LOG_FILE}
echo "Output: ${OUTPUT_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Run evaluation
python scripts/eval_asymmetric_dual_quick.py \
    --checkpoint "${CHECKPOINT}" \
    --datasets scifact nfcorpus \
    --output_file "${OUTPUT_FILE}" \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --output_dim 768 \
    2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "Evaluation Complete!" | tee -a ${LOG_FILE}
echo "Finished at: $(date)" | tee -a ${LOG_FILE}
echo "=============================================" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
echo "Results saved to: ${OUTPUT_FILE}" | tee -a ${LOG_FILE}
echo "Log saved to: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Show quick comparison
if [ -f "${OUTPUT_FILE}" ]; then
    echo "=============================================" | tee -a ${LOG_FILE}
    echo "QUICK COMPARISON" | tee -a ${LOG_FILE}
    echo "=============================================" | tee -a ${LOG_FILE}
    python3 -c "
import json

# Load results
try:
    with open('results/beir_standard/mpnet_frozen.json') as f:
        baseline = json.load(f)
    
    with open('results/beir_standard/jepa_iso15_pred12.json') as f:
        iso15 = json.load(f)
    
    with open('${OUTPUT_FILE}') as f:
        asymmetric = json.load(f)
    
    datasets = ['scifact', 'nfcorpus']
    
    print()
    print('Model              | Avg NDCG@10 | vs Baseline')
    print('-------------------+-------------+------------')
    
    avg_base = sum(baseline[ds]['NDCG@10'] for ds in datasets) / len(datasets)
    avg_iso = sum(iso15[ds]['NDCG@10'] for ds in datasets) / len(datasets)
    avg_asym = sum(asymmetric[ds]['NDCG@10'] for ds in datasets if 'error' not in asymmetric[ds]) / len(datasets)
    
    print(f'Baseline (MPNet)   | {avg_base:.4f}      | -')
    print(f'ISO15_PRED12       | {avg_iso:.4f}      | +{(avg_iso-avg_base)/avg_base*100:.2f}%')
    print(f'Asymmetric (NEW!)  | {avg_asym:.4f}      | {(avg_asym-avg_base)/avg_base*100:+.2f}%')
    print()
    
    if avg_asym > avg_iso:
        print('🎉 Asymmetric BEATS ISO15_PRED12!')
    elif avg_asym > avg_base:
        print('✅ Asymmetric beats baseline (but not ISO15)')
    else:
        print('❌ Asymmetric underperforms baseline')
    print()

except Exception as e:
    print(f'Error loading results: {e}')
" | tee -a ${LOG_FILE}
fi

echo "See full analysis in: analysis/" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

