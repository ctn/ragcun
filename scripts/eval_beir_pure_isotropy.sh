#!/bin/bash
# Evaluate Pure Isotropy model (no contrastive loss) on BEIR

set -e

echo "============================================="
echo "BEIR Evaluation: Pure Isotropy Model"
echo "============================================="
echo ""
echo "Model: Pure Isotropy (λ_contrastive=0.0, λ_isotropy=1.0, λ_reg=0.0)"
echo "  • Base Model: sentence-transformers/all-mpnet-base-v2 (frozen)"
echo "  • Training: 10K MS MARCO examples, 1 epoch"
echo "  • Strategy: Only isotropy regularization, no contrastive loss"
echo ""

MODEL_PATH="checkpoints/pure_isotropy_only/best_model.pt"
OUTPUT_FILE="results/beir_standard/pure_isotropy_only.json"
BASE_MODEL="sentence-transformers/all-mpnet-base-v2"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

mkdir -p results/beir_standard

echo "Evaluating on 5 BEIR datasets..."
echo "  Datasets: SciFact, NFCorpus, ArguAna, FiQA, TREC-COVID"
echo ""

python scripts/evaluate_beir.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --freeze_base \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file "$OUTPUT_FILE" \
    > logs/pure_isotropy_only_beir_eval.log 2>&1

echo ""
echo "✅ BEIR evaluation complete for Pure Isotropy model!"
echo "Results saved to $OUTPUT_FILE"
echo ""
echo "Next: Compare against vanilla baseline using scripts/compare_baseline_vs_sigreg.py"

