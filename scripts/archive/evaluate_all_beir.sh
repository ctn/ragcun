#!/bin/bash
# Evaluate all trained models on BEIR benchmark
# This evaluates the models from the recommended training path

set -e

echo "============================================"
echo "BEIR Evaluation: All Models"
echo "============================================"
echo ""
echo "This will evaluate 4 models on BEIR:"
echo "  0. Original MPNet (baseline)"
echo "  1. Full FT without isotropy"
echo "  2. Full FT with isotropy (YOUR METHOD)"
echo "  3. Frozen base with isotropy (efficiency)"
echo ""
echo "Each evaluation takes ~3-4 hours on GPU"
echo "Total time: ~12-16 hours"
echo ""

# Confirmation
read -p "Continue with evaluation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

# Create results directory
mkdir -p results

# Baseline: Original MPNet (no fine-tuning)
echo ""
echo "============================================"
echo "Evaluating: Original MPNet (no fine-tuning)"
echo "Started: $(date)"
echo "============================================"
python scripts/eval/beir.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --datasets all \
    --output_file results/beir_mpnet_original.json \
    2>&1 | tee logs/eval_mpnet_original.log

echo "✅ Original MPNet evaluation complete: $(date)"

# Experiment 1: Baseline (no isotropy)
echo ""
echo "============================================"
echo "Evaluating: Baseline (no isotropy)"
echo "Started: $(date)"
echo "============================================"

if [ ! -f "checkpoints/baseline_no_isotropy/best_model.pt" ]; then
    echo "⚠️  Model not found: checkpoints/baseline_no_isotropy/best_model.pt"
    echo "Skipping..."
else
    python scripts/eval/beir.py \
        --model_path checkpoints/baseline_no_isotropy/best_model.pt \
        --datasets all \
        --output_file results/beir_baseline.json \
        2>&1 | tee logs/eval_baseline.log
    
    echo "✅ Baseline evaluation complete: $(date)"
fi

# Experiment 2: With isotropy (YOUR METHOD)
echo ""
echo "============================================"
echo "Evaluating: With Isotropy (YOUR METHOD)"
echo "Started: $(date)"
echo "============================================"

if [ ! -f "checkpoints/with_isotropy/best_model.pt" ]; then
    echo "⚠️  Model not found: checkpoints/with_isotropy/best_model.pt"
    echo "Skipping..."
else
    python scripts/eval/beir.py \
        --model_path checkpoints/with_isotropy/best_model.pt \
        --datasets all \
        --output_file results/beir_with_isotropy.json \
        2>&1 | tee logs/eval_with_isotropy.log
    
    echo "✅ With isotropy evaluation complete: $(date)"
fi

# Experiment 3: Frozen base (efficiency)
echo ""
echo "============================================"
echo "Evaluating: Frozen Base (Efficiency)"
echo "Started: $(date)"
echo "============================================"

if [ ! -f "checkpoints/frozen_efficient/best_model.pt" ]; then
    echo "⚠️  Model not found: checkpoints/frozen_efficient/best_model.pt"
    echo "Skipping..."
else
    python scripts/eval/beir.py \
        --model_path checkpoints/frozen_efficient/best_model.pt \
        --datasets all \
        --output_file results/beir_frozen.json \
        2>&1 | tee logs/eval_frozen.log
    
    echo "✅ Frozen base evaluation complete: $(date)"
fi

# Summary
echo ""
echo "============================================"
echo "✅ All Evaluations Complete!"
echo "============================================"
echo ""
echo "Results saved:"
echo "  - results/beir_mpnet_original.json"
echo "  - results/beir_baseline.json"
echo "  - results/beir_with_isotropy.json"
echo "  - results/beir_frozen.json"
echo ""
echo "Evaluation logs saved in logs/"
echo ""
echo "Next step: Generate comparison table for paper"
echo "  python scripts/generate_comparison_table.py \\"
echo "    --baseline results/beir_mpnet_original.json \\"
echo "    --no_isotropy results/beir_baseline.json \\"
echo "    --with_isotropy results/beir_with_isotropy.json \\"
echo "    --frozen results/beir_frozen.json \\"
echo "    --output paper/results_table.tex"
echo ""

# Quick summary of results
echo "Quick Results Summary:"
echo "======================"
for result_file in results/beir_*.json; do
    if [ -f "$result_file" ]; then
        avg=$(python -c "import json; data=json.load(open('$result_file')); print(f\"{data.get('average', {}).get('ndcg@10', 0)*100:.1f}%\" if 'average' in data else 'N/A')" 2>/dev/null || echo "N/A")
        echo "$(basename $result_file .json): NDCG@10 = $avg"
    fi
done
echo ""

