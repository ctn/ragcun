#!/bin/bash
set -e

# BEIR Standard Subset Evaluation
# 3 models × 3 variants × 5 datasets = ~4.5 hours

echo "======================================"
echo "BEIR STANDARD SUBSET EVALUATION"
echo "======================================"
echo "Models: mpnet, minilm-l6, distilroberta"
echo "Variants: baseline, full-ft, frozen"
echo "Datasets: scifact, nfcorpus, arguana, fiqa, trec-covid"
echo "Expected time: ~4.5 hours"
echo "======================================"
echo ""

# Standard subset datasets
DATASETS="scifact nfcorpus arguana fiqa trec-covid"

# Model configurations
declare -A BASE_MODELS
BASE_MODELS["mpnet"]="sentence-transformers/all-mpnet-base-v2"
BASE_MODELS["minilm-l6"]="sentence-transformers/all-MiniLM-L6-v2"
BASE_MODELS["distilroberta"]="sentence-transformers/all-distilroberta-v1"

# Create results directory
mkdir -p results/beir_standard
mkdir -p logs/beir_standard

START_TIME=$(date +%s)

for MODEL_KEY in mpnet minilm-l6 distilroberta; do
    BASE_MODEL="${BASE_MODELS[$MODEL_KEY]}"
    
    echo ""
    echo "======================================"
    echo "EVALUATING: $MODEL_KEY"
    echo "======================================"
    
    # 1. Baseline
    echo ""
    echo "📊 Baseline (λ_iso=0.0)"
    echo "--------------------------------------"
    python scripts/eval/beir.py \
        --model_path "checkpoints/smoke_multi/${MODEL_KEY}_baseline/best_model.pt" \
        --base_model "$BASE_MODEL" \
        --output_dim 512 \
        --datasets $DATASETS \
        --output_file "results/beir_standard/${MODEL_KEY}_baseline.json" \
        2>&1 | tee "logs/beir_standard/${MODEL_KEY}_baseline.log"
    
    echo "✅ Baseline done"
    
    # 2. Full Fine-Tune
    echo ""
    echo "📊 Full Fine-Tune (λ_iso=1.0)"
    echo "--------------------------------------"
    python scripts/eval/beir.py \
        --model_path "checkpoints/smoke_multi/${MODEL_KEY}_isotropy/best_model.pt" \
        --base_model "$BASE_MODEL" \
        --output_dim 512 \
        --datasets $DATASETS \
        --output_file "results/beir_standard/${MODEL_KEY}_fullft.json" \
        2>&1 | tee "logs/beir_standard/${MODEL_KEY}_fullft.log"
    
    echo "✅ Full FT done"
    
    # 3. Frozen Base
    echo ""
    echo "📊 Frozen Base (λ_iso=1.0, base frozen)"
    echo "--------------------------------------"
    python scripts/eval/beir.py \
        --model_path "checkpoints/smoke_frozen/${MODEL_KEY}_frozen_isotropy/best_model.pt" \
        --base_model "$BASE_MODEL" \
        --output_dim 512 \
        --freeze_base \
        --datasets $DATASETS \
        --output_file "results/beir_standard/${MODEL_KEY}_frozen.json" \
        2>&1 | tee "logs/beir_standard/${MODEL_KEY}_frozen.log"
    
    echo "✅ Frozen done"
    
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    echo ""
    echo "⏱️  Elapsed: ${ELAPSED_MIN} minutes"
done

# Generate summary
echo ""
echo "======================================"
echo "GENERATING SUMMARY"
echo "======================================"

python << 'EOF'
import json
import sys
from pathlib import Path

results_dir = Path('results/beir_standard')
models = ['mpnet', 'minilm-l6', 'distilroberta']
variants = ['baseline', 'fullft', 'frozen']

print("\n" + "="*100)
print("BEIR STANDARD SUBSET RESULTS (5 datasets)")
print("="*100)
print()

all_results = {}
for model in models:
    all_results[model] = {}
    for variant in variants:
        result_file = results_dir / f"{model}_{variant}.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results[model][variant] = json.load(f)

# Print dataset-by-dataset comparison
datasets = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']

for dataset in datasets:
    print(f"\n{dataset.upper()}")
    print("-" * 100)
    print(f"{'Model':<20} {'Baseline':<15} {'Full FT':<15} {'Frozen':<15} {'Best':<15}")
    print("-" * 100)
    
    for model in models:
        try:
            base_ndcg = all_results[model]['baseline'][dataset]['ndcg_at_10'] * 100
            full_ndcg = all_results[model]['fullft'][dataset]['ndcg_at_10'] * 100
            froz_ndcg = all_results[model]['frozen'][dataset]['ndcg_at_10'] * 100
            
            best = max(base_ndcg, full_ndcg, froz_ndcg)
            if froz_ndcg == best:
                best_name = "Frozen"
            elif full_ndcg == best:
                best_name = "Full FT"
            else:
                best_name = "Baseline"
            
            print(f"{model:<20} {base_ndcg:5.1f}%          {full_ndcg:5.1f}%          {froz_ndcg:5.1f}%          {best_name}")
        except:
            print(f"{model:<20} ERROR")

# Print average across datasets
print("\n" + "="*100)
print("AVERAGE ACROSS 5 DATASETS")
print("="*100)
print(f"{'Model':<20} {'Baseline':<15} {'Full FT':<15} {'Frozen':<15} {'Winner':<15}")
print("-" * 100)

for model in models:
    try:
        base_avg = sum(all_results[model]['baseline'][d]['ndcg_at_10'] for d in datasets) / len(datasets) * 100
        full_avg = sum(all_results[model]['fullft'][d]['ndcg_at_10'] for d in datasets) / len(datasets) * 100
        froz_avg = sum(all_results[model]['frozen'][d]['ndcg_at_10'] for d in datasets) / len(datasets) * 100
        
        if froz_avg > full_avg and froz_avg > base_avg:
            winner = "Frozen"
        elif full_avg > base_avg:
            winner = "Full FT"
        else:
            winner = "Baseline"
        
        full_delta = full_avg - base_avg
        froz_delta = froz_avg - base_avg
        
        print(f"{model:<20} {base_avg:5.1f}%          {full_avg:5.1f}% ({full_delta:+.1f}%)  {froz_avg:5.1f}% ({froz_delta:+.1f}%)  {winner}")
    except Exception as e:
        print(f"{model:<20} ERROR: {e}")

print("="*100)

# Overall conclusion
print("\n📊 KEY FINDINGS:")
print("="*100)

try:
    # Check if frozen consistently wins
    frozen_wins = 0
    full_wins = 0
    
    for model in models:
        for dataset in datasets:
            base_ndcg = all_results[model]['baseline'][dataset]['ndcg_at_10']
            full_ndcg = all_results[model]['fullft'][dataset]['ndcg_at_10']
            froz_ndcg = all_results[model]['frozen'][dataset]['ndcg_at_10']
            
            if froz_ndcg > full_ndcg and froz_ndcg > base_ndcg:
                frozen_wins += 1
            elif full_ndcg > base_ndcg:
                full_wins += 1
    
    total_experiments = len(models) * len(datasets)
    
    print(f"Frozen Base wins: {frozen_wins}/{total_experiments} experiments ({frozen_wins/total_experiments*100:.0f}%)")
    print(f"Full FT wins: {full_wins}/{total_experiments} experiments ({full_wins/total_experiments*100:.0f}%)")
    print()
    
    if frozen_wins > total_experiments * 0.6:
        print("✅ FROZEN BASE DOMINATES! Achieves best performance with 3x faster training.")
        print("   → Publication-ready result for efficient RAG training")
    elif frozen_wins > total_experiments * 0.4:
        print("✅ MIXED RESULTS. Frozen base competitive but not consistently superior.")
        print("   → Dataset-dependent; frozen works well for some domains")
    else:
        print("✅ FULL FINE-TUNE PREFERRED. Frozen base improvements don't generalize well.")
        print("   → Original hypothesis: full model adaptation needed for zero-shot transfer")
    
except Exception as e:
    print(f"Could not generate conclusion: {e}")

print("="*100)
EOF

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_HR=$((TOTAL_MIN / 60))
REMAINING_MIN=$((TOTAL_MIN % 60))

echo ""
echo "======================================"
echo "EVALUATION COMPLETE"
echo "======================================"
echo "Total time: ${TOTAL_HR}h ${REMAINING_MIN}m"
echo "Results: results/beir_standard/"
echo "Logs: logs/beir_standard/"
echo "======================================"

