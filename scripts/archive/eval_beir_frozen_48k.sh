#!/bin/bash
# Evaluate Frozen+Iso 48K model on BEIR
# Compare to vanilla baseline (0.49 NDCG@10)

set -e

echo "============================================"
echo "BEIR Evaluation: Frozen+Iso (48K training)"
echo "============================================"
echo ""

MODEL_PATH="checkpoints/frozen_48k/mpnet_frozen_isotropy/best_model.pt"
OUTPUT_FILE="results/beir_standard/mpnet_frozen_48k.json"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Run training first: ./scripts/train_frozen_48k.sh"
    exit 1
fi

mkdir -p results/beir_standard

echo "Evaluating on 5 BEIR datasets..."
echo "  Model: MPNet Frozen+Iso (48K examples)"
echo "  Datasets: SciFact, NFCorpus, ArguAna, FiQA, TREC-COVID"
echo ""

python scripts/evaluate_beir.py \
    --model_path "$MODEL_PATH" \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --output_dim 512 \
    --freeze_base \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file "$OUTPUT_FILE" \
    --batch_size 64

echo ""
echo "============================================"
echo "RESULTS COMPARISON"
echo "============================================"
echo ""

python3 << 'EOF'
import json

# Load results
with open('results/beir_standard/mpnet_frozen_48k.json') as f:
    frozen_48k = json.load(f)

# Published vanilla baseline
vanilla_scores = {
    'scifact': 0.655,
    'nfcorpus': 0.335,
    'arguana': 0.453,
    'fiqa': 0.322,
    'trec-covid': 0.688
}

print("NDCG@10 Comparison:")
print()
print(f"{'Dataset':<15} {'Vanilla':>12} {'Frozen+Iso 10K':>18} {'Frozen+Iso 48K':>18} {'Δ vs Vanilla':>15}")
print("-"*85)

datasets = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']
vanilla_avg = sum(vanilla_scores.values()) / len(vanilla_scores)

# Load 10K results for comparison
try:
    with open('results/beir_standard/mpnet_frozen.json') as f:
        frozen_10k = json.load(f)
    has_10k = True
except:
    has_10k = False

total_48k = 0
for ds in datasets:
    vanilla = vanilla_scores[ds]
    new_score = frozen_48k[ds]['NDCG@10']
    total_48k += new_score
    delta = ((new_score - vanilla) / vanilla * 100)
    
    if has_10k:
        old_score = frozen_10k[ds]['NDCG@10']
        print(f"{ds:<15} {vanilla:>12.4f} {old_score:>18.4f} {new_score:>18.4f} {delta:>+14.1f}%")
    else:
        print(f"{ds:<15} {vanilla:>12.4f} {'N/A':>18} {new_score:>18.4f} {delta:>+14.1f}%")

print("-"*85)
avg_48k = total_48k / len(datasets)
avg_delta = ((avg_48k - vanilla_avg) / vanilla_avg * 100)

if has_10k:
    frozen_10k_avg = sum([frozen_10k[ds]['NDCG@10'] for ds in datasets]) / len(datasets)
    print(f"{'AVERAGE':<15} {vanilla_avg:>12.4f} {frozen_10k_avg:>18.4f} {avg_48k:>18.4f} {avg_delta:>+14.1f}%")
else:
    print(f"{'AVERAGE':<15} {vanilla_avg:>12.4f} {'N/A':>18} {avg_48k:>18.4f} {avg_delta:>+14.1f}%")

print()
if avg_48k > vanilla_avg:
    print(f"🎉 SUCCESS! Frozen+Iso (48K) BEATS vanilla baseline by {avg_delta:.1f}%")
elif avg_48k > 0.45:
    print(f"✅ GOOD! Frozen+Iso (48K) matches vanilla baseline ({avg_delta:+.1f}%)")
else:
    print(f"❌ Still below vanilla baseline ({avg_delta:+.1f}%)")
    print("   Consider: More data (100K+), different hyperparameters, or longer training")
EOF

echo ""

