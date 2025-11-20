#!/bin/bash
# Frozen Base Smoke Test: Train only projection layer for efficiency comparison
# Expected time: ~10-15 minutes for all 5 models

set -e

echo "============================================"
echo "🔬 Frozen Base Smoke Test"
echo "============================================"
echo ""
echo "Training ONLY projection layer (frozen base)"
echo "Comparing against full fine-tuning results"
echo "Expected time: ~10-15 minutes"
echo ""

# Create logs directory
mkdir -p logs/smoke_frozen results/smoke_frozen

# Define models to test (same as multi-model test)
declare -A MODELS=(
    ["mpnet"]="sentence-transformers/all-mpnet-base-v2"
    ["minilm-l6"]="sentence-transformers/all-MiniLM-L6-v2"
    ["minilm-l12"]="sentence-transformers/all-MiniLM-L12-v2"
    ["distilroberta"]="sentence-transformers/all-distilroberta-v1"
    ["paraphrase-minilm"]="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

declare -A MODEL_SIZES=(
    ["mpnet"]="110M"
    ["minilm-l6"]="22M"
    ["minilm-l12"]="33M"
    ["distilroberta"]="82M"
    ["paraphrase-minilm"]="22M"
)

echo "Models to test:"
for key in "${!MODELS[@]}"; do
    echo "  - $key: ${MODELS[$key]} (${MODEL_SIZES[$key]})"
done
echo ""

# Common training args
COMMON_ARGS="--train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --epochs 1 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 100 \
    --mixed_precision \
    --log_interval 50"

START_TIME=$(date +%s)

# Train each model with FROZEN BASE
for key in "${!MODELS[@]}"; do
    model_name="${MODELS[$key]}"
    model_size="${MODEL_SIZES[$key]}"
    
    echo ""
    echo "============================================"
    echo "Training: $key ($model_name)"
    echo "Size: $model_size"
    echo "Mode: FROZEN BASE (projection only)"
    echo "Started: $(date +'%H:%M:%S')"
    echo "============================================"
    
    # Train with frozen base + isotropy
    python scripts/train/isotropic.py \
        $COMMON_ARGS \
        --base_model "$model_name" \
        --freeze_base \
        --projection_learning_rate 1e-3 \
        --lambda_isotropy 1.0 \
        --lambda_reg 0.1 \
        --output_dir "checkpoints/smoke_frozen/${key}_frozen_isotropy" \
        > "logs/smoke_frozen/${key}_frozen_isotropy.log" 2>&1
    
    echo "✅ Completed: $(date +'%H:%M:%S')"
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()"
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo "============================================"
echo "✅ All Frozen Base Models Trained!"
echo "============================================"
echo "Total time: ${DURATION_MIN} minutes"
echo ""

# Generate comparison with full fine-tuning results
python << 'EOF'
import json
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from ragcun.model import IsotropicGaussianEncoder

def measure_isotropy(embeddings):
    cov = np.cov(embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) > 0:
        return eigenvalues.min() / eigenvalues.max()
    return 0.0

def test_retrieval(model, queries, passages):
    correct_at_1 = 0
    correct_at_5 = 0
    
    with torch.no_grad():
        for i, (query, pos_passage) in enumerate(zip(queries[:100], passages[:100])):
            negatives = [passages[j] for j in range(len(passages)) if j != i][:9]
            all_passages = [pos_passage] + negatives
            
            q_emb = model.encode(query, convert_to_numpy=False)
            p_embs = model.encode(all_passages, convert_to_numpy=False)
            
            q_emb = q_emb.unsqueeze(0) if q_emb.dim() == 1 else q_emb
            distances = torch.cdist(q_emb, p_embs).squeeze(0)
            ranks = distances.argsort()
            
            pos_rank = (ranks == 0).nonzero(as_tuple=True)[0].item()
            if pos_rank == 0:
                correct_at_1 += 1
                correct_at_5 += 1
            elif pos_rank < 5:
                correct_at_5 += 1
    
    return correct_at_1 / 100, correct_at_5 / 100

models_config = {
    'mpnet': ('sentence-transformers/all-mpnet-base-v2', '110M'),
    'minilm-l6': ('sentence-transformers/all-MiniLM-L6-v2', '22M'),
    'minilm-l12': ('sentence-transformers/all-MiniLM-L12-v2', '33M'),
    'distilroberta': ('sentence-transformers/all-distilroberta-v1', '82M'),
    'paraphrase-minilm': ('sentence-transformers/paraphrase-MiniLM-L6-v2', '22M'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data
with open('data/processed/msmarco_smoke/dev.json') as f:
    data = json.load(f)[:500]

queries = [d['query'] for d in data]
passages = [d['positive'] for d in data]

print("\n" + "="*80)
print("FROZEN BASE vs FULL FINE-TUNING COMPARISON")
print("="*80)
print()

results = []

for key, (base_model, size) in models_config.items():
    print(f"\nEvaluating {key} ({size})...")
    
    # Load baseline (from multi-model test)
    try:
        baseline = IsotropicGaussianEncoder.from_pretrained(
            f'checkpoints/smoke_multi/{key}_baseline/best_model.pt',
            base_model=base_model,
            output_dim=512
        ).to(device).eval()
    except:
        print(f"  ⚠️  Baseline not found, skipping {key}")
        continue
    
    # Load full fine-tune (from multi-model test)
    full_finetune = IsotropicGaussianEncoder.from_pretrained(
        f'checkpoints/smoke_multi/{key}_isotropy/best_model.pt',
        base_model=base_model,
        output_dim=512
    ).to(device).eval()
    
    # Load frozen base (new)
    frozen = IsotropicGaussianEncoder.from_pretrained(
        f'checkpoints/smoke_frozen/{key}_frozen_isotropy/best_model.pt',
        base_model=base_model,
        output_dim=512
    ).to(device).eval()
    
    # Measure isotropy
    with torch.no_grad():
        base_emb = baseline.encode(queries, convert_to_numpy=True)
        full_emb = full_finetune.encode(queries, convert_to_numpy=True)
        frozen_emb = frozen.encode(queries, convert_to_numpy=True)
    
    base_iso = measure_isotropy(base_emb)
    full_iso = measure_isotropy(full_emb)
    frozen_iso = measure_isotropy(frozen_emb)
    
    # Test retrieval
    base_acc1, base_acc5 = test_retrieval(baseline, queries, passages)
    full_acc1, full_acc5 = test_retrieval(full_finetune, queries, passages)
    frozen_acc1, frozen_acc5 = test_retrieval(frozen, queries, passages)
    
    results.append({
        'model': key,
        'size': size,
        'baseline': {'acc1': base_acc1, 'acc5': base_acc5, 'iso': base_iso},
        'full_finetune': {'acc1': full_acc1, 'acc5': full_acc5, 'iso': full_iso},
        'frozen_base': {'acc1': frozen_acc1, 'acc5': frozen_acc5, 'iso': frozen_iso},
    })
    
    # Save results
    with open(f'results/smoke_frozen/{key}_comparison.json', 'w') as f:
        json.dump(results[-1], f, indent=2)
    
    # Clear GPU
    del baseline, full_finetune, frozen
    torch.cuda.empty_cache()

# Print comparison table
print("\n" + "="*90)
print(f"{'Model':<20} {'Size':<8} {'Baseline':<15} {'Full FT':<15} {'Frozen':<15} {'Winner':<10}")
print("="*90)

for r in results:
    model = r['model']
    size = r['size']
    base_acc5 = r['baseline']['acc5'] * 100
    full_acc5 = r['full_finetune']['acc5'] * 100
    frozen_acc5 = r['frozen_base']['acc5'] * 100
    
    full_delta = full_acc5 - base_acc5
    frozen_delta = frozen_acc5 - base_acc5
    
    winner = "Full FT" if full_delta > frozen_delta else "Frozen"
    
    print(f"{model:<20} {size:<8} {base_acc5:5.1f}%          "
          f"{full_acc5:5.1f}% ({full_delta:+.1f}%)   "
          f"{frozen_acc5:5.1f}% ({frozen_delta:+.1f}%)   {winner}")

print("\n" + "="*90)
print("SUMMARY")
print("="*90)

avg_full_improvement = sum(r['full_finetune']['acc5'] - r['baseline']['acc5'] for r in results) / len(results) * 100
avg_frozen_improvement = sum(r['frozen_base']['acc5'] - r['baseline']['acc5'] for r in results) / len(results) * 100

print(f"Average Acc@5 improvement:")
print(f"  Full fine-tune: {avg_full_improvement:+.1f}%")
print(f"  Frozen base:    {avg_frozen_improvement:+.1f}%")
print(f"  Difference:     {avg_full_improvement - avg_frozen_improvement:.1f}%")
print()

if abs(avg_full_improvement - avg_frozen_improvement) < 1.0:
    print("✅ FROZEN BASE is competitive with full fine-tuning!")
    print("   → Huge efficiency gain with minimal performance loss")
elif avg_frozen_improvement > avg_full_improvement:
    print("✅ FROZEN BASE outperforms full fine-tuning!")
    print("   → Projection alone is sufficient (and better!)")
else:
    print("✅ FULL FINE-TUNING provides meaningful improvement")
    print(f"   → {avg_full_improvement - avg_frozen_improvement:.1f}% better performance")

print("="*90)

# Save summary
with open('results/smoke_frozen/summary.json', 'w') as f:
    json.dump({
        'results': results,
        'avg_full_improvement': avg_full_improvement,
        'avg_frozen_improvement': avg_frozen_improvement,
    }, f, indent=2)

print("\nResults saved to results/smoke_frozen/")
EOF

echo ""
echo "Next steps:"
echo "  1. Review comparison table above"
echo "  2. Decide: frozen base (fast) vs full fine-tune (better)?"
echo "  3. Use best approach for pilot/full training"
echo ""

