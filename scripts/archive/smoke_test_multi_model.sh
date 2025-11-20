#!/bin/bash
# Multi-Model Smoke Test: Test isotropy regularization across multiple models
# Expected time: ~1-2 hours for all models on T4

set -e

echo "============================================"
echo "🔬 Multi-Model Smoke Test"
echo "============================================"
echo ""
echo "Testing isotropy regularization across multiple architectures"
echo "This will take ~1-2 hours total"
echo ""

# Create logs directory
mkdir -p logs/smoke_multi results/smoke_multi

# Define models to test (110M and smaller)
declare -A MODELS=(
    ["mpnet"]="sentence-transformers/all-mpnet-base-v2"
    ["minilm-l6"]="sentence-transformers/all-MiniLM-L6-v2"
    ["minilm-l12"]="sentence-transformers/all-MiniLM-L12-v2"
    ["distilroberta"]="sentence-transformers/all-distilroberta-v1"
    ["paraphrase-minilm"]="sentence-transformers/paraphrase-MiniLM-L6-v2"
)

# Model sizes (approximate params)
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

# Results summary file
SUMMARY_FILE="results/smoke_multi/summary.txt"
echo "Multi-Model Smoke Test Results" > $SUMMARY_FILE
echo "===============================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Function to train and evaluate a model
train_model() {
    local model_key=$1
    local model_name=$2
    local model_size=$3
    
    echo ""
    echo "============================================"
    echo "Testing: $model_key ($model_name)"
    echo "Size: $model_size"
    echo "Started: $(date +'%H:%M:%S')"
    echo "============================================"
    
    # Baseline (no isotropy)
    echo "Training baseline (λ_iso=0)..."
    python scripts/train.py \
        $COMMON_ARGS \
        --base_model "$model_name" \
        --base_learning_rate 2e-5 \
        --projection_learning_rate 1e-3 \
        --lambda_isotropy 0.0 \
        --lambda_reg 0.0 \
        --output_dir "checkpoints/smoke_multi/${model_key}_baseline" \
        > "logs/smoke_multi/${model_key}_baseline.log" 2>&1
    
    echo "✅ Baseline done"
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
    
    # With isotropy
    echo "Training with isotropy (λ_iso=1.0)..."
    python scripts/train.py \
        $COMMON_ARGS \
        --base_model "$model_name" \
        --base_learning_rate 2e-5 \
        --projection_learning_rate 1e-3 \
        --lambda_isotropy 1.0 \
        --lambda_reg 0.1 \
        --output_dir "checkpoints/smoke_multi/${model_key}_isotropy" \
        > "logs/smoke_multi/${model_key}_isotropy.log" 2>&1
    
    echo "✅ With isotropy done"
    echo "Completed: $(date +'%H:%M:%S')"
}

# Function to analyze results for a model
analyze_model() {
    local model_key=$1
    local model_name=$2
    local model_size=$3
    
    echo ""
    echo "Analyzing $model_key..."
    
    python << EOF
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
try:
    baseline = IsotropicGaussianEncoder.from_pretrained(
        'checkpoints/smoke_multi/${model_key}_baseline/best_model.pt',
        base_model='${model_name}',
        output_dim=512
    ).to(device).eval()
    
    isotropy = IsotropicGaussianEncoder.from_pretrained(
        'checkpoints/smoke_multi/${model_key}_isotropy/best_model.pt',
        base_model='${model_name}',
        output_dim=512
    ).to(device).eval()
except Exception as e:
    print(f"ERROR loading models: {e}")
    sys.exit(1)

# Load test data
with open('data/processed/msmarco_smoke/dev.json') as f:
    data = json.load(f)[:500]

queries = [d['query'] for d in data]
passages = [d['positive'] for d in data]

# Measure isotropy
with torch.no_grad():
    base_emb = baseline.encode(queries, convert_to_numpy=True)
    iso_emb = isotropy.encode(queries, convert_to_numpy=True)

base_iso = measure_isotropy(base_emb)
iso_iso = measure_isotropy(iso_emb)

# Test retrieval
base_acc1, base_acc5 = test_retrieval(baseline, queries, passages)
iso_acc1, iso_acc5 = test_retrieval(isotropy, queries, passages)

# Save results
results = {
    'model': '${model_key}',
    'model_name': '${model_name}',
    'model_size': '${model_size}',
    'isotropy': {
        'baseline': float(base_iso),
        'with_reg': float(iso_iso),
        'improvement': float(iso_iso - base_iso)
    },
    'retrieval': {
        'baseline_acc1': float(base_acc1),
        'baseline_acc5': float(base_acc5),
        'isotropy_acc1': float(iso_acc1),
        'isotropy_acc5': float(iso_acc5),
        'improvement_acc1': float(iso_acc1 - base_acc1),
        'improvement_acc5': float(iso_acc5 - base_acc5)
    }
}

with open('results/smoke_multi/${model_key}_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"\\n{'='*60}")
print(f"Results for ${model_key} (${model_size})")
print(f"{'='*60}")
print(f"Isotropy: {base_iso:.4f} → {iso_iso:.4f} ({iso_iso-base_iso:+.4f})")
print(f"Acc@1:    {base_acc1*100:.1f}% → {iso_acc1*100:.1f}% ({(iso_acc1-base_acc1)*100:+.1f}%)")
print(f"Acc@5:    {base_acc5*100:.1f}% → {iso_acc5*100:.1f}% ({(iso_acc5-base_acc5)*100:+.1f}%)")
print(f"{'='*60}")
EOF
}

# Main execution
echo "Starting multi-model smoke tests..."
echo ""

START_TIME=$(date +%s)

# Test each model
for key in "${!MODELS[@]}"; do
    train_model "$key" "${MODELS[$key]}" "${MODEL_SIZES[$key]}"
    analyze_model "$key" "${MODELS[$key]}" "${MODEL_SIZES[$key]}"
    
    # Append to summary
    echo "" >> $SUMMARY_FILE
    cat "results/smoke_multi/${key}_results.json" >> $SUMMARY_FILE
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo "============================================"
echo "✅ All Models Tested!"
echo "============================================"
echo "Total time: ${DURATION_MIN} minutes"
echo ""

# Generate comparison table
python << 'EOF'
import json
from pathlib import Path

print("\n" + "="*80)
print("MULTI-MODEL COMPARISON")
print("="*80)
print()

results = []
for f in Path('results/smoke_multi').glob('*_results.json'):
    with open(f) as file:
        results.append(json.load(file))

# Sort by model size
results.sort(key=lambda x: x['model_size'])

print(f"{'Model':<20} {'Size':<8} {'Acc@1 Δ':<12} {'Acc@5 Δ':<12} {'Iso Δ':<10}")
print("-"*80)

for r in results:
    model = r['model']
    size = r['model_size']
    acc1_delta = r['retrieval']['improvement_acc1'] * 100
    acc5_delta = r['retrieval']['improvement_acc5'] * 100
    iso_delta = r['isotropy']['improvement']
    
    print(f"{model:<20} {size:<8} {acc1_delta:+6.1f}%      {acc5_delta:+6.1f}%      {iso_delta:+.4f}")

print()
print("="*80)
print("SUMMARY")
print("="*80)

avg_acc1 = sum(r['retrieval']['improvement_acc1'] for r in results) / len(results) * 100
avg_acc5 = sum(r['retrieval']['improvement_acc5'] for r in results) / len(results) * 100

print(f"Average Acc@1 improvement: {avg_acc1:+.1f}%")
print(f"Average Acc@5 improvement: {avg_acc5:+.1f}%")
print(f"Models tested: {len(results)}")

improvements = [r['retrieval']['improvement_acc5'] for r in results]
if all(imp > 0.05 for imp in improvements):  # >5% improvement
    print()
    print("✅ CONSISTENT IMPROVEMENTS across all models!")
    print("   → Publication-ready result!")
elif sum(1 for imp in improvements if imp > 0.05) >= len(improvements) * 0.66:
    print()
    print("✅ MAJORITY of models show strong improvement")
    print("   → Good publication evidence")
else:
    print()
    print("⚠️  Mixed results - may need further investigation")

print("="*80)
EOF

echo ""
echo "Results saved in results/smoke_multi/"
echo "Summary saved in $SUMMARY_FILE"
echo ""
echo "Next steps:"
echo "  1. Review results above"
echo "  2. If consistent improvements → proceed to pilot/full training"
echo "  3. If mixed → investigate why some models benefit more"
echo ""

