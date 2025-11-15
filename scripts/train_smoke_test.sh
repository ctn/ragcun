#!/bin/bash
# Smoke Test: Prove isotropy helps in ~1-2 hours
#
# This doesn't aim for publication-quality results, just answers:
#   1. Does LeJEPA loss actually improve isotropy? 
#   2. Does improved isotropy help retrieval?
#   3. Is our implementation working correctly?
#
# Expected time: 1-2 hours on T4

set -e

echo "============================================"
echo "🔬 Smoke Test: Validate Isotropy in 2 Hours"
echo "============================================"
echo ""
echo "Goal: Quick proof that isotropy regularization works"
echo "Strategy: Train on 10K examples, measure isotropy directly"
echo ""

# Create logs directory
mkdir -p logs results/smoke_test

# Check for MS MARCO
if [ ! -f "data/processed/msmarco/train.json" ]; then
    echo "❌ MS MARCO data not found"
    echo ""
    echo "Quick download (just what we need):"
    echo "  python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    exit 1
fi

# Create tiny dataset
echo "Step 1: Creating smoke test dataset (10K examples)..."
python << 'EOF'
import json
from pathlib import Path

# Load MS MARCO
with open('data/processed/msmarco/train.json') as f:
    full_data = json.load(f)

# Take 10K for training, 1K for dev
smoke_train = full_data[:10000]
smoke_dev = full_data[500000:501000] if len(full_data) > 500000 else full_data[-1000:]

# Save
Path('data/processed/msmarco_smoke').mkdir(parents=True, exist_ok=True)
with open('data/processed/msmarco_smoke/train.json', 'w') as f:
    json.dump(smoke_train, f)
with open('data/processed/msmarco_smoke/dev.json', 'w') as f:
    json.dump(smoke_dev, f)

print(f"✅ Smoke dataset: {len(smoke_train):,} train, {len(smoke_dev):,} dev")
EOF

# Common training args
COMMON="--train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 1 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 100 \
    --mixed_precision \
    --log_interval 50"

echo ""
echo "============================================"
echo "Experiment 1: Baseline (λ_iso=0)"
echo "Started: $(date +'%H:%M:%S')"
echo "============================================"

python scripts/train.py \
    $COMMON \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 1e-3 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --output_dir checkpoints/smoke_baseline \
    2>&1 | tee logs/smoke_baseline.log

echo "✅ Baseline done: $(date +'%H:%M:%S')"

# Clear GPU cache
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✅ GPU cache cleared')"

echo ""
echo "============================================"
echo "Experiment 2: With Isotropy (λ_iso=1.0)"
echo "Started: $(date +'%H:%M:%S')"
echo "============================================"

python scripts/train.py \
    $COMMON \
    --base_learning_rate 2e-5 \
    --projection_learning_rate 1e-3 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/smoke_isotropy \
    2>&1 | tee logs/smoke_isotropy.log

echo "✅ Isotropy done: $(date +'%H:%M:%S')"

# Quick analysis
echo ""
echo "============================================"
echo "📊 Quick Analysis"
echo "============================================"
echo ""

python << 'EOF'
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import GaussianEmbeddingGemma
from ragcun.retriever import GaussianRetriever
from torch.utils.data import DataLoader, Dataset

print("Loading models...")

# Load both models (specify base_model to match training)
baseline_model = GaussianEmbeddingGemma.from_pretrained(
    'checkpoints/smoke_baseline/best_model.pt',
    base_model='sentence-transformers/all-mpnet-base-v2',
    output_dim=512
)
isotropy_model = GaussianEmbeddingGemma.from_pretrained(
    'checkpoints/smoke_isotropy/best_model.pt',
    base_model='sentence-transformers/all-mpnet-base-v2',
    output_dim=512
)

baseline_model.eval()
isotropy_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
baseline_model = baseline_model.to(device)
isotropy_model = isotropy_model.to(device)

print("Loading test data...")

# Load dev set
with open('data/processed/msmarco_smoke/dev.json') as f:
    dev_data = json.load(f)

# Take 500 examples for quick test
test_data = dev_data[:500]

# Extract queries and passages
queries = [item['query'] for item in test_data]
pos_passages = [item['positive'] for item in test_data]

print(f"Testing on {len(queries)} query-passage pairs...")
print("")

# Measure isotropy
def measure_isotropy(model, texts):
    """Compute isotropy score (1 = perfectly isotropic)"""
    with torch.no_grad():
        embeddings = []
        for text in texts:
            inputs = model.tokenizer(text, return_tensors='pt', 
                                    padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            embeddings.append(outputs['mean'].cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Compute isotropy: ratio of smallest/largest eigenvalue of covariance
        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero
        
        if len(eigenvalues) > 0:
            isotropy = eigenvalues.min() / eigenvalues.max()
        else:
            isotropy = 0.0
        
        return isotropy, eigenvalues

# Measure both models
baseline_isotropy, baseline_eigs = measure_isotropy(baseline_model, queries)
isotropy_isotropy, isotropy_eigs = measure_isotropy(isotropy_model, queries)

print("=" * 60)
print("ISOTROPY SCORES (higher = better, 1.0 = perfect)")
print("=" * 60)
print(f"Baseline (λ=0):     {baseline_isotropy:.4f}")
print(f"With isotropy:      {isotropy_isotropy:.4f}")
print(f"Improvement:        {(isotropy_isotropy - baseline_isotropy):.4f}")
print("")

if isotropy_isotropy > baseline_isotropy + 0.01:
    print("✅ Isotropy regularization IS WORKING!")
else:
    print("⚠️  No clear isotropy improvement")

print("")
print("Eigenvalue spread (smaller = more isotropic):")
print(f"  Baseline:   {baseline_eigs.max()/baseline_eigs.min():.1f}x")
print(f"  Isotropy:   {isotropy_eigs.max()/isotropy_eigs.min():.1f}x")
print("")

# Quick retrieval test
print("=" * 60)
print("RETRIEVAL ACCURACY (on 500 queries)")
print("=" * 60)

def test_retrieval(model, queries, pos_passages):
    """Test if model ranks positive passage highly"""
    correct_at_1 = 0
    correct_at_5 = 0
    
    with torch.no_grad():
        for i, (query, pos_passage) in enumerate(zip(queries[:100], pos_passages[:100])):
            # Create some negatives (use other positives as distractors)
            negatives = [pos_passages[j] for j in range(len(pos_passages)) if j != i][:9]
            all_passages = [pos_passage] + negatives
            
            # Encode
            query_inputs = model.tokenizer(query, return_tensors='pt', 
                                          padding=True, truncation=True, max_length=512)
            query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
            q_out = model(**query_inputs)
            
            passage_inputs = model.tokenizer(all_passages, return_tensors='pt',
                                            padding=True, truncation=True, max_length=512)
            passage_inputs = {k: v.to(device) for k, v in passage_inputs.items()}
            p_out = model(**passage_inputs)
            
            # Compute distances (lower = more similar for Euclidean)
            q_mean = q_out['mean']
            p_means = p_out['mean']
            
            distances = torch.cdist(q_mean, p_means).squeeze(0)
            ranks = distances.argsort()
            
            # Check if positive (index 0) is in top-1 or top-5
            pos_rank = (ranks == 0).nonzero(as_tuple=True)[0].item()
            if pos_rank == 0:
                correct_at_1 += 1
                correct_at_5 += 1
            elif pos_rank < 5:
                correct_at_5 += 1
    
    return correct_at_1 / 100, correct_at_5 / 100

baseline_acc1, baseline_acc5 = test_retrieval(baseline_model, queries, pos_passages)
isotropy_acc1, isotropy_acc5 = test_retrieval(isotropy_model, queries, pos_passages)

print(f"Accuracy@1:")
print(f"  Baseline:   {baseline_acc1*100:.1f}%")
print(f"  Isotropy:   {isotropy_acc1*100:.1f}%")
print(f"  Δ:          {(isotropy_acc1-baseline_acc1)*100:+.1f}%")
print("")
print(f"Accuracy@5:")
print(f"  Baseline:   {baseline_acc5*100:.1f}%")
print(f"  Isotropy:   {isotropy_acc5*100:.1f}%")
print(f"  Δ:          {(isotropy_acc5-baseline_acc5)*100:+.1f}%")
print("")

# Save results
results = {
    'isotropy_scores': {
        'baseline': float(baseline_isotropy),
        'with_regularization': float(isotropy_isotropy),
        'improvement': float(isotropy_isotropy - baseline_isotropy)
    },
    'eigenvalue_spread': {
        'baseline': float(baseline_eigs.max()/baseline_eigs.min()),
        'with_regularization': float(isotropy_eigs.max()/isotropy_eigs.min())
    },
    'retrieval_accuracy': {
        'baseline_acc1': float(baseline_acc1),
        'baseline_acc5': float(baseline_acc5),
        'isotropy_acc1': float(isotropy_acc1),
        'isotropy_acc5': float(isotropy_acc5),
        'improvement_acc1': float(isotropy_acc1 - baseline_acc1),
        'improvement_acc5': float(isotropy_acc5 - baseline_acc5)
    }
}

with open('results/smoke_test/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("VERDICT")
print("=" * 60)

isotropy_improved = isotropy_isotropy > baseline_isotropy + 0.01
retrieval_improved = isotropy_acc1 > baseline_acc1

if isotropy_improved and retrieval_improved:
    print("✅ SUCCESS: Isotropy helps!")
    print("")
    print("Your method is working:")
    print("  1. LeJEPA loss improves isotropy ✓")
    print("  2. Better isotropy → better retrieval ✓")
    print("")
    print("→ Ready for full training!")
elif isotropy_improved:
    print("⚠️  PARTIAL: Isotropy improves but retrieval unclear")
    print("")
    print("Consider:")
    print("  - More training steps")
    print("  - Tune lambda_isotropy")
elif retrieval_improved:
    print("⚠️  UNEXPECTED: Retrieval improves without isotropy gain")
    print("")
    print("This might mean:")
    print("  - Regularization helps in other ways")
    print("  - Need more data to see isotropy effect")
else:
    print("❌ ISSUE: No clear improvement")
    print("")
    print("Debug steps:")
    print("  1. Check logs/smoke_*.log for loss curves")
    print("  2. Verify lambda_isotropy > 0 in training")
    print("  3. Try higher lambda_isotropy (e.g., 2.0)")

print("=" * 60)

EOF

echo ""
echo "============================================"
echo "✅ Smoke Test Complete!"
echo "============================================"
echo ""
echo "Time: ~1-2 hours"
echo "Results: results/smoke_test/results.json"
echo ""
echo "Next steps:"
echo "  - If ✅: Run pilot or full training"
echo "  - If ⚠️: Tune hyperparameters and re-run"
echo "  - If ❌: Debug before full training"
echo ""

