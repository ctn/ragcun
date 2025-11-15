#!/usr/bin/env python3
"""
Analyze smoke test results - loads trained models and compares them
"""
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import GaussianEmbeddingGemma

def measure_isotropy(embeddings):
    """Compute isotropy score (1 = perfectly isotropic)"""
    cov = np.cov(embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero
    
    if len(eigenvalues) > 0:
        isotropy = eigenvalues.min() / eigenvalues.max()
    else:
        isotropy = 0.0
    
    return isotropy, eigenvalues

print("=" * 70)
print("📊 Analyzing Smoke Test Results")
print("=" * 70)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Load models
print("Loading trained models...")
try:
    baseline_model = GaussianEmbeddingGemma.from_pretrained(
        'checkpoints/smoke_baseline/best_model.pt',
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=512
    )
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    print("✅ Loaded baseline model")
except Exception as e:
    print(f"❌ Failed to load baseline model: {e}")
    sys.exit(1)

try:
    isotropy_model = GaussianEmbeddingGemma.from_pretrained(
        'checkpoints/smoke_isotropy/best_model.pt',
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=512
    )
    isotropy_model = isotropy_model.to(device)
    isotropy_model.eval()
    print("✅ Loaded isotropy model")
except Exception as e:
    print(f"❌ Failed to load isotropy model: {e}")
    sys.exit(1)

print()

# Load test data
print("Loading test data...")
with open('data/processed/msmarco_smoke/dev.json') as f:
    dev_data = json.load(f)

# Take 500 examples for quick test
test_data = dev_data[:500]
queries = [item['query'] for item in test_data]
pos_passages = [item['positive'] for item in test_data]

print(f"Testing on {len(queries)} query-passage pairs...")
print()

# Measure isotropy
print("=" * 70)
print("ISOTROPY SCORES (higher = better, 1.0 = perfect)")
print("=" * 70)

with torch.no_grad():
    baseline_embeddings = baseline_model.encode(queries, convert_to_numpy=True)
    isotropy_embeddings = isotropy_model.encode(queries, convert_to_numpy=True)

baseline_isotropy, baseline_eigs = measure_isotropy(baseline_embeddings)
isotropy_isotropy, isotropy_eigs = measure_isotropy(isotropy_embeddings)

print(f"Baseline (λ=0):     {baseline_isotropy:.4f}")
print(f"With isotropy:      {isotropy_isotropy:.4f}")
print(f"Improvement:        {(isotropy_isotropy - baseline_isotropy):+.4f}")
print()

if isotropy_isotropy > baseline_isotropy + 0.01:
    print("✅ Isotropy regularization IS WORKING!")
else:
    print("⚠️  No clear isotropy improvement")

print()
print("Eigenvalue spread (smaller = more isotropic):")
print(f"  Baseline:   {baseline_eigs.max()/baseline_eigs.min():.1f}x")
print(f"  Isotropy:   {isotropy_eigs.max()/isotropy_eigs.min():.1f}x")
print()

# Quick retrieval test
print("=" * 70)
print("RETRIEVAL ACCURACY (on 500 queries)")
print("=" * 70)

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
            q_emb = model.encode(query, convert_to_numpy=False)
            p_embs = model.encode(all_passages, convert_to_numpy=False)
            
            # Compute distances (lower = more similar for Euclidean)
            q_emb = q_emb.unsqueeze(0) if q_emb.dim() == 1 else q_emb
            distances = torch.cdist(q_emb, p_embs).squeeze(0)
            ranks = distances.argsort()
            
            # Check if positive (index 0) is in top-1 or top-5
            pos_rank = (ranks == 0).nonzero(as_tuple=True)[0].item()
            if pos_rank == 0:
                correct_at_1 += 1
                correct_at_5 += 1
            elif pos_rank < 5:
                correct_at_5 += 1
    
    return correct_at_1 / 100, correct_at_5 / 100

print("Testing retrieval accuracy (this may take a moment)...")
baseline_acc1, baseline_acc5 = test_retrieval(baseline_model, queries, pos_passages)
isotropy_acc1, isotropy_acc5 = test_retrieval(isotropy_model, queries, pos_passages)

print()
print(f"Accuracy@1:")
print(f"  Baseline:   {baseline_acc1*100:.1f}%")
print(f"  Isotropy:   {isotropy_acc1*100:.1f}%")
print(f"  Δ:          {(isotropy_acc1-baseline_acc1)*100:+.1f}%")
print()
print(f"Accuracy@5:")
print(f"  Baseline:   {baseline_acc5*100:.1f}%")
print(f"  Isotropy:   {isotropy_acc5*100:.1f}%")
print(f"  Δ:          {(isotropy_acc5-baseline_acc5)*100:+.1f}%")
print()

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

Path('results/smoke_test').mkdir(parents=True, exist_ok=True)
with open('results/smoke_test/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("=" * 70)
print("VERDICT")
print("=" * 70)

isotropy_improved = isotropy_isotropy > baseline_isotropy + 0.01
retrieval_improved = isotropy_acc1 > baseline_acc1

if isotropy_improved and retrieval_improved:
    print("✅ SUCCESS: Isotropy helps!")
    print()
    print("Your method is working:")
    print("  1. LeJEPA loss improves isotropy ✓")
    print("  2. Better isotropy → better retrieval ✓")
    print()
    print("→ Ready for full training!")
elif isotropy_improved:
    print("⚠️  PARTIAL: Isotropy improves but retrieval unclear")
    print()
    print("Consider:")
    print("  - More training steps")
    print("  - Tune lambda_isotropy")
elif retrieval_improved:
    print("⚠️  UNEXPECTED: Retrieval improves without isotropy gain")
    print()
    print("This might mean:")
    print("  - Regularization helps in other ways")
    print("  - Need more data to see isotropy effect")
else:
    print("❌ ISSUE: No clear improvement")
    print()
    print("Debug steps:")
    print("  1. Check training logs for loss curves")
    print("  2. Verify lambda_isotropy > 0 in training")
    print("  3. Try higher lambda_isotropy (e.g., 2.0)")

print("=" * 70)
print()
print(f"Results saved: results/smoke_test/results.json")

