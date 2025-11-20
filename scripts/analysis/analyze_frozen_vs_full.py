#!/usr/bin/env python3
"""
Analyze frozen base vs full fine-tuning results (FIXED VERSION)
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from ragcun.model import IsotropicGaussianEncoder

def measure_isotropy(embeddings):
    """Measure isotropy score"""
    cov = np.cov(embeddings.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) > 0:
        return eigenvalues.min() / eigenvalues.max()
    return 0.0

def test_retrieval(model, queries, passages, device='cuda'):
    """
    Test retrieval accuracy
    
    For each query:
    - Positive passage is at the same index
    - Negative passages are all OTHER passages
    """
    correct_at_1 = 0
    correct_at_5 = 0
    
    with torch.no_grad():
        # Encode all passages once
        all_passages_emb = model.encode(passages, convert_to_numpy=False)
        all_passages_emb = all_passages_emb.to(device)
        
        # Test each query
        for i, query in enumerate(queries[:100]):  # Test on 100 queries
            # Encode query
            q_emb = model.encode(query, convert_to_numpy=False).to(device)
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            
            # Compute distances to ALL passages
            distances = torch.cdist(q_emb, all_passages_emb).squeeze(0)
            
            # Sort by distance (closest first)
            sorted_indices = distances.argsort()
            
            # Find rank of the positive passage (index i)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            
            if rank == 0:
                correct_at_1 += 1
                correct_at_5 += 1
            elif rank < 5:
                correct_at_5 += 1
    
    return correct_at_1 / 100, correct_at_5 / 100

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load test data
    with open('data/processed/msmarco_smoke/dev.json') as f:
        data = json.load(f)[:500]
    
    queries = [d['query'] for d in data]
    passages = [d['positive'] for d in data]
    
    print(f"Loaded {len(queries)} query-passage pairs\n")
    
    models_config = {
        'mpnet': ('sentence-transformers/all-mpnet-base-v2', '110M'),
        'minilm-l6': ('sentence-transformers/all-MiniLM-L6-v2', '22M'),
        'minilm-l12': ('sentence-transformers/all-MiniLM-L12-v2', '33M'),
        'distilroberta': ('sentence-transformers/all-distilroberta-v1', '82M'),
        'paraphrase-minilm': ('sentence-transformers/paraphrase-MiniLM-L6-v2', '22M'),
    }
    
    results = []
    
    for key, (base_model, size) in models_config.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {key} ({size})")
        print(f"{'='*80}\n")
        
        # Load models
        try:
            baseline = IsotropicGaussianEncoder.from_pretrained(
                f'checkpoints/smoke_multi/{key}_baseline/best_model.pt',
                base_model=base_model,
                output_dim=512
            )
        except Exception as e:
            print(f"⚠️  Baseline not found, skipping: {e}\n")
            continue
        
        full_finetune = IsotropicGaussianEncoder.from_pretrained(
            f'checkpoints/smoke_multi/{key}_isotropy/best_model.pt',
            base_model=base_model,
            output_dim=512
        )
        
        frozen = IsotropicGaussianEncoder.from_pretrained(
            f'checkpoints/smoke_frozen/{key}_frozen_isotropy/best_model.pt',
            base_model=base_model,
            output_dim=512,
            freeze_base=True  # Important!
        )
        
        # Move models to device
        baseline = baseline.to(device).eval()
        full_finetune = full_finetune.to(device).eval()
        frozen = frozen.to(device).eval()
        
        # Measure isotropy
        print("Measuring isotropy...")
        with torch.no_grad():
            base_emb = baseline.encode(queries, convert_to_numpy=True)
            full_emb = full_finetune.encode(queries, convert_to_numpy=True)
            frozen_emb = frozen.encode(queries, convert_to_numpy=True)
        
        base_iso = measure_isotropy(base_emb)
        full_iso = measure_isotropy(full_emb)
        frozen_iso = measure_isotropy(frozen_emb)
        
        print(f"  Baseline:      {base_iso:.6f}")
        print(f"  Full finetune: {full_iso:.6f}")
        print(f"  Frozen base:   {frozen_iso:.6f}")
        
        # Test retrieval
        print("\nTesting retrieval accuracy...")
        base_acc1, base_acc5 = test_retrieval(baseline, queries, passages, device)
        print(f"  Baseline:      Acc@1={base_acc1*100:.1f}%, Acc@5={base_acc5*100:.1f}%")
        
        full_acc1, full_acc5 = test_retrieval(full_finetune, queries, passages, device)
        print(f"  Full finetune: Acc@1={full_acc1*100:.1f}%, Acc@5={full_acc5*100:.1f}%")
        
        frozen_acc1, frozen_acc5 = test_retrieval(frozen, queries, passages, device)
        print(f"  Frozen base:   Acc@1={frozen_acc1*100:.1f}%, Acc@5={frozen_acc5*100:.1f}%")
        
        # Save results
        result = {
            'model': key,
            'size': size,
            'baseline': {'acc1': base_acc1, 'acc5': base_acc5, 'iso': base_iso},
            'full_finetune': {'acc1': full_acc1, 'acc5': full_acc5, 'iso': full_iso},
            'frozen_base': {'acc1': frozen_acc1, 'acc5': frozen_acc5, 'iso': frozen_iso},
        }
        results.append(result)
        
        # Save individual result
        Path('results/smoke_frozen_fixed').mkdir(parents=True, exist_ok=True)
        with open(f'results/smoke_frozen_fixed/{key}_comparison.json', 'w') as f:
            json.dump(result, f, indent=2)
        
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
        
        winner = "Full FT" if full_delta > frozen_delta else "Frozen" if frozen_delta > full_delta else "Tie"
        
        print(f"{model:<20} {size:<8} {base_acc5:5.1f}%          "
              f"{full_acc5:5.1f}% ({full_delta:+.1f}%)   "
              f"{frozen_acc5:5.1f}% ({frozen_delta:+.1f}%)   {winner}")
    
    # Summary
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
    
    efficiency = (avg_frozen_improvement / avg_full_improvement * 100) if avg_full_improvement > 0 else 0
    
    if abs(avg_full_improvement - avg_frozen_improvement) < 1.0:
        print("✅ FROZEN BASE is competitive with full fine-tuning!")
        print(f"   → {efficiency:.0f}% of performance with ~3x faster training")
    elif avg_frozen_improvement > avg_full_improvement:
        print("✅ FROZEN BASE outperforms full fine-tuning!")
        print("   → Projection alone is sufficient (and better!)")
    else:
        print("✅ FULL FINE-TUNING provides meaningful improvement")
        print(f"   → {avg_full_improvement - avg_frozen_improvement:.1f}% better")
        print(f"   → Frozen base achieves {efficiency:.0f}% of full FT gains")
    
    print("="*90)
    
    # Save summary
    with open('results/smoke_frozen_fixed/summary.json', 'w') as f:
        json.dump({
            'results': results,
            'avg_full_improvement': float(avg_full_improvement),
            'avg_frozen_improvement': float(avg_frozen_improvement),
        }, f, indent=2)
    
    print(f"\nResults saved to results/smoke_frozen_fixed/")

if __name__ == '__main__':
    main()

