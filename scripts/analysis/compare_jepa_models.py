#!/usr/bin/env python3
"""
Compare JEPA models: with vs without contrastive loss
"""

import json
from pathlib import Path

def load_results(filepath):
    """Load BEIR results from JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def main():
    print("=" * 80)
    print("JEPA Model Comparison: With vs Without Contrastive Loss")
    print("=" * 80)
    print()
    
    # Load results
    baseline = load_results('results/beir_standard/mpnet_frozen.json')
    jepa_with_contrastive = load_results('results/beir_standard/jepa_10k.json')
    jepa_pure = load_results('results/beir_standard/jepa_pure.json')
    
    datasets = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']
    
    print(f"{'Dataset':<15} {'Baseline':>12} {'JEPA+Cont':>12} {'JEPA Pure':>12} {'Δ (w/Cont)':>12} {'Δ (Pure)':>12}")
    print("-" * 80)
    
    baseline_scores = []
    with_contrastive_scores = []
    pure_scores = []
    
    for ds in datasets:
        base = baseline.get(ds, {}).get('NDCG@10', None)
        with_cont = jepa_with_contrastive.get(ds, {}).get('NDCG@10', None)
        pure = jepa_pure.get(ds, {}).get('NDCG@10', None)
        
        base_str = f'{base:.4f}' if base is not None else 'N/A'
        with_cont_str = f'{with_cont:.4f}' if with_cont is not None else 'N/A'
        pure_str = f'{pure:.4f}' if pure is not None else 'N/A'
        
        delta_with = f'{with_cont - base:+.4f}' if (with_cont is not None and base is not None) else 'N/A'
        delta_pure = f'{pure - base:+.4f}' if (pure is not None and base is not None) else 'N/A'
        
        # Mark winner
        if with_cont is not None and pure is not None:
            if with_cont > pure:
                with_cont_str = f'✅ {with_cont_str}'
            elif pure > with_cont:
                pure_str = f'✅ {pure_str}'
        
        print(f"{ds:<15} {base_str:>12} {with_cont_str:>12} {pure_str:>12} {delta_with:>12} {delta_pure:>12}")
        
        if base is not None:
            baseline_scores.append(base)
        if with_cont is not None:
            with_contrastive_scores.append(with_cont)
        if pure is not None:
            pure_scores.append(pure)
    
    print()
    print("Averages (completed datasets only):")
    print("-" * 80)
    if baseline_scores:
        print(f"Baseline:           {sum(baseline_scores)/len(baseline_scores):.4f} ({len(baseline_scores)}/{len(datasets)} datasets)")
    if with_contrastive_scores:
        avg_with = sum(with_contrastive_scores)/len(with_contrastive_scores)
        avg_base = sum(baseline_scores[:len(with_contrastive_scores)])/len(with_contrastive_scores) if baseline_scores else 0
        delta_avg = avg_with - avg_base
        print(f"JEPA + Contrastive: {avg_with:.4f} ({len(with_contrastive_scores)}/{len(datasets)} datasets) Δ={delta_avg:+.4f}")
    if pure_scores:
        avg_pure = sum(pure_scores)/len(pure_scores)
        avg_base = sum(baseline_scores[:len(pure_scores)])/len(pure_scores) if baseline_scores else 0
        delta_avg = avg_pure - avg_base
        print(f"JEPA Pure:          {avg_pure:.4f} ({len(pure_scores)}/{len(datasets)} datasets) Δ={delta_avg:+.4f}")
    
    print()
    print("Key Questions:")
    print("-" * 80)
    if with_contrastive_scores and pure_scores:
        avg_with = sum(with_contrastive_scores)/len(with_contrastive_scores)
        avg_pure = sum(pure_scores)/len(pure_scores)
        if avg_with > avg_pure:
            print(f"✅ Contrastive loss HELPS: +{avg_with - avg_pure:.4f} average improvement")
        elif avg_pure > avg_with:
            print(f"✅ Pure JEPA is BETTER: +{avg_pure - avg_with:.4f} average improvement")
        else:
            print("➡️  Contrastive loss has minimal impact")
    else:
        print("⏳ Waiting for evaluation results...")
    
    print()
    print("Training Configuration:")
    print("-" * 80)
    print("JEPA + Contrastive: λ_contrastive=0.1, λ_predictive=1.0, λ_isotropy=1.0")
    print("JEPA Pure:          λ_contrastive=0.0, λ_predictive=1.0, λ_isotropy=1.0")
    print("Both:               Frozen base, 768-dim, stop-gradient enabled")

if __name__ == '__main__':
    main()

