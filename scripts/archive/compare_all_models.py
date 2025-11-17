#!/usr/bin/env python3
"""
Compare all models: Vanilla Baseline, Pure Isotropy, and Pure SIGReg (with contrastive).
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

def extract_baseline_from_log(log_file: Path) -> Dict[str, float]:
    """Extract baseline NDCG@10 results from log file."""
    baseline_results = {}
    
    if not log_file.exists():
        return baseline_results
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    datasets = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']
    for dataset in datasets:
        pattern = rf"Results for.*?{dataset}.*?NDCG@10:\s+([0-9.]+)"
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            baseline_results[dataset] = float(match.group(1))
    
    return baseline_results


def load_results_from_json(result_file: Path) -> Dict[str, Dict]:
    """Load results from JSON file."""
    if not result_file.exists():
        return {}
    
    with open(result_file) as f:
        return json.load(f)


def print_comparison_table(
    baseline: Dict[str, float],
    pure_isotropy: Dict[str, Dict],
    pure_sigreg: Optional[Dict[str, Dict]] = None
):
    """Print a nice comparison table."""
    datasets = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']
    dataset_names = {
        'scifact': 'SciFact',
        'nfcorpus': 'NFCorpus',
        'arguana': 'ArguAna',
        'fiqa': 'FiQA',
        'trec-covid': 'TREC-COVID'
    }
    
    print("="*100)
    print("COMPARISON: Vanilla Baseline vs Pure Isotropy vs Pure SIGReg")
    print("="*100)
    print()
    
    if pure_sigreg:
        header = f"{'Dataset':<15} {'Baseline':>12} {'Pure Isotropy':>15} {'Pure SIGReg':>15} {'Best':>10} {'Δ vs Baseline':>15}"
    else:
        header = f"{'Dataset':<15} {'Baseline':>12} {'Pure Isotropy':>15} {'Best':>10} {'Δ vs Baseline':>15}"
    print(header)
    print("-"*100)
    
    baseline_total = 0
    isotropy_total = 0
    sigreg_total = 0
    completed = 0
    
    for dataset in datasets:
        baseline_score = baseline.get(dataset)
        isotropy_score = pure_isotropy.get(dataset, {}).get('NDCG@10') if pure_isotropy else None
        sigreg_score = pure_sigreg.get(dataset, {}).get('NDCG@10') if pure_sigreg else None
        
        if baseline_score:
            baseline_total += baseline_score
        
        if isotropy_score is not None:
            isotropy_total += isotropy_score
            completed += 1
        
        if sigreg_score is not None:
            sigreg_total += sigreg_score
        
        # Determine best
        scores = []
        if baseline_score:
            scores.append(('Baseline', baseline_score))
        if isotropy_score is not None:
            scores.append(('Pure Iso', isotropy_score))
        if sigreg_score is not None:
            scores.append(('SIGReg', sigreg_score))
        
        best_name = max(scores, key=lambda x: x[1])[0] if scores else '-'
        
        # Calculate delta
        if baseline_score and isotropy_score is not None:
            delta = ((isotropy_score - baseline_score) / baseline_score) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "-"
        
        # Status
        if isotropy_score is None:
            status = "⏳ Pending"
        else:
            status = "✅ Done"
        
        if pure_sigreg:
            sigreg_str = f"{sigreg_score:.4f}" if sigreg_score is not None else "-"
            print(f"{dataset_names[dataset]:<15} "
                  f"{baseline_score if baseline_score else '-':>12} "
                  f"{isotropy_score if isotropy_score is not None else '-':>15} "
                  f"{sigreg_str:>15} "
                  f"{best_name:>10} "
                  f"{delta_str:>15} "
                  f"{status}")
        else:
            print(f"{dataset_names[dataset]:<15} "
                  f"{baseline_score if baseline_score else '-':>12} "
                  f"{isotropy_score if isotropy_score is not None else '-':>15} "
                  f"{best_name:>10} "
                  f"{delta_str:>15} "
                  f"{status}")
    
    print("-"*100)
    
    if completed > 0:
        baseline_avg = baseline_total / len(datasets) if baseline else 0
        isotropy_avg = isotropy_total / completed
        sigreg_avg = sigreg_total / completed if pure_sigreg and completed > 0 else 0
        
        if baseline_avg > 0:
            isotropy_delta = ((isotropy_avg - baseline_avg) / baseline_avg) * 100
            if pure_sigreg and sigreg_avg > 0:
                sigreg_delta = ((sigreg_avg - baseline_avg) / baseline_avg) * 100
                print(f"{'AVERAGE':<15} "
                      f"{baseline_avg:>12.4f} "
                      f"{isotropy_avg:>15.4f} "
                      f"{sigreg_avg:>15.4f} "
                      f"{'':>10} "
                      f"{isotropy_delta:+.1f}%")
            else:
                print(f"{'AVERAGE':<15} "
                      f"{baseline_avg:>12.4f} "
                      f"{isotropy_avg:>15.4f} "
                      f"{'':>10} "
                      f"{isotropy_delta:+.1f}%")
        
        print()
        print(f"✅ Completed: {completed}/{len(datasets)} datasets")
        if baseline_avg > 0:
            if isotropy_avg > baseline_avg:
                print(f"🎉 Pure Isotropy BEATS baseline by {isotropy_delta:.1f}% on average!")
            elif isotropy_avg > baseline_avg * 0.95:
                print(f"✅ Pure Isotropy matches baseline ({isotropy_delta:+.1f}%)")
            else:
                print(f"⚠️  Pure Isotropy below baseline ({isotropy_delta:+.1f}%)")
    
    print("="*100)


if __name__ == '__main__':
    # Load vanilla baseline
    baseline_results = extract_baseline_from_log(Path("logs/vanilla_baseline_eval.log"))
    
    # Load pure isotropy results
    pure_isotropy_results = load_results_from_json(Path("results/beir_standard/pure_isotropy_only.json"))
    
    # Load pure SIGReg results (if available)
    pure_sigreg_results = load_results_from_json(Path("results/beir_standard/pure_sigreg_margin01_full.json"))
    if not pure_sigreg_results:
        pure_sigreg_results = None
    
    print_comparison_table(baseline_results, pure_isotropy_results, pure_sigreg_results)
    
    # Print detailed metrics for pure isotropy if available
    if pure_isotropy_results:
        print("\n" + "="*100)
        print("DETAILED METRICS - Pure Isotropy Model")
        print("="*100)
        print()
        for dataset, metrics in pure_isotropy_results.items():
            print(f"{dataset.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            print()

