#!/usr/bin/env python3
"""
Compare vanilla baseline vs pure SIGReg model on BEIR datasets.
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


def load_sigreg_results(result_file: Path) -> Dict[str, Dict]:
    """Load pure SIGReg results from JSON file."""
    if not result_file.exists():
        return {}
    
    with open(result_file) as f:
        return json.load(f)


def print_comparison_table(baseline: Dict[str, float], sigreg: Dict[str, Dict]):
    """Print a nice comparison table."""
    datasets = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']
    dataset_names = {
        'scifact': 'SciFact',
        'nfcorpus': 'NFCorpus',
        'arguana': 'ArguAna',
        'fiqa': 'FiQA',
        'trec-covid': 'TREC-COVID'
    }
    
    print("="*85)
    print("COMPARISON: Vanilla Baseline vs Pure SIGReg (margin=0.1)")
    print("="*85)
    print()
    print(f"{'Dataset':<15} {'Baseline':<12} {'Pure SIGReg':<12} {'Improvement':<12} {'% Change':<10}")
    print("-"*85)
    
    total_improvement = 0
    count = 0
    
    for dataset in datasets:
        baseline_score = baseline.get(dataset, None)
        sigreg_score = sigreg.get(dataset, {}).get('NDCG@10', None)
        
        if baseline_score is None:
            baseline_str = "-"
        else:
            baseline_str = f"{baseline_score:.4f}"
        
        if sigreg_score is None:
            sigreg_str = "-"
            improvement_str = "-"
            pct_str = "-"
        else:
            sigreg_str = f"{sigreg_score:.4f}"
            if baseline_score is not None:
                improvement = sigreg_score - baseline_score
                pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
                improvement_str = f"{improvement:+.4f}"
                pct_str = f"{pct:+.1f}%"
                total_improvement += improvement
                count += 1
            else:
                improvement_str = "-"
                pct_str = "-"
        
        print(f"{dataset_names[dataset]:<15} {baseline_str:<12} {sigreg_str:<12} {improvement_str:<12} {pct_str:<10}")
    
    print("-"*85)
    if count > 0:
        avg_improvement = total_improvement / count
        print(f"{'Average':<15} {'':<12} {'':<12} {avg_improvement:+.4f} {'':<10}")
    print()


if __name__ == '__main__':
    # Load baseline results
    baseline_log = Path("logs/vanilla_baseline_eval.log")
    baseline = extract_baseline_from_log(baseline_log)
    
    # Load SIGReg results
    sigreg_file = Path("results/beir_standard/pure_sigreg_margin01_full.json")
    if not sigreg_file.exists():
        # Try the partial results file
        sigreg_file = Path("results/beir_standard/pure_sigreg_margin01.json")
    
    sigreg = load_sigreg_results(sigreg_file)
    
    print_comparison_table(baseline, sigreg)
    
    # Also show detailed metrics for SIGReg
    if sigreg:
        print("="*85)
        print("DETAILED METRICS - Pure SIGReg Model")
        print("="*85)
        print()
        for dataset, metrics in sigreg.items():
            print(f"{dataset.upper()}:")
            print(f"  NDCG@10: {metrics.get('NDCG@10', 'N/A'):.4f}")
            print(f"  Recall@10: {metrics.get('Recall@10', 'N/A'):.4f}")
            print(f"  MRR: {metrics.get('MRR', 'N/A'):.4f}")
            print()

