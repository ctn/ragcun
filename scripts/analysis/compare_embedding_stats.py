#!/usr/bin/env python3
"""
Compare embedding statistics between two models.
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import IsotropicGaussianEncoder

def compute_stats(embeddings, name=""):
    """Compute embedding statistics."""
    embeddings = embeddings.detach().cpu().numpy()
    
    # Basic stats
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    
    # Global stats
    global_mean = np.mean(embeddings)
    global_std = np.std(embeddings)
    
    # Norms
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # Covariance
    cov = np.cov(embeddings.T)
    cov_trace = np.trace(cov)
    cov_frobenius = np.linalg.norm(cov, 'fro')
    
    # Distance to identity (isotropy measure)
    identity = np.eye(embeddings.shape[1]) * global_std**2
    isotropy_deviation = np.linalg.norm(cov - identity, 'fro')
    
    # Pairwise distances (sample)
    n_samples = min(1000, len(embeddings))
    sample_emb = embeddings[:n_samples]
    pairwise_dists = []
    for i in range(min(100, n_samples)):
        for j in range(i+1, min(100, n_samples)):
            dist = np.linalg.norm(sample_emb[i] - sample_emb[j])
            pairwise_dists.append(dist)
    
    mean_pairwise_dist = np.mean(pairwise_dists) if pairwise_dists else 0
    std_pairwise_dist = np.std(pairwise_dists) if pairwise_dists else 0
    
    stats = {
        'name': name,
        'n_samples': len(embeddings),
        'dim': embeddings.shape[1],
        'global_mean': global_mean,
        'global_std': global_std,
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'cov_trace': cov_trace,
        'cov_frobenius': cov_frobenius,
        'isotropy_deviation': isotropy_deviation,
        'mean_pairwise_dist': mean_pairwise_dist,
        'std_pairwise_dist': std_pairwise_dist,
        'min_norm': np.min(norms),
        'max_norm': np.max(norms),
        'min_std': np.min(std),
        'max_std': np.max(std),
    }
    
    return stats

def print_stats(stats):
    """Print statistics in a readable format."""
    print(f"\n{'='*80}")
    print(f"Embedding Statistics: {stats['name']}")
    print(f"{'='*80}")
    print(f"  Samples: {stats['n_samples']:,}")
    print(f"  Dimension: {stats['dim']}")
    print()
    print(f"  Global Statistics:")
    print(f"    Mean: {stats['global_mean']:.6f}")
    print(f"    Std:  {stats['global_std']:.6f}")
    print()
    print(f"  Norm Statistics:")
    print(f"    Mean: {stats['mean_norm']:.6f}")
    print(f"    Std:  {stats['std_norm']:.6f}")
    print(f"    Min:  {stats['min_norm']:.6f}")
    print(f"    Max:  {stats['max_norm']:.6f}")
    print()
    print(f"  Dimension-wise Std:")
    print(f"    Min:  {stats['min_std']:.6f}")
    print(f"    Max:  {stats['max_std']:.6f}")
    print()
    print(f"  Covariance:")
    print(f"    Trace: {stats['cov_trace']:.6f}")
    print(f"    Frobenius norm: {stats['cov_frobenius']:.6f}")
    print(f"    Isotropy deviation: {stats['isotropy_deviation']:.6f}")
    print()
    print(f"  Pairwise Distances (sample):")
    print(f"    Mean: {stats['mean_pairwise_dist']:.6f}")
    print(f"    Std:  {stats['std_pairwise_dist']:.6f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample texts for encoding
    sample_texts = [
        "What is machine learning?",
        "How does neural network training work?",
        "Explain deep learning architectures.",
        "What are transformers in NLP?",
        "Describe attention mechanisms.",
        "How do language models work?",
        "What is natural language processing?",
        "Explain word embeddings.",
        "What is information retrieval?",
        "How does semantic search work?",
        "Describe vector databases.",
        "What is RAG (Retrieval Augmented Generation)?",
        "Explain document embeddings.",
        "How does cosine similarity work?",
        "What is the difference between dense and sparse embeddings?",
    ] * 20  # 300 samples
    
    print("="*80)
    print("EMBEDDING STATISTICS COMPARISON")
    print("="*80)
    print(f"\nEncoding {len(sample_texts)} sample texts...")
    print(f"Device: {device}")
    
    # Load self-supervised model
    print("\n" + "="*80)
    print("Loading Self-Supervised X/Y-Masked Model")
    print("="*80)
    model_self = IsotropicGaussianEncoder.from_pretrained(
        'checkpoints/jepa_xy_masked/best_model.pt',
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=768,
        freeze_base=True,
        use_predictor=True
    )
    model_self = model_self.to(device)
    model_self.eval()
    
    with torch.no_grad():
        emb_self = model_self.encode(sample_texts, batch_size=32, convert_to_numpy=False)
        emb_self = torch.stack(emb_self) if isinstance(emb_self, list) else emb_self
    
    stats_self = compute_stats(emb_self, "Self-Supervised X/Y-Masked")
    print_stats(stats_self)
    
    # Load supervised fine-tuned model
    print("\n" + "="*80)
    print("Loading Supervised Fine-Tuned Model")
    print("="*80)
    model_sup = IsotropicGaussianEncoder.from_pretrained(
        'checkpoints/jepa_supervised_finetuned/best_model.pt',
        base_model='sentence-transformers/all-mpnet-base-v2',
        output_dim=768,
        freeze_base=True,
        use_predictor=False
    )
    model_sup = model_sup.to(device)
    model_sup.eval()
    
    with torch.no_grad():
        emb_sup = model_sup.encode(sample_texts, batch_size=32, convert_to_numpy=False)
        emb_sup = torch.stack(emb_sup) if isinstance(emb_sup, list) else emb_sup
    
    stats_sup = compute_stats(emb_sup, "Supervised Fine-Tuned")
    print_stats(stats_sup)
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {'Self-Supervised':>20} {'Supervised':>20} {'Ratio':>15}")
    print("-"*80)
    
    metrics = [
        ('Global Std', 'global_std'),
        ('Mean Norm', 'mean_norm'),
        ('Std Norm', 'std_norm'),
        ('Isotropy Deviation', 'isotropy_deviation'),
        ('Mean Pairwise Dist', 'mean_pairwise_dist'),
        ('Covariance Trace', 'cov_trace'),
        ('Covariance Frobenius', 'cov_frobenius'),
    ]
    
    for metric_name, metric_key in metrics:
        self_val = stats_self[metric_key]
        sup_val = stats_sup[metric_key]
        ratio = sup_val / self_val if self_val != 0 else float('inf')
        print(f"{metric_name:<30} {self_val:>20.6f} {sup_val:>20.6f} {ratio:>15.3f}x")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    std_ratio = stats_sup['global_std'] / stats_self['global_std'] if stats_self['global_std'] > 0 else 0
    norm_ratio = stats_sup['mean_norm'] / stats_self['mean_norm'] if stats_self['mean_norm'] > 0 else 0
    dist_ratio = stats_sup['mean_pairwise_dist'] / stats_self['mean_pairwise_dist'] if stats_self['mean_pairwise_dist'] > 0 else 0
    
    print(f"\n1. Embedding Scale:")
    print(f"   Self-supervised std: {stats_self['global_std']:.6f}")
    print(f"   Supervised std:      {stats_sup['global_std']:.6f}")
    print(f"   Ratio: {std_ratio:.3f}x ({'COLLAPSED' if std_ratio < 0.1 else 'NORMAL' if std_ratio > 0.5 else 'SMALL'})")
    
    print(f"\n2. Embedding Norms:")
    print(f"   Self-supervised: {stats_self['mean_norm']:.6f}")
    print(f"   Supervised:      {stats_sup['mean_norm']:.6f}")
    print(f"   Ratio: {norm_ratio:.3f}x")
    
    print(f"\n3. Pairwise Distances:")
    print(f"   Self-supervised: {stats_self['mean_pairwise_dist']:.6f}")
    print(f"   Supervised:       {stats_sup['mean_pairwise_dist']:.6f}")
    print(f"   Ratio: {dist_ratio:.3f}x")
    
    print(f"\n4. Isotropy:")
    print(f"   Self-supervised deviation: {stats_self['isotropy_deviation']:.6f}")
    print(f"   Supervised deviation:      {stats_sup['isotropy_deviation']:.6f}")
    
    if std_ratio < 0.1:
        print("\n⚠️  WARNING: Supervised model shows severe embedding collapse!")
        print("   Embeddings are ~{:.1f}x smaller than self-supervised model.".format(1/std_ratio))
        print("   This explains the terrible retrieval performance.")

if __name__ == '__main__':
    main()

