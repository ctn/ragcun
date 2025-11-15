#!/usr/bin/env python3
"""
Evaluate trained models on full MS MARCO dev set (1,939 queries)
Standard MS MARCO evaluation protocol
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from typing import List, Tuple

sys.path.insert(0, str(Path.cwd()))
from ragcun.model import GaussianEmbeddingGemma

def load_msmarco_dev(dev_path: str, max_queries: int = None):
    """Load MS MARCO dev set"""
    with open(dev_path) as f:
        data = json.load(f)
    
    if max_queries:
        data = data[:max_queries]
    
    queries = [d['query'] for d in data]
    positives = [d['positive'] for d in data]
    
    print(f"Loaded {len(queries)} queries from MS MARCO dev set")
    return queries, positives, data

def compute_mrr_at_k(ranks: List[int], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank @ k"""
    mrr = 0.0
    for rank in ranks:
        if rank < k:
            mrr += 1.0 / (rank + 1)
    return mrr / len(ranks)

def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """Compute Recall @ k"""
    return sum(1 for rank in ranks if rank < k) / len(ranks)

def evaluate_model(
    model,
    queries: List[str],
    positives: List[str],
    all_passages: List[str],
    batch_size: int = 32,
    top_k: int = 100,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate model on MS MARCO dev set
    
    For each query:
    - Encode query and all passages
    - Find rank of positive passage
    - Compute metrics
    """
    model.eval()
    ranks = []
    
    print(f"Evaluating on {len(queries)} queries...")
    print(f"Corpus size: {len(all_passages)} passages")
    
    # Encode all passages once (batched for efficiency)
    print("Encoding corpus...")
    passage_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_passages), batch_size), desc="Encoding passages"):
            batch = all_passages[i:i+batch_size]
            emb = model.encode(batch, convert_to_numpy=False)
            passage_embeddings.append(emb)
    
    passage_embeddings = torch.cat(passage_embeddings, dim=0)
    print(f"Corpus embeddings shape: {passage_embeddings.shape}")
    
    # Evaluate each query
    print("Evaluating queries...")
    for i, (query, pos_passage) in enumerate(tqdm(zip(queries, positives), total=len(queries), desc="Queries")):
        with torch.no_grad():
            # Encode query
            q_emb = model.encode(query, convert_to_numpy=False)
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            
            # Compute distances to all passages
            distances = torch.cdist(q_emb, passage_embeddings).squeeze(0)
            
            # Find rank of positive passage
            sorted_indices = distances.argsort()
            pos_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            ranks.append(pos_rank)
    
    # Compute metrics
    mrr_10 = compute_mrr_at_k(ranks, k=10)
    recall_1 = compute_recall_at_k(ranks, k=1)
    recall_5 = compute_recall_at_k(ranks, k=5)
    recall_10 = compute_recall_at_k(ranks, k=10)
    recall_100 = compute_recall_at_k(ranks, k=100)
    
    # Additional stats
    median_rank = np.median(ranks)
    mean_rank = np.mean(ranks)
    
    return {
        'mrr@10': mrr_10,
        'recall@1': recall_1,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'recall@100': recall_100,
        'median_rank': median_rank,
        'mean_rank': mean_rank,
        'total_queries': len(queries)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load MS MARCO dev set
    dev_path = 'data/processed/msmarco/dev.json'
    queries, positives, data = load_msmarco_dev(dev_path)
    
    # All passages form the corpus
    all_passages = positives  # In MS MARCO dev, we use all positive passages as corpus
    
    print(f"Total queries: {len(queries)}")
    print(f"Corpus size: {len(all_passages)}")
    print()
    
    # Models to evaluate
    models_config = [
        ('mpnet', 'sentence-transformers/all-mpnet-base-v2', '110M'),
        ('paraphrase-minilm', 'sentence-transformers/paraphrase-MiniLM-L6-v2', '22M'),
        ('distilroberta', 'sentence-transformers/all-distilroberta-v1', '82M'),
        ('minilm-l6', 'sentence-transformers/all-MiniLM-L6-v2', '22M'),
        ('minilm-l12', 'sentence-transformers/all-MiniLM-L12-v2', '33M'),
    ]
    
    results = []
    
    for model_key, base_model, size in models_config:
        print("="*80)
        print(f"Evaluating: {model_key} ({size})")
        print("="*80)
        
        # Evaluate baseline
        print(f"\n📊 Baseline (λ_iso=0.0)")
        print("-"*80)
        baseline_path = f'checkpoints/smoke_multi/{model_key}_baseline/best_model.pt'
        
        try:
            baseline_model = GaussianEmbeddingGemma.from_pretrained(
                baseline_path,
                base_model=base_model,
                output_dim=512
            ).to(device)
            
            baseline_results = evaluate_model(
                baseline_model,
                queries,
                positives,
                all_passages,
                device=device
            )
            
            print(f"\nResults:")
            print(f"  MRR@10:     {baseline_results['mrr@10']:.4f}")
            print(f"  Recall@1:   {baseline_results['recall@1']:.4f}")
            print(f"  Recall@5:   {baseline_results['recall@5']:.4f}")
            print(f"  Recall@10:  {baseline_results['recall@10']:.4f}")
            print(f"  Recall@100: {baseline_results['recall@100']:.4f}")
            print(f"  Median rank: {baseline_results['median_rank']:.1f}")
            
            # Clear GPU memory
            del baseline_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR loading baseline: {e}")
            baseline_results = None
        
        # Evaluate with isotropy
        print(f"\n📊 With Isotropy (λ_iso=1.0)")
        print("-"*80)
        isotropy_path = f'checkpoints/smoke_multi/{model_key}_isotropy/best_model.pt'
        
        try:
            isotropy_model = GaussianEmbeddingGemma.from_pretrained(
                isotropy_path,
                base_model=base_model,
                output_dim=512
            ).to(device)
            
            isotropy_results = evaluate_model(
                isotropy_model,
                queries,
                positives,
                all_passages,
                device=device
            )
            
            print(f"\nResults:")
            print(f"  MRR@10:     {isotropy_results['mrr@10']:.4f}")
            print(f"  Recall@1:   {isotropy_results['recall@1']:.4f}")
            print(f"  Recall@5:   {isotropy_results['recall@5']:.4f}")
            print(f"  Recall@10:  {isotropy_results['recall@10']:.4f}")
            print(f"  Recall@100: {isotropy_results['recall@100']:.4f}")
            print(f"  Median rank: {isotropy_results['median_rank']:.1f}")
            
            # Clear GPU memory
            del isotropy_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR loading isotropy model: {e}")
            isotropy_results = None
        
        # Compute improvements
        if baseline_results and isotropy_results:
            print(f"\n📈 Improvement")
            print("-"*80)
            print(f"  MRR@10:     {isotropy_results['mrr@10'] - baseline_results['mrr@10']:+.4f}")
            print(f"  Recall@1:   {(isotropy_results['recall@1'] - baseline_results['recall@1'])*100:+.1f}%")
            print(f"  Recall@5:   {(isotropy_results['recall@5'] - baseline_results['recall@5'])*100:+.1f}%")
            print(f"  Recall@10:  {(isotropy_results['recall@10'] - baseline_results['recall@10'])*100:+.1f}%")
            print(f"  Recall@100: {(isotropy_results['recall@100'] - baseline_results['recall@100'])*100:+.1f}%")
            
            results.append({
                'model': model_key,
                'model_name': base_model,
                'size': size,
                'baseline': baseline_results,
                'isotropy': isotropy_results,
                'improvement': {
                    'mrr@10': isotropy_results['mrr@10'] - baseline_results['mrr@10'],
                    'recall@1': isotropy_results['recall@1'] - baseline_results['recall@1'],
                    'recall@5': isotropy_results['recall@5'] - baseline_results['recall@5'],
                    'recall@10': isotropy_results['recall@10'] - baseline_results['recall@10'],
                    'recall@100': isotropy_results['recall@100'] - baseline_results['recall@100'],
                }
            })
        
        print()
    
    # Save results
    output_dir = Path('results/msmarco_full_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*90)
    print("MS MARCO DEV SET EVALUATION SUMMARY (1,939 queries)")
    print("="*90)
    print(f"{'Model':<20} {'Size':<8} {'MRR@10 Δ':<12} {'R@1 Δ':<10} {'R@5 Δ':<10} {'R@10 Δ':<10}")
    print("-"*90)
    
    for r in results:
        model = r['model']
        size = r['size']
        mrr_delta = r['improvement']['mrr@10']
        r1_delta = r['improvement']['recall@1'] * 100
        r5_delta = r['improvement']['recall@5'] * 100
        r10_delta = r['improvement']['recall@10'] * 100
        
        print(f"{model:<20} {size:<8} {mrr_delta:+.4f}       {r1_delta:+5.1f}%     {r5_delta:+5.1f}%     {r10_delta:+5.1f}%")
    
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    avg_mrr = sum(r['improvement']['mrr@10'] for r in results) / len(results)
    avg_r1 = sum(r['improvement']['recall@1'] for r in results) / len(results) * 100
    avg_r5 = sum(r['improvement']['recall@5'] for r in results) / len(results) * 100
    
    print(f"Average MRR@10 improvement: {avg_mrr:+.4f}")
    print(f"Average Recall@1 improvement: {avg_r1:+.1f}%")
    print(f"Average Recall@5 improvement: {avg_r5:+.1f}%")
    
    # Count improvements
    positive_improvements = sum(1 for r in results if r['improvement']['recall@5'] > 0.01)
    
    print(f"\nModels with >1% Recall@5 improvement: {positive_improvements}/{len(results)}")
    
    if positive_improvements >= len(results) * 0.75:
        print("\n✅ STRONG CONSISTENT IMPROVEMENTS - Publication ready!")
    elif positive_improvements >= len(results) * 0.5:
        print("\n✅ MAJORITY show improvements - Good publication evidence")
    else:
        print("\n⚠️  Mixed results - Further investigation needed")
    
    print("="*90)
    print(f"\nResults saved to: {output_dir / 'results.json'}")
    print()

if __name__ == '__main__':
    main()

