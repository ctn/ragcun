#!/usr/bin/env python3
"""
Evaluate frozen base models on MS MARCO full dev set (1,939 queries)
Same protocol as the baseline/full finetune evaluation
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))
from ragcun.model import IsotropicGaussianEncoder

def compute_mrr_at_k(ranks, k=10):
    """Compute Mean Reciprocal Rank @ k"""
    mrr = 0.0
    for rank in ranks:
        if rank < k:
            mrr += 1.0 / (rank + 1)
    return mrr / len(ranks)

def compute_recall_at_k(ranks, k):
    """Compute Recall @ k"""
    return sum(1 for rank in ranks if rank < k) / len(ranks)

def evaluate_model(model, queries, positives, device='cuda'):
    """
    Evaluate model on MS MARCO dev set
    
    For each query:
    - Encode query and all passages
    - Find rank of positive passage
    - Compute metrics
    """
    model.to(device).eval()
    ranks = []
    
    print(f"Evaluating on {len(queries)} queries...")
    print(f"Corpus size: {len(positives)} passages")
    
    # Encode all passages once (batched for efficiency)
    print("Encoding corpus...")
    passage_embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(positives), batch_size), desc="Encoding passages"):
            batch = positives[i:i+batch_size]
            emb = model.encode(batch, convert_to_numpy=False)
            passage_embeddings.append(emb)
    
    passage_embeddings = torch.cat(passage_embeddings, dim=0)
    print(f"Corpus embeddings shape: {passage_embeddings.shape}")
    
    # Evaluate each query
    print("Evaluating queries...")
    for i, query in enumerate(tqdm(queries, desc="Queries")):
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
    with open(dev_path) as f:
        data = json.load(f)
    
    queries = [d['query'] for d in data]
    positives = [d['positive'] for d in data]
    
    print(f"Total queries: {len(queries)}")
    print(f"Corpus size: {len(positives)}")
    print()
    
    # Models to evaluate
    models_config = [
        ('mpnet', 'sentence-transformers/all-mpnet-base-v2', '110M'),
        ('minilm-l6', 'sentence-transformers/all-MiniLM-L6-v2', '22M'),
        ('minilm-l12', 'sentence-transformers/all-MiniLM-L12-v2', '33M'),
        ('distilroberta', 'sentence-transformers/all-distilroberta-v1', '82M'),
        ('paraphrase-minilm', 'sentence-transformers/paraphrase-MiniLM-L6-v2', '22M'),
    ]
    
    results = []
    
    for model_key, base_model, size in models_config:
        print("="*80)
        print(f"Evaluating: {model_key} ({size})")
        print("="*80)
        print()
        
        # Evaluate frozen base model
        frozen_path = f'checkpoints/smoke_frozen/{model_key}_frozen_isotropy/best_model.pt'
        
        try:
            frozen_model = IsotropicGaussianEncoder.from_pretrained(
                frozen_path,
                base_model=base_model,
                output_dim=512,
                freeze_base=True
            )
            
            frozen_results = evaluate_model(
                frozen_model,
                queries,
                positives,
                device=device
            )
            
            print(f"\nFrozen Base Results:")
            print(f"  MRR@10:     {frozen_results['mrr@10']:.4f}")
            print(f"  Recall@1:   {frozen_results['recall@1']:.4f}")
            print(f"  Recall@5:   {frozen_results['recall@5']:.4f}")
            print(f"  Recall@10:  {frozen_results['recall@10']:.4f}")
            print(f"  Recall@100: {frozen_results['recall@100']:.4f}")
            print(f"  Median rank: {frozen_results['median_rank']:.1f}")
            print()
            
            # Load existing baseline and full FT results
            with open(f'results/msmarco_full_eval/results.json') as f:
                existing_results = json.load(f)
            
            # Find this model's baseline and full FT results
            model_existing = next((r for r in existing_results if r['model'] == model_key), None)
            
            if model_existing:
                baseline_results = model_existing['baseline']
                full_ft_results = model_existing['isotropy']
                
                print(f"Comparison:")
                print(f"  Baseline:    MRR@10={baseline_results['mrr@10']:.4f}, R@5={baseline_results['recall@5']:.4f}")
                print(f"  Full FT:     MRR@10={full_ft_results['mrr@10']:.4f}, R@5={full_ft_results['recall@5']:.4f}")
                print(f"  Frozen:      MRR@10={frozen_results['mrr@10']:.4f}, R@5={frozen_results['recall@5']:.4f}")
                print()
                print(f"Improvements vs Baseline:")
                print(f"  Full FT:  +{(full_ft_results['recall@5'] - baseline_results['recall@5'])*100:.1f}% R@5")
                print(f"  Frozen:   +{(frozen_results['recall@5'] - baseline_results['recall@5'])*100:.1f}% R@5")
                print()
            
            results.append({
                'model': model_key,
                'model_name': base_model,
                'size': size,
                'frozen': frozen_results
            })
            
            # Clear GPU memory
            del frozen_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR evaluating {model_key}: {e}")
            print()
            continue
    
    # Save results
    output_dir = Path('results/frozen_msmarco_full_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'frozen_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate comparison table
    print("\n" + "="*90)
    print("MS MARCO DEV SET - FROZEN BASE vs FULL FINE-TUNE COMPARISON")
    print("="*90)
    
    # Load existing results
    with open('results/msmarco_full_eval/results.json') as f:
        existing_results = json.load(f)
    
    print(f"{'Model':<20} {'Size':<8} {'Baseline':<12} {'Full FT':<12} {'Frozen':<12} {'Winner':<10}")
    print("-"*90)
    
    for r in results:
        model = r['model']
        size = r['size']
        frozen_r5 = r['frozen']['recall@5'] * 100
        
        model_existing = next((e for e in existing_results if e['model'] == model), None)
        if model_existing:
            base_r5 = model_existing['baseline']['recall@5'] * 100
            full_r5 = model_existing['isotropy']['recall@5'] * 100
            
            full_delta = full_r5 - base_r5
            frozen_delta = frozen_r5 - base_r5
            
            if frozen_delta > full_delta:
                winner = "Frozen"
            elif full_delta > frozen_delta:
                winner = "Full FT"
            else:
                winner = "Tie"
            
            print(f"{model:<20} {size:<8} {base_r5:5.1f}%       "
                  f"{full_r5:5.1f}% ({full_delta:+.1f}%) {frozen_r5:5.1f}% ({frozen_delta:+.1f}%) {winner}")
    
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    # Compute averages
    avg_frozen_improvement = 0
    avg_full_improvement = 0
    count = 0
    
    for r in results:
        model_existing = next((e for e in existing_results if e['model'] == r['model']), None)
        if model_existing:
            base_r5 = model_existing['baseline']['recall@5']
            full_r5 = model_existing['isotropy']['recall@5']
            frozen_r5 = r['frozen']['recall@5']
            
            avg_full_improvement += (full_r5 - base_r5) * 100
            avg_frozen_improvement += (frozen_r5 - base_r5) * 100
            count += 1
    
    if count > 0:
        avg_full_improvement /= count
        avg_frozen_improvement /= count
        
        print(f"Average Recall@5 improvement:")
        print(f"  Full fine-tune: {avg_full_improvement:+.1f}%")
        print(f"  Frozen base:    {avg_frozen_improvement:+.1f}%")
        print()
        
        if avg_frozen_improvement > avg_full_improvement + 0.5:
            print("✅ FROZEN BASE WINS! Better performance with 3x faster training")
        elif avg_full_improvement > avg_frozen_improvement + 0.5:
            print("✅ FULL FINE-TUNE WINS! Worth the extra training time")
        else:
            print("✅ TIE! Both approaches perform similarly")
    
    print("="*90)
    print(f"\nResults saved to: {output_dir / 'frozen_results.json'}")

if __name__ == '__main__':
    main()

