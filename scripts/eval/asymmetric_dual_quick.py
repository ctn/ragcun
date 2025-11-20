#!/usr/bin/env python3
"""
Quick evaluation script for Asymmetric Projection model on 2 BEIR datasets.
Handles query vs document encoding properly.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.asymmetric_model import AsymmetricDualEncoder


def evaluate_dataset(model, dataset_name, data_dir="data/beir", batch_size=64):
    """Evaluate on a single BEIR dataset using asymmetric encoding."""
    print(f"\n{'='*80}")
    print(f"Evaluating on: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, data_dir)
    
    # Load data
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    print(f"Corpus: {len(corpus):,} documents")
    print(f"Queries: {len(queries):,}")
    
    # Encode corpus (as DOCUMENTS)
    print("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [f"{corpus[did]['title']} {corpus[did]['text']}".strip() for did in corpus_ids]
    
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Corpus"):
        batch = corpus_texts[i:i+batch_size]
        embs = model.encode_docs(batch, convert_to_numpy=True)  # Use doc projection
        corpus_embeddings.append(embs)
    corpus_embeddings = np.vstack(corpus_embeddings)
    
    # Encode queries (as QUERIES)
    print("Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_embeddings = []
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Queries"):
        batch = query_texts[i:i+batch_size]
        embs = model.encode_queries(batch, convert_to_numpy=True)  # Use query projection
        query_embeddings.append(embs)
    query_embeddings = np.vstack(query_embeddings)
    
    # Compute similarities (COSINE SIMILARITY - matching training objective)
    print("Computing similarities...")
    
    # Normalize embeddings for cosine similarity
    corpus_embeddings_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    query_embeddings_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    results = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
        query_emb = query_embeddings_norm[i]
        # Cosine similarity (matching training loss)
        scores = corpus_embeddings_norm @ query_emb
        results[qid] = {corpus_ids[j]: float(scores[j]) for j in range(len(corpus_ids))}
    
    # Evaluate
    print("Computing metrics...")
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10, 100])
    
    return {
        'NDCG@10': ndcg['NDCG@10'],
        'NDCG@100': ndcg['NDCG@100'],
        'MAP@10': _map['MAP@10'],
        'Recall@100': recall['Recall@100']
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--datasets', nargs='+', default=['scifact', 'nfcorpus'])
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--base_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--output_dim', type=int, default=768)
    
    args = parser.parse_args()
    
    print("="*80)
    print("Asymmetric Projection BEIR Evaluation (Quick)")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {args.datasets}")
    print(f"Base model: {args.base_model}")
    print(f"Output dim: {args.output_dim}")
    print("="*80)
    
    # Load model
    print("\nLoading Asymmetric Projection model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure we don't track gradients
    torch.set_grad_enabled(False)
    
    model = AsymmetricDualEncoder(
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=True,
        normalize_embeddings=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Training loss: {checkpoint.get('train_loss', 'N/A')}")
    
    # Evaluate on each dataset
    results = {}
    for dataset in args.datasets:
        try:
            metrics = evaluate_dataset(model, dataset)
            results[dataset] = metrics
            
            print(f"\n{dataset} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"\n❌ Error evaluating {dataset}: {e}")
            results[dataset] = {'error': str(e)}
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {args.output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for dataset, metrics in results.items():
        if 'error' not in metrics:
            print(f"{dataset:15} NDCG@10: {metrics['NDCG@10']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()

