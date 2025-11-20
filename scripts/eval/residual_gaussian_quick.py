#!/usr/bin/env python3
"""
Quick evaluation script for ResPred model on 2 BEIR datasets.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.respred_model import ResidualGaussianEncoder


class ResPredRetriever:
    """BEIR-compatible retriever for ResPred."""
    
    def __init__(self, model, batch_size=128):
        self.model = model
        self.batch_size = batch_size
        
    def encode_queries(self, queries, batch_size=None, **kwargs):
        """Encode queries (does NOT use predictor)."""
        batch_size = batch_size or self.batch_size
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        embeddings = self.model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress=True,
            convert_to_numpy=True
        )
        return {query_ids[i]: embeddings[i] for i in range(len(query_ids))}
    
    def encode_corpus(self, corpus, batch_size=None, **kwargs):
        """Encode corpus documents."""
        batch_size = batch_size or self.batch_size
        corpus_ids = list(corpus.keys())
        doc_texts = [f"{corpus[did]['title']} {corpus[did]['text']}".strip() 
                     for did in corpus_ids]
        embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress=True,
            convert_to_numpy=True
        )
        return {corpus_ids[i]: embeddings[i] for i in range(len(corpus_ids))}


def evaluate_dataset(model, dataset_name, data_dir="data/beir", batch_size=64):
    """Evaluate on a single BEIR dataset using direct approach."""
    import numpy as np
    from tqdm import tqdm
    
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
    
    # Encode corpus
    print("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [f"{corpus[did]['title']} {corpus[did]['text']}".strip() for did in corpus_ids]
    
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Corpus"):
        batch = corpus_texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True)
        corpus_embeddings.append(embs)
    corpus_embeddings = np.vstack(corpus_embeddings)
    
    # Encode queries
    print("Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_embeddings = []
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Queries"):
        batch = query_texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True)
        query_embeddings.append(embs)
    query_embeddings = np.vstack(query_embeddings)
    
    # Compute similarities (negative Euclidean distance for unnormalized embeddings)
    print("Computing similarities...")
    results = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
        query_emb = query_embeddings[i]
        distances = np.linalg.norm(corpus_embeddings - query_emb, axis=1)
        scores = -distances  # Negative distance = similarity
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
    print("ResPred BEIR Evaluation (Quick)")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {args.datasets}")
    print(f"Base model: {args.base_model}")
    print(f"Output dim: {args.output_dim}")
    print("="*80)
    
    # Load model
    print("\nLoading ResPred model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure we don't track gradients
    torch.set_grad_enabled(False)
    
    model = ResidualGaussianEncoder(
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=True,
        normalize_embeddings=False,
        use_predictor=True
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

