#!/usr/bin/env python3
"""
Evaluate vanilla (untrained) sentence-transformers models on BEIR.

This gives us the TRUE baseline for comparison - the off-the-shelf 
pre-trained model with NO additional training.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
from sentence_transformers import SentenceTransformer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_beir import (
    download_beir_dataset, 
    compute_metrics,
    BEIR_DATASETS,
    STANDARD_EVAL
)

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_vanilla_model(
    model_name: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 64
):
    """Evaluate vanilla sentence-transformers model on BEIR."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name} on {BEIR_DATASETS[dataset_name]['name']}")
    logger.info(f"{'='*60}")
    
    # Load vanilla model (uses cosine similarity by default)
    logger.info(f"Loading vanilla model: {model_name}")
    model = SentenceTransformer(model_name, device=str(device))
    
    # Download dataset
    corpus, queries, qrels = download_beir_dataset(dataset_name)
    
    logger.info(f"Corpus: {len(corpus):,} documents")
    logger.info(f"Queries: {len(queries):,}")
    
    # Encode corpus
    logger.info("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        corpus[doc_id]['title'] + ' ' + corpus[doc_id]['text'] 
        for doc_id in corpus_ids
    ]
    
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Encode queries
    logger.info("Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Compute cosine similarities (default for sentence-transformers)
    logger.info("Computing similarities (cosine)...")
    results = {}
    
    # Normalize embeddings for cosine similarity
    corpus_embeddings = corpus_embeddings / np.linalg.norm(
        corpus_embeddings, axis=1, keepdims=True
    )
    query_embeddings = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    
    for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
        query_emb = query_embeddings[i]
        
        # Cosine similarity = dot product of normalized vectors
        scores = corpus_embeddings @ query_emb
        
        # Store results
        results[qid] = {
            corpus_ids[j]: float(scores[j]) 
            for j in range(len(corpus_ids))
        }
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(results, qrels)
    
    # Log results
    logger.info(f"\nResults for {BEIR_DATASETS[dataset_name]['name']}:")
    logger.info(f"  MRR:      {metrics['MRR']:.4f}")
    logger.info(f"  NDCG@10:  {metrics['NDCG@10']:.4f}")
    logger.info(f"  Recall@10: {metrics['Recall@10']:.4f}")
    logger.info(f"  MAP@10:   {metrics['MAP@10']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vanilla sentence-transformers on BEIR"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Sentence-transformers model name'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=STANDARD_EVAL,
        help='BEIR datasets to evaluate on'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results/beir_standard/vanilla_baseline.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for encoding'
    )
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Evaluate on all datasets
    all_results = {}
    
    for dataset in args.datasets:
        metrics = evaluate_vanilla_model(
            args.model_name,
            dataset,
            device,
            args.batch_size
        )
        all_results[dataset] = metrics
    
    # Compute average
    avg_ndcg = np.mean([
        all_results[ds]['NDCG@10'] 
        for ds in args.datasets
    ])
    avg_recall = np.mean([
        all_results[ds]['Recall@10'] 
        for ds in args.datasets
    ])
    
    all_results['Average'] = {
        'NDCG_at_10': avg_ndcg,
        'Recall_at_10': avg_recall
    }
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    for ds in args.datasets:
        logger.info(f"\n{BEIR_DATASETS[ds]['name']}:")
        logger.info(f"  NDCG@10:  {all_results[ds]['NDCG@10']:.4f}")
        logger.info(f"  Recall@10: {all_results[ds]['Recall@10']:.4f}")
    
    logger.info(f"\nAverage ({len(args.datasets)} datasets):")
    logger.info(f"  NDCG@10:  {avg_ndcg:.4f}")
    logger.info(f"  Recall@10: {avg_recall:.4f}")


if __name__ == '__main__':
    main()

