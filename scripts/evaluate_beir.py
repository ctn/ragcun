#!/usr/bin/env python3
"""
Evaluate model on BEIR (Benchmarking IR) suite.

BEIR is the standard evaluation benchmark for retrieval models,
testing on 15+ diverse datasets without dataset-specific tuning.

Usage:
    python scripts/evaluate_beir.py \\
        --model_path checkpoints/smart_hybrid/best_model.pt \\
        --datasets scifact nfcorpus \\
        --output_file results/beir_scifact_nfcorpus.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import numpy as np

import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import IsotropicGaussianEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# BEIR datasets (ordered by difficulty and size)
BEIR_DATASETS = {
    # Small, quick datasets (< 10 min on single GPU)
    'scifact': {'name': 'SciFact', 'queries': 300, 'corpus': 5183},
    'nfcorpus': {'name': 'NFCorpus', 'queries': 323, 'corpus': 3633},
    'arguana': {'name': 'ArguAna', 'queries': 1406, 'corpus': 8674},
    'fiqa': {'name': 'FiQA', 'queries': 648, 'corpus': 57638},
    'trec-covid': {'name': 'TREC-COVID', 'queries': 50, 'corpus': 171332},
    
    # Medium datasets (10-30 min)
    'scidocs': {'name': 'SCIDOCS', 'queries': 1000, 'corpus': 25657},
    'quora': {'name': 'Quora', 'queries': 10000, 'corpus': 522931},
    'dbpedia-entity': {'name': 'DBPedia', 'queries': 400, 'corpus': 4635922},
    
    # Large datasets (30-60 min)
    'fever': {'name': 'FEVER', 'queries': 6666, 'corpus': 5416568},
    'climate-fever': {'name': 'Climate-FEVER', 'queries': 1535, 'corpus': 5416593},
    'hotpotqa': {'name': 'HotpotQA', 'queries': 7405, 'corpus': 5233329},
    'nq': {'name': 'Natural Questions', 'queries': 3452, 'corpus': 2681468},
    
    # Very large (1+ hours)
    'msmarco': {'name': 'MS MARCO', 'queries': 6980, 'corpus': 8841823},
}

# Recommended subsets for different purposes
QUICK_EVAL = ['scifact', 'nfcorpus']  # 5-10 min
STANDARD_EVAL = ['scifact', 'nfcorpus', 'arguana', 'fiqa', 'trec-covid']  # 30 min
FULL_EVAL = list(BEIR_DATASETS.keys())  # Several hours


def download_beir_dataset(dataset_name: str, data_dir: str = 'data/beir'):
    """
    Download BEIR dataset.
    
    Args:
        dataset_name: Name of BEIR dataset
        data_dir: Directory to store data
        
    Returns:
        Tuple of (corpus, queries, qrels)
    """
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        logger.error("BEIR not installed. Install: pip install beir")
        sys.exit(1)
    
    data_path = Path(data_dir)
    dataset_path = data_path / dataset_name
    
    if not dataset_path.exists():
        logger.info(f"Downloading {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        beir_util.download_and_unzip(url, str(data_path))
    else:
        logger.info(f"Using cached {dataset_name} from {dataset_path}")
    
    # Load data
    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split="test")
    
    return corpus, queries, qrels


def compute_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 3, 5, 10, 100, 1000]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        results: Query ID -> {Doc ID -> score}
        qrels: Query ID -> {Doc ID -> relevance}
        k_values: K values for Recall@K and Precision@K
        
    Returns:
        Dictionary of metrics
    """
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = []
        _map[f"MAP@{k}"] = []
        recall[f"Recall@{k}"] = []
        precision[f"P@{k}"] = []
    
    # Also compute MRR
    mrr_list = []
    
    for query_id in qrels:
        if query_id not in results:
            continue
        
        # Get relevant docs
        query_relevant_docs = {doc_id: rel for doc_id, rel in qrels[query_id].items() if rel > 0}
        
        if not query_relevant_docs:
            continue
        
        # Sort results by score
        sorted_results = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)
        
        # Compute MRR
        for rank, (doc_id, _) in enumerate(sorted_results, 1):
            if doc_id in query_relevant_docs:
                mrr_list.append(1.0 / rank)
                break
        else:
            mrr_list.append(0.0)
        
        # Compute metrics at different k
        for k in k_values:
            # Get top k
            top_k = sorted_results[:k]
            top_k_ids = [doc_id for doc_id, _ in top_k]
            
            # Recall@k
            relevant_in_k = len(set(top_k_ids) & set(query_relevant_docs.keys()))
            recall[f"Recall@{k}"].append(relevant_in_k / len(query_relevant_docs))
            
            # Precision@k
            precision[f"P@{k}"].append(relevant_in_k / k)
            
            # NDCG@k
            dcg = 0.0
            idcg = 0.0
            
            for i, (doc_id, _) in enumerate(top_k, 1):
                rel = query_relevant_docs.get(doc_id, 0)
                dcg += rel / np.log2(i + 1)
            
            # Ideal DCG
            ideal_rels = sorted(query_relevant_docs.values(), reverse=True)[:k]
            for i, rel in enumerate(ideal_rels, 1):
                idcg += rel / np.log2(i + 1)
            
            ndcg[f"NDCG@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
            
            # MAP@k
            avg_precision = 0.0
            num_relevant = 0
            for i, (doc_id, _) in enumerate(top_k, 1):
                if doc_id in query_relevant_docs:
                    num_relevant += 1
                    precision_at_i = num_relevant / i
                    avg_precision += precision_at_i
            
            if num_relevant > 0:
                avg_precision /= min(len(query_relevant_docs), k)
            
            _map[f"MAP@{k}"].append(avg_precision)
    
    # Average metrics
    metrics = {'MRR': np.mean(mrr_list) if mrr_list else 0.0}
    
    for k in k_values:
        metrics[f"NDCG@{k}"] = np.mean(ndcg[f"NDCG@{k}"]) if ndcg[f"NDCG@{k}"] else 0.0
        metrics[f"MAP@{k}"] = np.mean(_map[f"MAP@{k}"]) if _map[f"MAP@{k}"] else 0.0
        metrics[f"Recall@{k}"] = np.mean(recall[f"Recall@{k}"]) if recall[f"Recall@{k}"] else 0.0
        metrics[f"P@{k}"] = np.mean(precision[f"P@{k}"]) if precision[f"P@{k}"] else 0.0
    
    return metrics


def evaluate_beir_dataset(
    model: IsotropicGaussianEncoder,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 64,
    data_dir: str = 'data/beir'
) -> Dict[str, float]:
    """
    Evaluate model on a single BEIR dataset.
    
    Args:
        model: Trained model
        dataset_name: BEIR dataset name
        device: Device to use
        batch_size: Batch size for encoding
        data_dir: Directory with BEIR data
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating on {BEIR_DATASETS[dataset_name]['name']}")
    logger.info(f"{'='*60}")
    
    # Download dataset
    corpus, queries, qrels = download_beir_dataset(dataset_name, data_dir)
    
    logger.info(f"Corpus: {len(corpus):,} documents")
    logger.info(f"Queries: {len(queries):,}")
    logger.info(f"Qrels: {len(qrels):,}")
    
    # Encode corpus
    logger.info("Encoding corpus...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id]['title'] + ' ' + corpus[doc_id]['text'] for doc_id in corpus_ids]
    
    model.eval()
    corpus_embeddings = []
    
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding corpus"):
        batch_texts = corpus_texts[i:i+batch_size]
        with torch.no_grad():
            batch_embs = model.encode(batch_texts, convert_to_numpy=True)
        corpus_embeddings.append(batch_embs)
    
    corpus_embeddings = np.vstack(corpus_embeddings)
    logger.info(f"Corpus embeddings shape: {corpus_embeddings.shape}")
    
    # Encode queries
    logger.info("Encoding queries...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_embeddings = []
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Encoding queries"):
        batch_texts = query_texts[i:i+batch_size]
        with torch.no_grad():
            batch_embs = model.encode(batch_texts, convert_to_numpy=True)
        query_embeddings.append(batch_embs)
    
    query_embeddings = np.vstack(query_embeddings)
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Compute similarities (negative Euclidean distance)
    logger.info("Computing similarities...")
    results = {}
    
    for i, qid in enumerate(tqdm(query_ids, desc="Ranking")):
        query_emb = query_embeddings[i]
        
        # Compute Euclidean distances
        distances = np.linalg.norm(corpus_embeddings - query_emb, axis=1)
        
        # Convert to similarity scores (negative distance)
        scores = -distances
        
        # Store results
        results[qid] = {corpus_ids[j]: float(scores[j]) for j in range(len(corpus_ids))}
    
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
        description="Evaluate model on BEIR benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Quick evaluation (5-10 min)
  python scripts/evaluate_beir.py \\
    --model_path checkpoints/smart_hybrid/best_model.pt \\
    --datasets scifact nfcorpus

  # Standard evaluation (30 min)
  python scripts/evaluate_beir.py \\
    --model_path checkpoints/smart_hybrid/best_model.pt \\
    --datasets scifact nfcorpus arguana fiqa trec-covid

  # Specific output file
  python scripts/evaluate_beir.py \\
    --model_path checkpoints/smart_hybrid/best_model.pt \\
    --datasets scifact \\
    --output_file results/beir_scifact.json

Available datasets: {', '.join(BEIR_DATASETS.keys())}
"""
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=QUICK_EVAL,
        help=f'BEIR datasets to evaluate (default: {" ".join(QUICK_EVAL)})'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSON file (default: auto-generated in results/)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for encoding (default: 64)'
    )
    
    parser.add_argument(
        '--output_dim',
        type=int,
        default=512,
        help='Model output dimension (default: 512)'
    )
    
    parser.add_argument(
        '--base_model',
        type=str,
        default=None,
        help='Base model name (e.g., sentence-transformers/all-mpnet-base-v2)'
    )
    
    parser.add_argument(
        '--freeze_base',
        action='store_true',
        help='Load with frozen base encoder (for frozen models)'
    )

    parser.add_argument(
        '--no_normalize_embeddings',
        action='store_true',
        help='Disable normalization of base model embeddings (for models trained without normalization)'
    )

    parser.add_argument(
        '--use_predictor',
        action='store_true',
        help='Load model with predictor network (for JEPA-like models)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/beir',
        help='Directory to store BEIR datasets (default: data/beir)'
    )

    args = parser.parse_args()
    
    # Validate datasets
    invalid = [d for d in args.datasets if d not in BEIR_DATASETS]
    if invalid:
        logger.error(f"Invalid datasets: {invalid}")
        logger.error(f"Available: {list(BEIR_DATASETS.keys())}")
        sys.exit(1)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    load_kwargs = {'output_dim': args.output_dim}
    if args.base_model:
        load_kwargs['base_model'] = args.base_model
    if args.freeze_base:
        load_kwargs['freeze_base'] = True
    if args.no_normalize_embeddings:
        load_kwargs['normalize_embeddings'] = False
    if args.use_predictor:
        load_kwargs['use_predictor'] = True

    model = IsotropicGaussianEncoder.from_pretrained(
        args.model_path,
        **load_kwargs
    )
    model = model.to(device)
    model.eval()
    
    # Evaluate on each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        try:
            metrics = evaluate_beir_dataset(
                model,
                dataset_name,
                device,
                args.batch_size,
                args.data_dir
            )
            all_results[dataset_name] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            continue
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Auto-generate filename
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        datasets_str = '_'.join(args.datasets)
        output_path = output_dir / f'beir_{datasets_str}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_path}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for dataset_name, metrics in all_results.items():
        logger.info(f"\n{BEIR_DATASETS[dataset_name]['name']}:")
        logger.info(f"  NDCG@10:  {metrics['NDCG@10']:.4f}")
        logger.info(f"  Recall@10: {metrics['Recall@10']:.4f}")
    
    # Average
    if len(all_results) > 1:
        avg_ndcg = np.mean([m['NDCG@10'] for m in all_results.values()])
        avg_recall = np.mean([m['Recall@10'] for m in all_results.values()])
        
        logger.info(f"\nAverage ({len(all_results)} datasets):")
        logger.info(f"  NDCG@10:  {avg_ndcg:.4f}")
        logger.info(f"  Recall@10: {avg_recall:.4f}")


if __name__ == '__main__':
    main()

