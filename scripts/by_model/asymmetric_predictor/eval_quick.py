#!/usr/bin/env python3
"""
Quick evaluation script for Asymmetric + Predictor model.

Evaluates on SciFact and NFCorpus for quick feedback.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import beir.util as util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from ragcun.asymmetric_predictor_model import AsymmetricWithPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AsymmetricPredictorRetriever:
    """Wrapper for BEIR evaluation."""
    
    def __init__(self, model, batch_size=64):
        self.model = model
        self.batch_size = batch_size
        self.model.eval()
    
    def encode_queries(self, queries, **kwargs):
        """Encode queries using query projection."""
        query_list = [queries[qid] for qid in sorted(queries.keys())]
        
        embeddings = []
        for i in range(0, len(query_list), self.batch_size):
            batch = query_list[i:i+self.batch_size]
            with torch.no_grad():
                batch_emb = self.model.encode_queries(batch, convert_to_numpy=True)
            embeddings.append(batch_emb)
        
        import numpy as np
        all_embeddings = np.vstack(embeddings)
        
        # Return dict: qid → embedding
        result = {}
        for idx, qid in enumerate(sorted(queries.keys())):
            result[qid] = all_embeddings[idx]
        
        return result
    
    def encode_corpus(self, corpus, **kwargs):
        """Encode documents using doc projection."""
        # Combine title and text
        doc_list = []
        doc_ids = []
        for doc_id in sorted(corpus.keys()):
            doc = corpus[doc_id]
            text = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
            doc_list.append(text)
            doc_ids.append(doc_id)
        
        embeddings = []
        logger.info(f"Encoding {len(doc_list)} documents...")
        for i in range(0, len(doc_list), self.batch_size):
            batch = doc_list[i:i+self.batch_size]
            with torch.no_grad():
                batch_emb = self.model.encode_docs(batch, convert_to_numpy=True)
            embeddings.append(batch_emb)
            
            if (i // self.batch_size + 1) % 100 == 0:
                logger.info(f"  Encoded {i+len(batch)}/{len(doc_list)} documents")
        
        import numpy as np
        all_embeddings = np.vstack(embeddings)
        
        # Return dict: doc_id → embedding
        result = {}
        for idx, doc_id in enumerate(doc_ids):
            result[doc_id] = all_embeddings[idx]
        
        return result


def evaluate_dataset(model, dataset_name, data_dir="data/beir", batch_size=64):
    """Evaluate on a single BEIR dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating on {dataset_name}")
    logger.info(f"{'='*80}")
    
    # Load dataset
    dataset_path = f"{data_dir}/{dataset_name}"
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
    
    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
    
    # Create retriever
    retriever = AsymmetricPredictorRetriever(model, batch_size=batch_size)
    
    # Evaluate
    evaluator = EvaluateRetrieval(retriever, score_function="dot")
    results = evaluator.retrieve(corpus, queries)
    
    # Compute metrics
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10, 100, 1000])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 3, 5, 10, 100, 1000], metric="mrr")
    
    # Combine all metrics
    metrics = {}
    for metric_dict, prefix in [(mrr, "MRR"), (ndcg, "NDCG"), (_map, "MAP"), (recall, "Recall"), (precision, "P")]:
        for k, v in metric_dict.items():
            if prefix == "MRR":
                metrics[prefix] = v
            else:
                metrics[f"{prefix}@{k.split('@')[1]}"] = v
    
    # Print key metrics
    logger.info(f"\nResults for {dataset_name}:")
    logger.info(f"  NDCG@10: {metrics['NDCG@10']:.4f}")
    logger.info(f"  MRR: {metrics['MRR']:.4f}")
    logger.info(f"  Recall@10: {metrics['Recall@10']:.4f}")
    
    return {dataset_name: metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--base_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--output_dim', type=int, default=768)
    parser.add_argument('--datasets', nargs='+', default=['scifact', 'nfcorpus'], 
                       help='Datasets to evaluate on')
    parser.add_argument('--data_dir', default='data/beir')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_file', default=None)
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Asymmetric + Predictor Model Evaluation")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info("")
    
    # Load model
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = AsymmetricWithPredictor.from_pretrained(
        args.checkpoint,
        base_model=args.base_model,
        output_dim=args.output_dim,
        freeze_base=True,
        normalize_embeddings=False
    )
    model = model.to(device)
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    
    # Evaluate on each dataset
    all_results = {}
    for dataset in args.datasets:
        try:
            results = evaluate_dataset(model, dataset, args.data_dir, args.batch_size)
            all_results.update(results)
        except Exception as e:
            logger.error(f"Error evaluating {dataset}: {e}")
            continue
    
    # Compute average NDCG@10
    ndcg10_scores = [metrics['NDCG@10'] for metrics in all_results.values()]
    avg_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores) if ndcg10_scores else 0
    
    logger.info(f"\n{'='*80}")
    logger.info("Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Average NDCG@10: {avg_ndcg10:.4f}")
    logger.info("")
    for dataset, metrics in all_results.items():
        logger.info(f"{dataset}: NDCG@10 = {metrics['NDCG@10']:.4f}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✅ Results saved to {args.output_file}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Evaluation complete!")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()

