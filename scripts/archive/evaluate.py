#!/usr/bin/env python3
"""
Evaluation script for GaussianEmbeddingGemma retrieval performance.

This script evaluates the trained model on various retrieval metrics including:
- Recall@K (1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Mean Average Precision (MAP)

Usage:
    python scripts/evaluate.py --model_path checkpoints/best_model.pt --test_data data/processed/test.json
    python scripts/evaluate.py --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import GaussianEmbeddingGemma

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Evaluator for retrieval tasks.

    Computes various retrieval metrics including Recall@K, MRR, NDCG, and MAP.
    """

    def __init__(self, model: GaussianEmbeddingGemma, device: torch.device):
        """
        Initialize evaluator.

        Args:
            model: Trained GaussianEmbeddingGemma model
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device
        self.model.eval()

    def encode_corpus(
        self,
        documents: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode a corpus of documents.

        Args:
            documents: List of document strings
            batch_size: Batch size for encoding

        Returns:
            Document embeddings as numpy array
        """
        logger.info(f"Encoding {len(documents)} documents...")

        with torch.no_grad():
            embeddings = self.model.encode(
                documents,
                batch_size=batch_size,
                show_progress=True,
                convert_to_numpy=True
            )

        return embeddings

    def retrieve(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query string
            corpus_embeddings: Corpus embeddings (num_docs, dim)
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (indices, distances) for top-k documents
        """
        with torch.no_grad():
            query_emb = self.model.encode([query], convert_to_numpy=True)

        # Compute Euclidean distances
        distances = np.linalg.norm(
            corpus_embeddings - query_emb,
            axis=1
        )

        # Get top-k indices (smallest distances)
        top_k = min(top_k, len(distances))
        indices = np.argsort(distances)[:top_k]
        top_distances = distances[indices]

        return indices, top_distances

    @staticmethod
    def compute_recall_at_k(
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k_values: List[int] = [1, 5, 10, 20, 50, 100]
    ) -> Dict[int, float]:
        """
        Compute Recall@K for different k values.

        Args:
            retrieved_indices: Retrieved document indices
            relevant_indices: Ground truth relevant document indices
            k_values: List of k values to compute

        Returns:
            Dictionary mapping k -> Recall@K
        """
        relevant_set = set(relevant_indices)
        recall = {}

        for k in k_values:
            # Use min to cap k, but keep original k as the dictionary key
            effective_k = min(k, len(retrieved_indices))

            retrieved_at_k = set(retrieved_indices[:effective_k])
            num_relevant_retrieved = len(retrieved_at_k & relevant_set)

            recall[k] = num_relevant_retrieved / len(relevant_set) if relevant_set else 0.0

        return recall

    @staticmethod
    def compute_mrr(
        retrieved_indices: np.ndarray,
        relevant_indices: List[int]
    ) -> float:
        """
        Compute Mean Reciprocal Rank.

        Args:
            retrieved_indices: Retrieved document indices
            relevant_indices: Ground truth relevant document indices

        Returns:
            MRR score
        """
        relevant_set = set(relevant_indices)

        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def compute_ndcg(
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@K.

        Args:
            retrieved_indices: Retrieved document indices
            relevant_indices: Ground truth relevant document indices
            k: Cutoff for NDCG

        Returns:
            NDCG@K score
        """
        k = min(k, len(retrieved_indices))
        relevant_set = set(relevant_indices)

        # DCG: sum of (relevance / log2(rank + 1))
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, idx in enumerate(retrieved_indices[:k])
            if idx in relevant_set
        )

        # IDCG: DCG if all relevant docs were retrieved first
        idcg = sum(
            1.0 / np.log2(rank + 2)
            for rank in range(min(k, len(relevant_indices)))
        )

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_map(
        retrieved_indices: np.ndarray,
        relevant_indices: List[int],
        k: int = 100
    ) -> float:
        """
        Compute Mean Average Precision@K.

        Args:
            retrieved_indices: Retrieved document indices
            relevant_indices: Ground truth relevant document indices
            k: Cutoff for MAP

        Returns:
            MAP@K score
        """
        k = min(k, len(retrieved_indices))
        relevant_set = set(relevant_indices)

        if not relevant_set:
            return 0.0

        precisions = []
        num_relevant = 0

        for rank, idx in enumerate(retrieved_indices[:k], 1):
            if idx in relevant_set:
                num_relevant += 1
                precision = num_relevant / rank
                precisions.append(precision)

        return sum(precisions) / len(relevant_set) if precisions else 0.0

    def evaluate(
        self,
        queries: List[str],
        corpus: List[str],
        relevance: List[List[int]],
        batch_size: int = 32,
        top_k: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate the model on a retrieval task.

        Args:
            queries: List of query strings
            corpus: List of document strings
            relevance: List of relevant document indices for each query
            batch_size: Batch size for encoding
            top_k: Number of documents to retrieve

        Returns:
            Dictionary of evaluation metrics
        """
        assert len(queries) == len(relevance), "Number of queries must match relevance lists"

        # Encode corpus once
        corpus_embeddings = self.encode_corpus(corpus, batch_size)

        # Evaluate each query
        all_recall = {k: [] for k in [1, 5, 10, 20, 50, 100]}
        all_mrr = []
        all_ndcg = []
        all_map = []

        logger.info(f"Evaluating {len(queries)} queries...")

        for query, relevant_indices in tqdm(zip(queries, relevance), total=len(queries)):
            # Retrieve
            retrieved_indices, _ = self.retrieve(query, corpus_embeddings, top_k)

            # Compute metrics
            recall = self.compute_recall_at_k(retrieved_indices, relevant_indices)
            mrr = self.compute_mrr(retrieved_indices, relevant_indices)
            ndcg = self.compute_ndcg(retrieved_indices, relevant_indices, k=10)
            map_score = self.compute_map(retrieved_indices, relevant_indices, k=100)

            # Accumulate
            for k in all_recall:
                all_recall[k].append(recall.get(k, 0.0))
            all_mrr.append(mrr)
            all_ndcg.append(ndcg)
            all_map.append(map_score)

        # Average metrics
        metrics = {
            'MRR': np.mean(all_mrr),
            'NDCG@10': np.mean(all_ndcg),
            'MAP@100': np.mean(all_map),
        }

        for k in all_recall:
            metrics[f'Recall@{k}'] = np.mean(all_recall[k])

        return metrics


def load_test_data(data_path: str) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    Load test data from JSON file.

    Expected format:
    {
        "corpus": ["doc1", "doc2", ...],
        "queries": ["query1", "query2", ...],
        "relevance": [[0, 5, 10], [1, 3], ...]  # Indices of relevant docs for each query
    }

    Args:
        data_path: Path to test data JSON file

    Returns:
        Tuple of (queries, corpus, relevance)
    """
    logger.info(f"Loading test data from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = data['queries']
    corpus = data['corpus']
    relevance = data['relevance']

    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} documents")

    return queries, corpus, relevance


def main():
    parser = argparse.ArgumentParser(description="Evaluate GaussianEmbeddingGemma")

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--output_dim', type=int, default=512,
                        help='Embedding dimension (must match trained model)')

    # Data arguments
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data JSON file')

    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of documents to retrieve per query')

    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results (JSON)')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = GaussianEmbeddingGemma.from_pretrained(
        args.model_path,
        output_dim=args.output_dim
    )
    model = model.to(device)

    # Load test data
    queries, corpus, relevance = load_test_data(args.test_data)

    # Create evaluator
    evaluator = RetrievalEvaluator(model, device)

    # Evaluate
    logger.info("\nStarting evaluation...")
    metrics = evaluator.evaluate(
        queries=queries,
        corpus=corpus,
        relevance=relevance,
        batch_size=args.batch_size,
        top_k=args.top_k
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)

    for metric_name, value in sorted(metrics.items()):
        logger.info(f"{metric_name:20s}: {value:.4f}")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'model_path': args.model_path,
            'test_data': args.test_data,
            'metrics': metrics,
            'config': vars(args)
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved to {output_path}")

    logger.info("\n" + "="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
