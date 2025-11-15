"""
Unit tests for evaluation metrics.

Tests Recall@K, MRR, NDCG, MAP, and other retrieval metrics.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add scripts to path for evaluate module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluate import RetrievalEvaluator
from ragcun.model import GaussianEmbeddingGemma


class TestRecallAtK:
    """Test Recall@K metric."""

    def test_perfect_recall(self):
        """Test Recall@K when all relevant docs are retrieved."""
        retrieved = np.array([0, 1, 2, 3, 4])
        relevant = [0, 1]  # First two docs are relevant

        recall = RetrievalEvaluator.compute_recall_at_k(
            retrieved,
            relevant,
            k_values=[1, 5]
        )

        assert recall[1] == 0.5  # Got 1 out of 2
        assert recall[5] == 1.0  # Got both

    def test_zero_recall(self):
        """Test Recall@K when no relevant docs are retrieved."""
        retrieved = np.array([2, 3, 4, 5])
        relevant = [0, 1]  # Not in retrieved

        recall = RetrievalEvaluator.compute_recall_at_k(
            retrieved,
            relevant,
            k_values=[1, 4]
        )

        assert recall[1] == 0.0
        assert recall[4] == 0.0

    def test_partial_recall(self):
        """Test partial recall."""
        retrieved = np.array([0, 5, 10, 15, 20])
        relevant = [0, 10, 25]  # 2 out of 3 in retrieved

        recall = RetrievalEvaluator.compute_recall_at_k(
            retrieved,
            relevant,
            k_values=[5]
        )

        assert recall[5] == 2.0 / 3.0

    def test_k_larger_than_retrieved(self):
        """Test when k > number of retrieved docs."""
        retrieved = np.array([0, 1, 2])
        relevant = [0, 1]

        recall = RetrievalEvaluator.compute_recall_at_k(
            retrieved,
            relevant,
            k_values=[10]
        )

        # When k > retrieved, implementation caps k to len(retrieved) but keeps original k as key
        # This makes sense: "What was recall@10?" even if only 3 docs retrieved
        assert recall[10] == 1.0

    def test_empty_relevant(self):
        """Test with no relevant documents."""
        retrieved = np.array([0, 1, 2])
        relevant = []

        recall = RetrievalEvaluator.compute_recall_at_k(
            retrieved,
            relevant,
            k_values=[3]
        )

        assert recall[3] == 0.0

    def test_multiple_k_values(self):
        """Test with multiple k values."""
        retrieved = np.array([0, 1, 2, 3, 4, 5])
        relevant = [1, 3, 7]  # 2 in top-6

        recall = RetrievalEvaluator.compute_recall_at_k(
            retrieved,
            relevant,
            k_values=[1, 3, 5, 6]
        )

        assert recall[1] == 0.0  # 0/3
        assert recall[3] == 1.0 / 3.0  # 1/3 (got idx 1)
        assert recall[5] == 2.0 / 3.0  # 2/3 (got idx 1, 3)
        assert recall[6] == 2.0 / 3.0  # Still 2/3


class TestMRR:
    """Test Mean Reciprocal Rank metric."""

    def test_first_position(self):
        """Test MRR when relevant doc is at position 1."""
        retrieved = np.array([5, 1, 2, 3])
        relevant = [5]

        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        assert mrr == 1.0

    def test_second_position(self):
        """Test MRR when relevant doc is at position 2."""
        retrieved = np.array([0, 5, 1, 2])
        relevant = [5]

        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        assert mrr == 0.5

    def test_third_position(self):
        """Test MRR at position 3."""
        retrieved = np.array([0, 1, 5, 2])
        relevant = [5]

        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        assert abs(mrr - 1.0/3.0) < 1e-6

    def test_no_relevant_found(self):
        """Test MRR when no relevant docs are retrieved."""
        retrieved = np.array([0, 1, 2, 3])
        relevant = [10]

        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        assert mrr == 0.0

    def test_multiple_relevant_first_counts(self):
        """Test that only first relevant doc counts."""
        retrieved = np.array([0, 5, 10, 15])
        relevant = [5, 10]  # Both retrieved, but only first matters

        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        assert mrr == 0.5  # Position 2

    def test_empty_relevant(self):
        """Test with no relevant documents."""
        retrieved = np.array([0, 1, 2])
        relevant = []

        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        assert mrr == 0.0


class TestNDCG:
    """Test Normalized Discounted Cumulative Gain metric."""

    def test_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        retrieved = np.array([0, 1, 2, 3, 4])
        relevant = [0, 1]  # All relevant docs at top

        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=5)

        assert ndcg == 1.0

    def test_reversed_ranking(self):
        """Test NDCG with worst ranking."""
        retrieved = np.array([3, 4, 2, 1, 0])
        relevant = [0, 1]  # All relevant docs at bottom

        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=5)

        # Should be less than perfect
        assert 0.0 < ndcg < 1.0

    def test_no_relevant_found(self):
        """Test NDCG when no relevant docs retrieved."""
        retrieved = np.array([2, 3, 4, 5])
        relevant = [0, 1]

        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=5)

        assert ndcg == 0.0

    def test_partial_ranking(self):
        """Test NDCG with some relevant docs."""
        retrieved = np.array([0, 5, 1, 6, 7])
        relevant = [0, 1]  # Scattered in ranking

        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=5)

        # Should be between 0 and 1
        assert 0.0 < ndcg <= 1.0

    def test_k_cutoff(self):
        """Test NDCG with k cutoff."""
        retrieved = np.array([5, 6, 7, 0, 1])
        relevant = [0, 1]  # Relevant docs after k=3

        ndcg_3 = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=3)
        ndcg_5 = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=5)

        # k=3 should be worse than k=5
        assert ndcg_3 < ndcg_5

    def test_single_relevant_doc(self):
        """Test NDCG with single relevant doc."""
        retrieved = np.array([0, 1, 2])
        relevant = [0]

        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=3)

        assert ndcg == 1.0


class TestMAP:
    """Test Mean Average Precision metric."""

    def test_perfect_precision(self):
        """Test MAP when all retrieved are relevant."""
        retrieved = np.array([0, 1, 2])
        relevant = [0, 1, 2]

        map_score = RetrievalEvaluator.compute_map(retrieved, relevant, k=3)

        assert map_score == 1.0

    def test_alternating_relevant(self):
        """Test MAP with alternating relevant/irrelevant."""
        retrieved = np.array([0, 5, 1, 6, 2])
        relevant = [0, 1, 2]

        map_score = RetrievalEvaluator.compute_map(retrieved, relevant, k=5)

        # P@1 = 1/1, P@3 = 2/3, P@5 = 3/5
        # MAP = (1.0 + 2/3 + 3/5) / 3
        expected = (1.0 + 2.0/3.0 + 3.0/5.0) / 3.0
        assert abs(map_score - expected) < 1e-6

    def test_no_relevant_found(self):
        """Test MAP when no relevant docs retrieved."""
        retrieved = np.array([5, 6, 7])
        relevant = [0, 1, 2]

        map_score = RetrievalEvaluator.compute_map(retrieved, relevant, k=3)

        assert map_score == 0.0

    def test_k_cutoff(self):
        """Test MAP with different k values."""
        retrieved = np.array([0, 5, 6, 1, 7, 8, 2])
        relevant = [0, 1, 2]

        map_3 = RetrievalEvaluator.compute_map(retrieved, relevant, k=3)
        map_7 = RetrievalEvaluator.compute_map(retrieved, relevant, k=7)

        # k=7 should be better (retrieves all relevant)
        assert map_7 >= map_3

    def test_empty_relevant(self):
        """Test MAP with no relevant documents."""
        retrieved = np.array([0, 1, 2])
        relevant = []

        map_score = RetrievalEvaluator.compute_map(retrieved, relevant, k=3)

        assert map_score == 0.0


class TestRetrievalEvaluatorCore:
    """Test RetrievalEvaluator core functionality."""

    @pytest.mark.slow
    def test_evaluator_initialization(self, embedding_dim, device):
        """Test evaluator initializes correctly."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        evaluator = RetrievalEvaluator(model, device)

        assert evaluator.model is not None
        assert not evaluator.model.training  # Should be in eval mode

    @pytest.mark.slow
    def test_encode_corpus(self, embedding_dim, device, sample_texts):
        """Test corpus encoding."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        evaluator = RetrievalEvaluator(model, device)

        embeddings = evaluator.encode_corpus(sample_texts, batch_size=16)

        assert embeddings.shape == (len(sample_texts), embedding_dim)
        assert isinstance(embeddings, np.ndarray)

    @pytest.mark.slow
    def test_retrieve_returns_correct_format(self, embedding_dim, device, sample_texts):
        """Test that retrieve returns indices and distances."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        evaluator = RetrievalEvaluator(model, device)

        corpus_embeddings = evaluator.encode_corpus(sample_texts)
        indices, distances = evaluator.retrieve("query", corpus_embeddings, top_k=3)

        assert len(indices) == 3
        assert len(distances) == 3
        assert all(isinstance(i, (int, np.integer)) for i in indices)
        assert all(isinstance(d, (float, np.floating)) for d in distances)

    @pytest.mark.slow
    def test_retrieve_uses_euclidean_distance(self, embedding_dim, device, sample_texts):
        """Test that retrieve uses Euclidean distance."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        evaluator = RetrievalEvaluator(model, device)

        corpus_embeddings = evaluator.encode_corpus(sample_texts)
        _, distances = evaluator.retrieve("query", corpus_embeddings, top_k=3)

        # Euclidean distances are non-negative
        assert all(d >= 0 for d in distances)

    @pytest.mark.slow
    def test_retrieve_sorted_by_distance(self, embedding_dim, device, sample_texts):
        """Test that results are sorted by distance (ascending)."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        evaluator = RetrievalEvaluator(model, device)

        corpus_embeddings = evaluator.encode_corpus(sample_texts)
        _, distances = evaluator.retrieve("query", corpus_embeddings, top_k=5)

        # Should be sorted ascending
        assert list(distances) == sorted(distances)

    @pytest.mark.slow
    def test_retrieve_top_k_parameter(self, embedding_dim, device, sample_texts):
        """Test that top_k parameter works correctly."""
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        evaluator = RetrievalEvaluator(model, device)

        corpus_embeddings = evaluator.encode_corpus(sample_texts)

        for k in [1, 3, 5]:
            indices, distances = evaluator.retrieve("query", corpus_embeddings, top_k=k)
            assert len(indices) == min(k, len(sample_texts))
            assert len(distances) == min(k, len(sample_texts))


class TestMetricEdgeCases:
    """Test edge cases for metrics."""

    def test_all_metrics_with_empty_retrieved(self):
        """Test all metrics with empty retrieved list."""
        retrieved = np.array([])
        relevant = [0, 1, 2]

        recall = RetrievalEvaluator.compute_recall_at_k(retrieved, relevant, k_values=[1])
        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)
        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=1)
        map_score = RetrievalEvaluator.compute_map(retrieved, relevant, k=1)

        # Empty retrieved - key is still original k (1), not effective k (0)
        assert recall[1] == 0.0
        assert mrr == 0.0
        assert ndcg == 0.0
        assert map_score == 0.0

    def test_metrics_with_single_doc(self):
        """Test metrics with single document."""
        retrieved = np.array([0])
        relevant = [0]

        recall = RetrievalEvaluator.compute_recall_at_k(retrieved, relevant, k_values=[1])
        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)
        ndcg = RetrievalEvaluator.compute_ndcg(retrieved, relevant, k=1)
        map_score = RetrievalEvaluator.compute_map(retrieved, relevant, k=1)

        # All should be perfect
        assert recall[1] == 1.0
        assert mrr == 1.0
        assert ndcg == 1.0
        assert map_score == 1.0

    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        retrieved = np.array([0, 1, 2, 3, 4])
        relevant = [0]  # Only first is relevant

        recall_1 = RetrievalEvaluator.compute_recall_at_k(retrieved, relevant, k_values=[1])
        mrr = RetrievalEvaluator.compute_mrr(retrieved, relevant)

        # Both should indicate perfect retrieval at top-1
        assert recall_1[1] == 1.0
        assert mrr == 1.0
