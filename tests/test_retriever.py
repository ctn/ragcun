"""
Unit tests for GaussianRetriever.

Tests retrieval functionality, Euclidean distance usage,
and document management.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from ragcun.retriever import GaussianRetriever


class TestGaussianRetrieverInitialization:
    """Test retriever initialization."""

    @pytest.mark.slow
    def test_init_with_model_path(self, mock_model_checkpoint, embedding_dim):
        """Test initialization with a model path."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        assert retriever is not None
        assert retriever.embedding_dim == embedding_dim
        assert retriever.model is not None
        assert not retriever.model.training  # Should be in eval mode

    @pytest.mark.slow
    def test_init_without_model_path(self, embedding_dim):
        """Test initialization without model path (untrained)."""
        # Should show warning but still work
        retriever = GaussianRetriever(
            model_path=None,
            embedding_dim=embedding_dim
        )

        assert retriever is not None
        assert retriever.model is not None

    @pytest.mark.slow
    def test_embedding_dim_parameter(self, mock_model_checkpoint, embedding_dim):
        """Test that embedding_dim parameter is stored correctly."""
        # Use the same dim as the checkpoint was created with
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        assert retriever.embedding_dim == embedding_dim

    @pytest.mark.slow
    def test_initial_state_is_empty(self, mock_model_checkpoint, embedding_dim):
        """Test that retriever starts with no documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        assert len(retriever.documents) == 0
        assert retriever.embeddings is None
        assert retriever.index is None

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_gpu_flag(self, mock_model_checkpoint, embedding_dim):
        """Test GPU flag when CUDA is available."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim,
            use_gpu=True
        )

        # Should be True only if CUDA is actually available
        assert retriever.use_gpu == torch.cuda.is_available()


class TestGaussianRetrieverDocumentManagement:
    """Test document addition and management."""

    @pytest.mark.slow
    def test_add_documents_single_batch(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test adding documents in a single batch."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)

        assert len(retriever.documents) == len(sample_texts)
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape == (len(sample_texts), embedding_dim)

    @pytest.mark.slow
    def test_add_documents_multiple_batches(self, mock_model_checkpoint, embedding_dim):
        """Test adding documents in multiple batches."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        batch1 = ["Doc 1", "Doc 2"]
        batch2 = ["Doc 3", "Doc 4", "Doc 5"]

        retriever.add_documents(batch1)
        assert len(retriever.documents) == 2

        retriever.add_documents(batch2)
        assert len(retriever.documents) == 5
        assert retriever.embeddings.shape == (5, embedding_dim)

    @pytest.mark.slow
    def test_add_empty_documents(self, mock_model_checkpoint, embedding_dim):
        """Test adding empty list of documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents([])

        assert len(retriever.documents) == 0
        assert retriever.embeddings is None

    @pytest.mark.slow
    def test_clear_documents(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test clearing all documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)
        assert len(retriever.documents) > 0

        retriever.clear()

        assert len(retriever.documents) == 0
        assert retriever.embeddings is None
        assert retriever.index is None

    @pytest.mark.slow
    def test_document_count_after_additions(self, mock_model_checkpoint, embedding_dim):
        """Test document count is correct after multiple additions."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(["A", "B"])
        retriever.add_documents(["C", "D", "E"])
        retriever.add_documents(["F"])

        assert len(retriever.documents) == 6


class TestGaussianRetrieverRetrieval:
    """Test retrieval functionality."""

    @pytest.mark.slow
    def test_retrieve_returns_top_k_results(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that retrieve returns exactly top_k results."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)

        results = retriever.retrieve("Python programming", top_k=3)

        assert len(results) == 3

    @pytest.mark.slow
    def test_retrieve_with_k_greater_than_docs(self, mock_model_checkpoint, embedding_dim):
        """Test retrieve when k > number of documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        docs = ["Doc 1", "Doc 2"]
        retriever.add_documents(docs)

        results = retriever.retrieve("query", top_k=10)

        # Should return only available docs
        assert len(results) <= len(docs)

    @pytest.mark.slow
    def test_retrieve_on_empty_index(self, mock_model_checkpoint, embedding_dim):
        """Test retrieve when no documents are added."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        results = retriever.retrieve("query", top_k=5)

        assert results == []

    @pytest.mark.slow
    def test_retrieve_results_format(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that results are (document, distance) tuples."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)
        results = retriever.retrieve("test query", top_k=2)

        assert len(results) == 2
        for doc, distance in results:
            assert isinstance(doc, str)
            assert isinstance(distance, float)
            assert distance >= 0  # Euclidean distance is non-negative

    @pytest.mark.slow
    def test_retrieve_results_sorted_by_distance(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that results are sorted by distance (ascending)."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)
        results = retriever.retrieve("query", top_k=len(sample_texts))

        # Distances should be in ascending order
        distances = [dist for _, dist in results]
        assert distances == sorted(distances)

    @pytest.mark.slow
    def test_retrieve_finds_semantically_similar(self, mock_model_checkpoint, embedding_dim):
        """Test that retrieval finds semantically similar documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        docs = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "Python programming is popular"
        ]
        retriever.add_documents(docs)

        # Query about Python should retrieve Python-related docs
        results = retriever.retrieve("Python coding", top_k=2)

        # Top results should contain "Python"
        top_docs = [doc for doc, _ in results]
        python_count = sum(1 for doc in top_docs if "Python" in doc or "python" in doc)
        assert python_count >= 1, "Should retrieve at least one Python-related document"

    @pytest.mark.slow
    def test_retrieve_uses_euclidean_distance(self, mock_model_checkpoint, embedding_dim):
        """Test that distances are Euclidean (L2), not cosine similarity."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        docs = ["Test document 1", "Test document 2"]
        retriever.add_documents(docs)

        results = retriever.retrieve("query", top_k=2)

        # Euclidean distances can be > 1 (unlike cosine similarity which is [0,1])
        # Just verify they're non-negative
        for _, distance in results:
            assert distance >= 0
            # Could be > 1, which would be impossible for cosine similarity


class TestGaussianRetrieverIndexOperations:
    """Test FAISS index building and operations."""

    @pytest.mark.slow
    def test_index_is_built_after_adding_docs(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that index is built after adding documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)

        # Index should be built (either FAISS or None for numpy fallback)
        # Just check that _build_index was called by verifying embeddings exist
        assert retriever.embeddings is not None

    @pytest.mark.slow
    def test_save_and_load_index(self, mock_model_checkpoint, embedding_dim, sample_texts, temp_dir):
        """Test saving and loading index."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)

        # Save index
        index_path = temp_dir / "test_index.pkl"
        retriever.save_index(str(index_path))

        assert index_path.exists()

        # Load into new retriever
        retriever2 = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever2.load_index(str(index_path))

        assert len(retriever2.documents) == len(sample_texts)
        assert retriever2.embeddings.shape == (len(sample_texts), embedding_dim)

    @pytest.mark.slow
    def test_loaded_index_produces_same_results(self, mock_model_checkpoint, embedding_dim, sample_texts, temp_dir):
        """Test that loaded index produces same retrieval results."""
        retriever1 = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever1.add_documents(sample_texts)

        query = "test query"
        results1 = retriever1.retrieve(query, top_k=3)

        # Save and load
        index_path = temp_dir / "test_index.pkl"
        retriever1.save_index(str(index_path))

        retriever2 = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever2.load_index(str(index_path))

        results2 = retriever2.retrieve(query, top_k=3)

        # Results should be identical
        assert len(results1) == len(results2)
        for (doc1, dist1), (doc2, dist2) in zip(results1, results2):
            assert doc1 == doc2
            assert abs(dist1 - dist2) < 1e-5  # Allow small floating point differences


class TestGaussianRetrieverEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.slow
    def test_retrieve_with_top_k_zero(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test retrieve with top_k=0."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)
        results = retriever.retrieve("query", top_k=0)

        assert len(results) == 0

    @pytest.mark.slow
    def test_retrieve_with_empty_query(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test retrieve with empty string query."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)

        # Should still work, just might not be meaningful
        results = retriever.retrieve("", top_k=3)
        assert len(results) == 3

    @pytest.mark.slow
    def test_add_documents_with_duplicates(self, mock_model_checkpoint, embedding_dim):
        """Test adding duplicate documents."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        docs = ["Same doc", "Same doc", "Different doc"]
        retriever.add_documents(docs)

        # Should add all, including duplicates
        assert len(retriever.documents) == 3

    @pytest.mark.slow
    def test_retrieve_after_clear(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that retrieve returns empty after clear."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)
        retriever.clear()

        results = retriever.retrieve("query", top_k=5)
        assert results == []
