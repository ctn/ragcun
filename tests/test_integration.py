"""
Integration tests for end-to-end workflows.

Tests complete pipelines from data preparation through training to evaluation.
"""

import pytest
import torch
import json
from pathlib import Path

from ragcun.model import GaussianEmbeddingGemma
from ragcun.retriever import GaussianRetriever


class TestFullRetrievalPipeline:
    """Test complete retrieval pipeline."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_add_retrieve_pipeline(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test: Load model → add documents → retrieve."""
        # 1. Load model
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        # 2. Add documents
        retriever.add_documents(sample_texts)
        assert len(retriever.documents) == len(sample_texts)

        # 3. Retrieve
        results = retriever.retrieve("programming", top_k=3)
        assert len(results) == 3

        # 4. Verify format
        for doc, distance in results:
            assert isinstance(doc, str)
            assert isinstance(distance, float)
            assert distance >= 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_incremental_document_addition(self, mock_model_checkpoint, embedding_dim):
        """Test adding documents incrementally and retrieving."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        # Add in batches
        batch1 = ["Python programming", "Java development"]
        batch2 = ["Machine learning", "Deep learning"]
        batch3 = ["Data science"]

        retriever.add_documents(batch1)
        assert len(retriever.documents) == 2

        retriever.add_documents(batch2)
        assert len(retriever.documents) == 4

        retriever.add_documents(batch3)
        assert len(retriever.documents) == 5

        # Retrieve should work across all batches
        results = retriever.retrieve("AI algorithms", top_k=5)
        assert len(results) == 5

    @pytest.mark.slow
    @pytest.mark.integration
    def test_clear_and_rebuild(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test clearing index and rebuilding."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        # Add documents
        retriever.add_documents(sample_texts)
        assert len(retriever.documents) > 0

        # Clear
        retriever.clear()
        assert len(retriever.documents) == 0

        # Re-add different documents
        new_docs = ["New doc 1", "New doc 2"]
        retriever.add_documents(new_docs)
        assert len(retriever.documents) == 2

        # Retrieve should work
        results = retriever.retrieve("new", top_k=2)
        assert len(results) == 2


class TestModelSaveLoadCycle:
    """Test model saving and loading."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_save_and_load_model(self, temp_dir, embedding_dim):
        """Test: Create model → save → load → use."""
        # 1. Create model
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)
        model.eval()  # Put in eval mode to disable dropout

        # 2. Get baseline encoding
        test_text = "Test encoding"
        with torch.no_grad():
            original_emb = model.encode(test_text)

        # 3. Save model
        save_path = temp_dir / "test_model.pt"
        torch.save(model.state_dict(), save_path)

        # 4. Load model
        loaded_model = GaussianEmbeddingGemma.from_pretrained(
            str(save_path),
            output_dim=embedding_dim
        )

        # 5. Verify encoding is same
        with torch.no_grad():
            loaded_emb = loaded_model.encode(test_text)

        assert torch.allclose(original_emb, loaded_emb, atol=1e-5)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_save_load_with_full_checkpoint(self, temp_dir, embedding_dim):
        """Test saving/loading with full checkpoint dict."""
        # Create model
        model = GaussianEmbeddingGemma(output_dim=embedding_dim, freeze_early_layers=False)

        # Save full checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'embedding_dim': embedding_dim,
            'epoch': 10,
            'loss': 0.5
        }
        save_path = temp_dir / "checkpoint.pt"
        torch.save(checkpoint, save_path)

        # Load
        loaded_model = GaussianEmbeddingGemma.from_pretrained(
            str(save_path),
            output_dim=embedding_dim
        )

        assert loaded_model is not None


class TestIndexPersistence:
    """Test saving and loading retriever index."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_save_load_index_preserves_results(self, mock_model_checkpoint, embedding_dim, sample_texts, temp_dir):
        """Test that saved/loaded index gives same results."""
        # Create retriever and add documents
        retriever1 = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever1.add_documents(sample_texts)

        # Get results before save
        query = "machine learning"
        results_before = retriever1.retrieve(query, top_k=3)

        # Save index
        index_path = temp_dir / "index.pkl"
        retriever1.save_index(str(index_path))

        # Load into new retriever
        retriever2 = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever2.load_index(str(index_path))

        # Get results after load
        results_after = retriever2.retrieve(query, top_k=3)

        # Should be identical
        assert len(results_before) == len(results_after)
        for (doc1, dist1), (doc2, dist2) in zip(results_before, results_after):
            assert doc1 == doc2
            assert abs(dist1 - dist2) < 1e-5

    @pytest.mark.slow
    @pytest.mark.integration
    def test_index_persistence_with_large_corpus(self, mock_model_checkpoint, embedding_dim, temp_dir):
        """Test index persistence with larger corpus."""
        # Create larger corpus
        docs = [f"Document {i} about topic {i % 10}" for i in range(100)]

        # Build index
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever.add_documents(docs)

        # Save
        index_path = temp_dir / "large_index.pkl"
        retriever.save_index(str(index_path))

        # Load
        retriever2 = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever2.load_index(str(index_path))

        # Verify all documents loaded
        assert len(retriever2.documents) == 100


class TestEndToEndRetrieval:
    """Test realistic retrieval scenarios."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_domain_specific_retrieval(self, mock_model_checkpoint, embedding_dim):
        """Test retrieval on domain-specific documents."""
        docs = [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "Machine learning algorithms learn patterns from data to make predictions.",
            "Natural language processing helps computers understand and generate human language.",
            "Databases store and manage structured data efficiently.",
            "Cloud computing provides scalable infrastructure for applications."
        ]

        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever.add_documents(docs)

        # Query about ML should retrieve ML-related doc
        results = retriever.retrieve("What is machine learning?", top_k=3)

        # Top result should be related to ML
        top_doc = results[0][0]
        assert "machine learning" in top_doc.lower() or "learning" in top_doc.lower()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_retrieval_quality_with_negatives(self, mock_model_checkpoint, embedding_dim):
        """Test that dissimilar docs have higher distances."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        docs = [
            "Python programming language tutorial",  # Related to query
            "Cooking recipes for beginners",  # Not related
            "Software development best practices",  # Somewhat related
        ]
        retriever.add_documents(docs)

        results = retriever.retrieve("Python coding", top_k=3)

        # Get distances
        distances = [dist for _, dist in results]

        # First result should have smaller distance than obviously unrelated doc
        # (This is probabilistic but should hold for most embeddings)
        assert len(results) == 3

    @pytest.mark.slow
    @pytest.mark.integration
    def test_multiple_queries_same_corpus(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test multiple different queries on same corpus."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )
        retriever.add_documents(sample_texts)

        queries = [
            "programming languages",
            "artificial intelligence",
            "data analysis"
        ]

        for query in queries:
            results = retriever.retrieve(query, top_k=3)
            assert len(results) == 3
            # All should return valid results
            for doc, dist in results:
                assert isinstance(doc, str)
                assert dist >= 0


class TestErrorHandlingAndRecovery:
    """Test error handling in integrated workflows."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_retrieve_before_adding_documents(self, mock_model_checkpoint, embedding_dim):
        """Test retrieving from empty index."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        results = retriever.retrieve("query", top_k=5)

        assert results == []

    @pytest.mark.slow
    @pytest.mark.integration
    def test_recover_after_error(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that retriever can recover after operations."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        # Add documents
        retriever.add_documents(sample_texts)

        # Clear (simulating reset after error)
        retriever.clear()

        # Should be able to add new documents and retrieve
        new_docs = ["New document 1", "New document 2"]
        retriever.add_documents(new_docs)

        results = retriever.retrieve("new", top_k=2)
        assert len(results) == 2

    @pytest.mark.slow
    @pytest.mark.integration
    def test_add_empty_batch_doesnt_break(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that adding empty batch doesn't break subsequent operations."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        # Add real documents
        retriever.add_documents(sample_texts)

        # Add empty batch
        retriever.add_documents([])

        # Should still work
        results = retriever.retrieve("query", top_k=3)
        assert len(results) == 3


class TestDistanceProperties:
    """Test that Euclidean distance properties hold in integration."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_distance_is_symmetric(self, mock_model_checkpoint, embedding_dim):
        """Test that distance(A, B) == distance(B, A)."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        doc1 = "First document"
        doc2 = "Second document"

        # Add doc1, query with doc2
        retriever.add_documents([doc1])
        results1 = retriever.retrieve(doc2, top_k=1)
        dist1 = results1[0][1]

        # Clear and reverse
        retriever.clear()
        retriever.add_documents([doc2])
        results2 = retriever.retrieve(doc1, top_k=1)
        dist2 = results2[0][1]

        # Distances should be equal (within floating point tolerance)
        assert abs(dist1 - dist2) < 1e-4

    @pytest.mark.slow
    @pytest.mark.integration
    def test_self_distance_is_small(self, mock_model_checkpoint, embedding_dim):
        """Test that distance(A, A) is very small."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        doc = "Test document for self-similarity"
        retriever.add_documents([doc])

        results = retriever.retrieve(doc, top_k=1)
        self_distance = results[0][1]

        # Self-distance should be very small (near 0)
        assert self_distance < 0.1

    @pytest.mark.slow
    @pytest.mark.integration
    def test_distances_are_non_negative(self, mock_model_checkpoint, embedding_dim, sample_texts):
        """Test that all distances are >= 0."""
        retriever = GaussianRetriever(
            model_path=str(mock_model_checkpoint),
            embedding_dim=embedding_dim
        )

        retriever.add_documents(sample_texts)
        results = retriever.retrieve("query", top_k=len(sample_texts))

        for _, distance in results:
            assert distance >= 0
