"""
Pytest configuration and shared fixtures for RAGCUN tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, MagicMock


class MockSentenceTransformer(nn.Module):
    """Mock SentenceTransformer for testing without loading gated models."""

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        # Create a simple embedding layer to simulate the base model
        self.embedding_dim = 768
        self.embedder = nn.Linear(100, self.embedding_dim)  # Dummy embedder

        # Create encoder layers that can actually be frozen
        # Create a nested structure that mimics transformer architecture
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.Linear(768, 768),
                'output': nn.Linear(768, 768)
            }) for _ in range(12)
        ])

    def encode(self, texts, batch_size=32, convert_to_tensor=False,
               show_progress_bar=False, normalize_embeddings=True):
        """Mock encode method that returns dummy embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty list
        if len(texts) == 0:
            if convert_to_tensor:
                return torch.empty(0, self.embedding_dim)
            return np.empty((0, self.embedding_dim))

        # Create deterministic embeddings based on text hash
        # Use larger standard deviation to create more varied embeddings
        embeddings = []
        for i, text in enumerate(texts):
            # Use hash for deterministic but varied embeddings
            seed = hash(text) % (2**32)
            torch.manual_seed(seed)
            # Create embeddings with varying norms (more realistic for Gaussian embeddings)
            # Add text-specific scaling to create more norm variation
            scale = 1.0 + (seed % 100) / 50.0  # Scale between 1.0 and 3.0
            emb = torch.randn(self.embedding_dim) * scale
            if normalize_embeddings:
                emb = emb / emb.norm()
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)

        if convert_to_tensor:
            return embeddings
        return embeddings.numpy()


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Automatically mock SentenceTransformer for all tests."""
    # Patch where it's used, not where it's defined
    import ragcun.model
    monkeypatch.setattr(ragcun.model, 'SentenceTransformer', MockSentenceTransformer)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Data science combines statistics, programming, and domain knowledge."
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for retrieval testing."""
    return [
        "What is machine learning?",
        "Tell me about Python programming",
        "How does NLP work?"
    ]


@pytest.fixture
def sample_triplets():
    """Sample query-positive-negative triplets for training."""
    return [
        {
            "query": "What is Python?",
            "positive": "Python is a high-level programming language.",
            "negative": "JavaScript is used for web development."
        },
        {
            "query": "Explain machine learning",
            "positive": "Machine learning is a subset of artificial intelligence.",
            "negative": "Databases store structured data."
        },
        {
            "query": "What is NLP?",
            "positive": "Natural language processing enables computers to understand human language.",
            "negative": "Computer graphics renders visual images."
        }
    ]


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension for tests."""
    return 128  # Smaller dim for faster tests


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing (not normalized, Gaussian-like)."""
    # Create embeddings with varying norms (characteristic of Gaussian embeddings)
    np.random.seed(42)
    embeddings = np.random.randn(10, 128)  # Mean ~0, std ~1
    return embeddings


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create a sample CSV file for testing data loading."""
    csv_path = temp_dir / "sample_pairs.csv"
    content = """query,positive,negative
What is Python?,Python is a programming language.,Java is compiled.
Explain ML,Machine learning uses algorithms.,Databases store data.
What is NLP?,NLP processes human language.,Graphics render images.
"""
    csv_path.write_text(content)
    return csv_path


@pytest.fixture
def sample_documents_dir(temp_dir, sample_texts):
    """Create a directory with sample text files."""
    docs_dir = temp_dir / "documents"
    docs_dir.mkdir()

    for i, text in enumerate(sample_texts):
        doc_path = docs_dir / f"doc_{i}.txt"
        doc_path.write_text(text)

    return docs_dir


@pytest.fixture
def sample_json_data(temp_dir, sample_triplets):
    """Create a sample JSON file with training data."""
    json_path = temp_dir / "train.json"
    import json
    with open(json_path, 'w') as f:
        json.dump(sample_triplets, f)
    return json_path


@pytest.fixture(scope="session")
def device():
    """Get the device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def mock_model_checkpoint(temp_dir, embedding_dim):
    """Create a mock model checkpoint for testing loading."""
    # Manually create state dict without loading the actual model
    # This mimics the structure of GaussianEmbeddingGemma
    state_dict = {
        # Projection layers (768 -> 768*2 -> output_dim)
        'projection.0.weight': torch.randn(768 * 2, 768),
        'projection.0.bias': torch.randn(768 * 2),
        'projection.3.weight': torch.randn(embedding_dim, 768 * 2),
        'projection.3.bias': torch.randn(embedding_dim),
        # Mock base model embedder
        'base.embedder.weight': torch.randn(768, 100),
        'base.embedder.bias': torch.randn(768),
    }

    # Add encoder layers
    for i in range(12):
        state_dict[f'base.encoder.layer.{i}.attention.weight'] = torch.randn(768, 768)
        state_dict[f'base.encoder.layer.{i}.attention.bias'] = torch.randn(768)
        state_dict[f'base.encoder.layer.{i}.output.weight'] = torch.randn(768, 768)
        state_dict[f'base.encoder.layer.{i}.output.bias'] = torch.randn(768)

    # Save checkpoint
    checkpoint_path = temp_dir / "mock_model.pt"
    torch.save({
        'model_state_dict': state_dict,
        'embedding_dim': embedding_dim,
        'epoch': 5
    }, checkpoint_path)

    return checkpoint_path


def pytest_configure(config):
    """Pytest configuration hook."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
