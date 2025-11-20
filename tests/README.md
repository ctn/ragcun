# RAGCUN Unit Tests

Comprehensive unit tests for LeJEPA Isotropic Gaussian Embeddings.

## Overview

This test suite ensures that:
- ✅ Model encoding works correctly
- ✅ Embeddings are NOT normalized (critical!)
- ✅ Retrieval uses Euclidean distance (not cosine)
- ✅ Isotropic Gaussian properties hold
- ✅ Data preparation functions work
- ✅ Evaluation metrics are correct
- ✅ End-to-end workflows complete successfully

## Test Files

```
tests/
├── conftest.py                    # Pytest fixtures and configuration
├── test_model.py                  # IsotropicGaussianEncoder tests (P0)
├── test_retriever.py              # IsotropicRetriever tests (P0)
├── test_data_preparation.py       # Data loading/preparation tests
├── test_evaluation.py             # Metrics (Recall@K, MRR, NDCG, MAP)
├── test_integration.py            # End-to-end workflows
├── test_properties.py             # Isotropy & Gaussian properties
└── fixtures/                      # Sample test data
    ├── sample_documents.txt
    ├── sample_queries.txt
    └── README.md
```

## Running Tests

### Install Test Dependencies

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Or install pytest manually
pip install pytest pytest-cov scipy
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Model tests only
pytest tests/test_model.py

# Retriever tests only
pytest tests/test_retriever.py

# Fast tests only (skip slow model loading)
pytest tests/ -m "not slow"

# Integration tests only
pytest tests/ -m integration
```

### Run with Coverage

```bash
pytest tests/ --cov=ragcun --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Run Specific Tests

```bash
# Run single test
pytest tests/test_model.py::TestIsotropicGaussianEncoderEncoding::test_encode_single_string

# Run test class
pytest tests/test_model.py::TestIsotropicGaussianEncoderEncoding

# Run tests matching pattern
pytest tests/ -k "encode"
```

### Verbose Output

```bash
pytest tests/ -v           # Verbose
pytest tests/ -vv          # Very verbose
pytest tests/ -s           # Show print statements
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.slow` - Tests that load models (slow, ~5-10s each)
- `@pytest.mark.gpu` - Tests that require GPU
- `@pytest.mark.integration` - End-to-end integration tests

### Skip Slow Tests

```bash
# Skip slow tests (useful for quick iteration)
pytest tests/ -m "not slow"
```

### Run Only GPU Tests

```bash
pytest tests/ -m gpu
```

## Test Categories

### P0 (Critical - Must Pass)

These tests ensure core functionality:

```bash
# Critical model tests
pytest tests/test_model.py::TestIsotropicGaussianEncoderEncoding::test_embeddings_are_not_normalized
pytest tests/test_retriever.py::TestIsotropicRetrieverRetrieval::test_retrieve_uses_euclidean_distance
```

**Why P0?**
- Embeddings MUST NOT be normalized (core innovation)
- Retrieval MUST use Euclidean distance (not cosine)
- These properties are what make the method work

### P1 (High Priority)

```bash
pytest tests/test_model.py
pytest tests/test_retriever.py
pytest tests/test_evaluation.py
```

### P2 (Nice to Have)

```bash
pytest tests/test_properties.py
pytest tests/test_integration.py
```

## Key Tests Explained

### 1. Embeddings Are NOT Normalized

```python
# tests/test_model.py
def test_embeddings_are_not_normalized():
    """CRITICAL: Embeddings must have varying norms."""
```

**Why?** This is the core difference from traditional embeddings. Normalized embeddings waste the magnitude dimension.

### 2. Euclidean Distance Used

```python
# tests/test_retriever.py
def test_retrieve_uses_euclidean_distance():
    """Retrieval must use L2 distance, not cosine."""
```

**Why?** Euclidean distance provides better separation than cosine similarity for Gaussian embeddings.

### 3. Isotropy Properties

```python
# tests/test_properties.py
def test_embedding_mean_near_zero():
    """Mean of embeddings should be ~0."""
```

**Why?** Verifies that embeddings follow N(0,I) distribution as intended.

## Continuous Integration

For CI/CD pipelines:

```bash
# Fast test suite (skip slow model loading)
pytest tests/ -m "not slow" --cov=ragcun

# Full test suite
pytest tests/ --cov=ragcun --cov-report=xml
```

## Troubleshooting

### Tests are slow

```bash
# Skip slow tests during development
pytest tests/ -m "not slow"
```

Most slow tests involve loading the EmbeddingGemma-300M model. These are marked with `@pytest.mark.slow`.

### Model download issues

If tests fail due to model download:
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('google/embeddinggemma-300m')"
```

### GPU tests failing

GPU tests are automatically skipped if CUDA is not available. To run them:
```bash
pytest tests/ -m gpu  # Only if you have GPU
```

### Import errors

Make sure the package is installed:
```bash
pip install -e .
```

## Writing New Tests

### Test Structure

```python
import pytest
from ragcun.model import IsotropicGaussianEncoder

class TestMyFeature:
    """Test description."""

    @pytest.mark.slow  # If it loads the model
    def test_something(self, embedding_dim):
        """Test that something works."""
        model = IsotropicGaussianEncoder(output_dim=embedding_dim)
        # ... test code
        assert result == expected
```

### Using Fixtures

Common fixtures from `conftest.py`:

- `sample_texts` - List of sample documents
- `sample_queries` - List of sample queries
- `embedding_dim` - Small embedding dimension (128) for fast tests
- `temp_dir` - Temporary directory for file operations
- `mock_model_checkpoint` - Pre-created model checkpoint

### Best Practices

1. **Mark slow tests**: Use `@pytest.mark.slow` for model loading
2. **Use fixtures**: Don't recreate common test data
3. **Clear assertions**: Use descriptive assertion messages
4. **Test edge cases**: Empty inputs, single items, etc.
5. **Test critical properties**: Focus on what makes this method unique

## Expected Test Results

When all tests pass, you should see:

```
tests/test_model.py ..................                 [ 20%]
tests/test_retriever.py .......................        [ 45%]
tests/test_data_preparation.py ................        [ 65%]
tests/test_evaluation.py ...................           [ 85%]
tests/test_integration.py ..........                   [ 95%]
tests/test_properties.py ........                      [100%]

=================== X passed in Y.YYs ===================
```

**Note:** Slow tests (marked with `@pytest.mark.slow`) will take longer due to model loading.

## Coverage Goals

Target coverage by module:

- `ragcun/model.py`: 90%+
- `ragcun/retriever.py`: 95%+
- `scripts/prepare_data.py`: 80%+
- `scripts/evaluate.py`: 85%+

Check coverage:
```bash
pytest tests/ --cov=ragcun --cov-report=term-missing
```

## Questions?

- Check test output with `-v` flag
- Read test docstrings for explanations
- Review `conftest.py` for available fixtures
- See individual test files for examples
