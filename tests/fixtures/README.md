# Test Fixtures

This directory contains sample data files used in unit tests.

## Files

- `sample_documents.txt` - Sample documents for retrieval testing
- `sample_queries.txt` - Sample queries for retrieval testing
- `sample_pairs.csv` - Created dynamically by conftest.py fixtures

## Usage

These fixtures are automatically loaded by pytest through the conftest.py file.
You don't need to manually load them in your tests - just use the appropriate fixture.

## Example

```python
def test_something(sample_texts):
    # sample_texts is automatically loaded from fixtures
    assert len(sample_texts) > 0
```
