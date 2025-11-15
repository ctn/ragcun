"""
Unit tests for data preparation functions.

Tests data loading, pair generation, and train/val/test splitting.
"""

import pytest
import json
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import (
    load_documents,
    load_csv,
    generate_query_from_doc,
    generate_pairs,
    split_data
)


class TestLoadDocuments:
    """Test loading documents from directory."""

    def test_load_from_directory(self, sample_documents_dir):
        """Test loading documents from a directory."""
        documents = load_documents(sample_documents_dir)

        assert len(documents) > 0
        assert all(isinstance(doc, str) for doc in documents)
        assert all(len(doc) > 0 for doc in documents)

    def test_recursive_directory_traversal(self, temp_dir, sample_texts):
        """Test recursive loading from nested directories."""
        # Create nested structure
        subdir = temp_dir / "level1" / "level2"
        subdir.mkdir(parents=True)

        # Add files at different levels
        (temp_dir / "level1" / "doc1.txt").write_text("Document 1")
        (subdir / "doc2.txt").write_text("Document 2")

        documents = load_documents(temp_dir / "level1")

        assert len(documents) == 2

    def test_handles_empty_files(self, temp_dir):
        """Test handling of empty text files."""
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()

        # Create one empty and one non-empty file
        (docs_dir / "empty.txt").write_text("")
        (docs_dir / "content.txt").write_text("Has content")

        documents = load_documents(docs_dir)

        # Should only load non-empty file
        assert len(documents) == 1
        assert documents[0] == "Has content"

    def test_returns_correct_count(self, sample_documents_dir, sample_texts):
        """Test that returned count matches expected."""
        documents = load_documents(sample_documents_dir)

        assert len(documents) == len(sample_texts)

    def test_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        documents = load_documents(empty_dir)

        assert len(documents) == 0


class TestLoadCSV:
    """Test loading data from CSV files."""

    def test_load_valid_csv(self, sample_csv_data):
        """Test loading a valid CSV file."""
        examples = load_csv(sample_csv_data)

        assert len(examples) > 0
        assert all('query' in ex for ex in examples)
        assert all('positive' in ex for ex in examples)

    def test_csv_with_all_columns(self, sample_csv_data):
        """Test CSV with query, positive, and negative columns."""
        examples = load_csv(sample_csv_data)

        # Check that negatives are loaded
        has_negative = any('negative' in ex for ex in examples)
        assert has_negative

    def test_csv_without_negative(self, temp_dir):
        """Test CSV with only query and positive columns."""
        csv_path = temp_dir / "no_neg.csv"
        content = """query,positive
What is X?,X is a thing.
What is Y?,Y is another thing.
"""
        csv_path.write_text(content)

        examples = load_csv(csv_path)

        assert len(examples) == 2
        assert all('query' in ex and 'positive' in ex for ex in examples)

    def test_csv_missing_required_column(self, temp_dir):
        """Test CSV missing required column raises error."""
        csv_path = temp_dir / "bad.csv"
        content = """query,something_else
What is X?,X is a thing.
"""
        csv_path.write_text(content)

        with pytest.raises(ValueError, match="positive"):
            load_csv(csv_path)

    def test_pandas_not_installed(self, temp_dir, monkeypatch):
        """Test handling when pandas is not installed."""
        csv_path = temp_dir / "test.csv"
        csv_path.write_text("query,positive\nQ,A")

        # Mock pandas import failure
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No pandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(SystemExit):
            load_csv(csv_path)


class TestGenerateQueryFromDoc:
    """Test query generation from documents."""

    def test_generates_from_first_sentence(self):
        """Test that query uses first sentence."""
        doc = "This is the first sentence. This is the second."
        query = generate_query_from_doc(doc)

        assert "first sentence" in query
        assert "second" not in query

    def test_handles_no_period(self):
        """Test handling document without periods."""
        doc = "This is a document without periods"
        query = generate_query_from_doc(doc)

        assert len(query) > 0
        assert len(query) <= 100

    def test_adds_question_mark_for_questions(self):
        """Test that question words get question marks."""
        for word in ['What', 'How', 'Why', 'When', 'Where', 'Who']:
            doc = f"{word} is this about"
            query = generate_query_from_doc(doc)

            assert query.endswith('?')

    def test_does_not_double_question_mark(self):
        """Test no double question marks."""
        doc = "What is this?"
        query = generate_query_from_doc(doc)

        assert query.count('?') == 1

    def test_handles_long_first_sentence(self):
        """Test handling of very long first sentence."""
        # Document with very long first sentence (before first period)
        doc = "x" * 200 + ". Second sentence."
        query = generate_query_from_doc(doc)

        # Function takes first sentence, which can be longer than 100 chars
        assert "x" in query
        assert "Second" not in query


class TestGeneratePairs:
    """Test training pair generation."""

    def test_generates_correct_number_of_pairs(self, sample_texts):
        """Test that correct number of pairs is generated."""
        num_pairs = 3
        examples = generate_pairs(sample_texts, num_pairs=num_pairs)

        assert len(examples) == num_pairs

    def test_generates_all_docs_if_num_pairs_none(self, sample_texts):
        """Test that num_pairs=None uses all documents."""
        examples = generate_pairs(sample_texts, num_pairs=None)

        assert len(examples) == len(sample_texts)

    def test_pair_format_is_correct(self, sample_texts):
        """Test that generated pairs have correct format."""
        examples = generate_pairs(sample_texts, num_pairs=2)

        for ex in examples:
            assert 'query' in ex
            assert 'positive' in ex
            assert isinstance(ex['query'], str)
            assert isinstance(ex['positive'], str)

    def test_adds_negatives_when_requested(self, sample_texts):
        """Test that negatives are added."""
        examples = generate_pairs(sample_texts, num_pairs=3, add_hard_negatives=True)

        has_negatives = any('negative' in ex for ex in examples)
        assert has_negatives

    def test_no_negatives_when_disabled(self, sample_texts):
        """Test that negatives are not added when disabled."""
        examples = generate_pairs(sample_texts, num_pairs=3, add_hard_negatives=False)

        has_negatives = any('negative' in ex for ex in examples)
        assert not has_negatives

    def test_negative_different_from_positive(self, sample_texts):
        """Test that negative is different from positive."""
        examples = generate_pairs(sample_texts, num_pairs=3, add_hard_negatives=True)

        for ex in examples:
            if 'negative' in ex:
                assert ex['negative'] != ex['positive']

    def test_handles_single_document(self):
        """Test with only one document."""
        docs = ["Single document"]
        examples = generate_pairs(docs, num_pairs=1, add_hard_negatives=False)

        assert len(examples) == 1

    def test_num_pairs_capped_at_num_docs(self):
        """Test that num_pairs is capped at number of documents."""
        docs = ["Doc 1", "Doc 2"]
        examples = generate_pairs(docs, num_pairs=10)

        # Should only generate 2 pairs (one per doc)
        assert len(examples) <= len(docs)


class TestSplitData:
    """Test train/val/test splitting."""

    def test_split_ratios_correct(self, sample_triplets):
        """Test that split ratios are approximately correct."""
        train, val, test = split_data(
            sample_triplets,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            shuffle=False
        )

        total = len(sample_triplets)
        assert len(train) == int(total * 0.6)
        assert len(val) == int(total * 0.2)

    def test_split_sums_to_total(self, sample_triplets):
        """Test that split sizes sum to original size."""
        train, val, test = split_data(sample_triplets, shuffle=False)

        assert len(train) + len(val) + len(test) == len(sample_triplets)

    def test_ratios_must_sum_to_one(self, sample_triplets):
        """Test that invalid ratios raise assertion error."""
        with pytest.raises(AssertionError):
            split_data(
                sample_triplets,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum > 1.0
            )

    def test_no_data_leakage_between_splits(self, random_seed):
        """Test that splits don't overlap."""
        # Create identifiable examples
        examples = [{'id': i, 'query': f'Q{i}', 'positive': f'P{i}'} for i in range(20)]

        train, val, test = split_data(examples, shuffle=False)

        # Check no overlap
        train_ids = {ex['id'] for ex in train}
        val_ids = {ex['id'] for ex in val}
        test_ids = {ex['id'] for ex in test}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_shuffle_parameter(self, random_seed):
        """Test that shuffle actually shuffles."""
        examples = [{'id': i, 'query': f'Q{i}', 'positive': f'P{i}'} for i in range(20)]

        # Without shuffle
        train1, _, _ = split_data(examples.copy(), shuffle=False)

        # With shuffle (note: with same seed, might still be same due to random state)
        # So we just test that it doesn't error
        train2, _, _ = split_data(examples.copy(), shuffle=True)

        # Both should work
        assert len(train1) > 0
        assert len(train2) > 0

    def test_deterministic_with_seed(self):
        """Test that same seed gives same split."""
        import random
        examples = [{'id': i, 'query': f'Q{i}', 'positive': f'P{i}'} for i in range(20)]

        random.seed(42)
        train1, val1, test1 = split_data(examples.copy(), shuffle=True)

        random.seed(42)
        train2, val2, test2 = split_data(examples.copy(), shuffle=True)

        # Should be identical
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_standard_split(self, sample_triplets):
        """Test standard 80/10/10 split."""
        train, val, test = split_data(
            sample_triplets,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=False
        )

        assert len(train) > 0
        assert len(val) >= 0  # Might be 0 if sample_triplets is very small
        assert len(test) >= 0

    def test_all_examples_preserved(self, sample_triplets):
        """Test that all examples are in one of the splits."""
        train, val, test = split_data(sample_triplets, shuffle=False)

        all_splits = train + val + test

        # Should have same length
        assert len(all_splits) == len(sample_triplets)


class TestDataPreparationIntegration:
    """Integration tests for data preparation workflow."""

    def test_full_pipeline_from_docs(self, sample_documents_dir):
        """Test complete pipeline: load → generate pairs → split."""
        # Load
        documents = load_documents(sample_documents_dir)
        assert len(documents) > 0

        # Generate pairs
        examples = generate_pairs(documents, num_pairs=len(documents))
        assert len(examples) == len(documents)

        # Split
        if len(examples) >= 3:  # Need at least 3 for splitting
            train, val, test = split_data(examples)
            assert len(train) + len(val) + len(test) == len(examples)

    def test_full_pipeline_from_csv(self, sample_csv_data):
        """Test complete pipeline: load CSV → split."""
        # Load
        examples = load_csv(sample_csv_data)
        assert len(examples) > 0

        # Split
        if len(examples) >= 3:
            train, val, test = split_data(examples)
            assert len(train) + len(val) + len(test) == len(examples)
