#!/usr/bin/env python3
"""
Data preparation script for training IsotropicGaussianEncoder.

This script helps prepare training data in the correct format for the training script.
Supports various input formats and generates contrastive pairs.

Usage:
    # From raw text files
    python scripts/prepare_data.py --input_dir data/raw --output data/processed/train.json

    # From CSV with columns: query, positive, negative
    python scripts/prepare_data.py --csv_file data/pairs.csv --output data/processed/train.json

    # Generate synthetic pairs from documents
    python scripts/prepare_data.py --documents data/docs.txt --generate_pairs --output data/processed/train.json

    # Split into train/val/test
    python scripts/prepare_data.py --input data/all_pairs.json --split 0.8 0.1 0.1
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(input_dir: Path) -> List[str]:
    """
    Load documents from a directory of text files.

    Args:
        input_dir: Directory containing text files

    Returns:
        List of document strings
    """
    documents = []

    for file_path in input_dir.glob('**/*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    logger.info(f"Loaded {len(documents)} documents from {input_dir}")
    return documents


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load training pairs from CSV file.

    Expected columns: query, positive, negative (optional)

    Args:
        csv_path: Path to CSV file

    Returns:
        List of training examples
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for CSV loading. Install with: pip install pandas")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = ['query', 'positive']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must have '{col}' column")

    examples = []
    for _, row in df.iterrows():
        example = {
            'query': str(row['query']),
            'positive': str(row['positive']),
        }

        if 'negative' in df.columns and pd.notna(row['negative']):
            example['negative'] = str(row['negative'])

        examples.append(example)

    logger.info(f"Loaded {len(examples)} examples from {csv_path}")
    return examples


def generate_query_from_doc(doc: str) -> str:
    """
    Generate a simple query from a document.

    This is a basic heuristic - for better results, use a model or manual curation.

    Args:
        doc: Document text

    Returns:
        Query string
    """
    # Take first sentence or first 100 chars
    sentences = doc.split('.')
    if sentences:
        query = sentences[0].strip()
    else:
        query = doc[:100].strip()

    # Add question marker if appropriate
    if not query.endswith('?'):
        # Simple heuristic: if starts with what/how/why/when/where, add ?
        if any(query.lower().startswith(w) for w in ['what', 'how', 'why', 'when', 'where', 'who']):
            query += '?'

    return query


def generate_pairs(
    documents: List[str],
    num_pairs: Optional[int] = None,
    add_hard_negatives: bool = True
) -> List[Dict[str, str]]:
    """
    Generate training pairs from documents.

    Strategy:
    - Query: Generated from document (or use document itself)
    - Positive: The source document
    - Negative: Random document (optional)

    Args:
        documents: List of documents
        num_pairs: Number of pairs to generate (None = use all docs)
        add_hard_negatives: Whether to add negative examples

    Returns:
        List of training examples
    """
    if num_pairs is None:
        num_pairs = len(documents)

    num_pairs = min(num_pairs, len(documents))

    examples = []
    used_indices = set()

    logger.info(f"Generating {num_pairs} training pairs...")

    for _ in range(num_pairs):
        # Select a random document as positive
        pos_idx = random.randint(0, len(documents) - 1)
        while pos_idx in used_indices and len(used_indices) < len(documents):
            pos_idx = random.randint(0, len(documents) - 1)

        used_indices.add(pos_idx)
        positive = documents[pos_idx]

        # Generate query (you might want to use a better method)
        query = generate_query_from_doc(positive)

        example = {
            'query': query,
            'positive': positive,
        }

        # Add hard negative
        if add_hard_negatives and len(documents) > 1:
            neg_idx = random.randint(0, len(documents) - 1)
            while neg_idx == pos_idx:
                neg_idx = random.randint(0, len(documents) - 1)

            example['negative'] = documents[neg_idx]

        examples.append(example)

    logger.info(f"Generated {len(examples)} training pairs")
    return examples


def split_data(
    examples: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split data into train/val/test sets.

    Args:
        examples: List of training examples
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train, val, test) examples
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    if shuffle:
        examples = examples.copy()
        random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


def create_test_format(
    examples: List[Dict[str, str]],
    output_path: Path
) -> None:
    """
    Create test data in evaluation format.

    Args:
        examples: List of training examples
        output_path: Path to save test data
    """
    # Extract unique corpus
    corpus = []
    corpus_set = set()

    for ex in examples:
        if ex['positive'] not in corpus_set:
            corpus.append(ex['positive'])
            corpus_set.add(ex['positive'])

    # Create query -> relevant doc mapping
    queries = []
    relevance = []

    for ex in examples:
        queries.append(ex['query'])

        # Find index of positive doc in corpus
        pos_idx = corpus.index(ex['positive'])
        relevance.append([pos_idx])

    test_data = {
        'corpus': corpus,
        'queries': queries,
        'relevance': relevance
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved test data to {output_path}")
    logger.info(f"  Corpus: {len(corpus)} documents")
    logger.info(f"  Queries: {len(queries)}")


def validate_examples(examples: List[Dict[str, str]]) -> None:
    """
    Validate training examples format.

    Args:
        examples: List of training examples
    """
    for i, ex in enumerate(examples):
        if 'query' not in ex:
            raise ValueError(f"Example {i} missing 'query' field")
        if 'positive' not in ex:
            raise ValueError(f"Example {i} missing 'positive' field")

        if not ex['query'] or not ex['positive']:
            raise ValueError(f"Example {i} has empty query or positive")

    logger.info(f"✅ Validated {len(examples)} examples")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', type=str,
                             help='Directory containing text files')
    input_group.add_argument('--csv_file', type=str,
                             help='CSV file with query, positive, negative columns')
    input_group.add_argument('--documents', type=str,
                             help='Text file with one document per line')
    input_group.add_argument('--input', type=str,
                             help='JSON file with existing pairs (for splitting only)')

    # Processing options
    parser.add_argument('--generate_pairs', action='store_true',
                        help='Generate synthetic query-doc pairs from documents')
    parser.add_argument('--num_pairs', type=int, default=None,
                        help='Number of pairs to generate (default: all)')
    parser.add_argument('--add_negatives', action='store_true', default=True,
                        help='Add hard negative examples')

    # Splitting options
    parser.add_argument('--split', nargs=3, type=float, metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios for train/val/test (e.g., 0.8 0.1 0.1)')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle before splitting')

    # Output options
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for training data (JSON)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for train/val/test split (overrides --output)')

    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Load or generate examples
    examples = []

    if args.input_dir:
        # Load documents from directory
        documents = load_documents(Path(args.input_dir))

        if args.generate_pairs:
            examples = generate_pairs(
                documents,
                num_pairs=args.num_pairs,
                add_hard_negatives=args.add_negatives
            )
        else:
            logger.error("--generate_pairs required when using --input_dir")
            sys.exit(1)

    elif args.csv_file:
        # Load from CSV
        examples = load_csv(Path(args.csv_file))

    elif args.documents:
        # Load documents from text file
        with open(args.documents, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]

        if args.generate_pairs:
            examples = generate_pairs(
                documents,
                num_pairs=args.num_pairs,
                add_hard_negatives=args.add_negatives
            )
        else:
            logger.error("--generate_pairs required when using --documents")
            sys.exit(1)

    elif args.input:
        # Load existing pairs
        with open(args.input, 'r', encoding='utf-8') as f:
            examples = json.load(f)

    # Validate examples
    validate_examples(examples)

    # Split or save
    if args.split:
        train_ratio, val_ratio, test_ratio = args.split

        train_examples, val_examples, test_examples = split_data(
            examples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            shuffle=not args.no_shuffle
        )

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.output).parent

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save splits
        train_path = output_dir / 'train.json'
        val_path = output_dir / 'val.json'
        test_path = output_dir / 'test.json'

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(train_examples)} training examples to {train_path}")

        if val_examples:
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_examples, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved {len(val_examples)} validation examples to {val_path}")

        if test_examples:
            # Save in evaluation format
            test_eval_path = output_dir / 'test_eval.json'
            create_test_format(test_examples, test_eval_path)

    else:
        # Save all examples
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(examples)} examples to {output_path}")

    logger.info("\n" + "="*60)
    logger.info("Data preparation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
