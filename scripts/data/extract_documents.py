#!/usr/bin/env python3
"""
Extract documents from MS MARCO or other query-document datasets.

This script extracts documents (positive passages) from training data
for self-supervised JEPA training.

Usage:
    # Extract from MS MARCO
    python scripts/data/extract_documents.py \
        --input data/processed/msmarco/train.json \
        --output data/raw/msmarco_documents.json

    # Extract with deduplication
    python scripts/data/extract_documents.py \
        --input data/processed/msmarco/train.json \
        --output data/raw/msmarco_documents.json \
        --deduplicate

    # Extract and filter by length
    python scripts/data/extract_documents.py \
        --input data/processed/msmarco/train.json \
        --output data/raw/msmarco_documents.json \
        --min_length 50 \
        --max_length 2000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Set
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hash_document(text: str) -> str:
    """Generate hash for document deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def extract_documents(
    input_path: str,
    output_path: str,
    deduplicate: bool = False,
    min_length: int = 0,
    max_length: int = None,
    field_name: str = 'positive'
) -> None:
    """
    Extract documents from MS MARCO or similar format.
    
    Args:
        input_path: Path to input JSON file with query-doc pairs
        output_path: Path to output JSON file (array of documents)
        deduplicate: Whether to remove duplicate documents
        min_length: Minimum document length (characters)
        max_length: Maximum document length (characters, None = no limit)
        field_name: Field name containing documents (default: 'positive')
    """
    logger.info(f"Loading data from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data):,} examples")
    
    # Extract documents
    documents = []
    seen_hashes: Set[str] = set()
    skipped_short = 0
    skipped_long = 0
    skipped_duplicate = 0
    
    for i, example in enumerate(data):
        # Extract document
        if field_name not in example:
            logger.warning(f"Example {i} missing '{field_name}' field, skipping")
            continue
        
        doc = example[field_name]
        
        # Filter by length
        doc_length = len(doc)
        if doc_length < min_length:
            skipped_short += 1
            continue
        
        if max_length and doc_length > max_length:
            # Truncate if too long
            doc = doc[:max_length]
            skipped_long += 1
        
        # Deduplicate if requested
        if deduplicate:
            doc_hash = hash_document(doc)
            if doc_hash in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(doc_hash)
        
        documents.append(doc)
        
        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1:,}/{len(data):,} examples")
    
    logger.info(f"\nExtraction complete:")
    logger.info(f"  Total documents: {len(documents):,}")
    if skipped_short > 0:
        logger.info(f"  Skipped (too short): {skipped_short:,}")
    if skipped_long > 0:
        logger.info(f"  Truncated (too long): {skipped_long:,}")
    if skipped_duplicate > 0:
        logger.info(f"  Skipped (duplicates): {skipped_duplicate:,}")
    
    # Save output
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving to {output_path}")
    
    # Save as JSON array
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    # Calculate stats
    file_size_mb = output_path_obj.stat().st_size / 1e6
    avg_length = sum(len(d) for d in documents) / len(documents) if documents else 0
    
    logger.info(f"""
✅ Document extraction complete!

Output file: {output_path}
Documents: {len(documents):,}
File size: {file_size_mb:.1f} MB
Avg document length: {avg_length:.0f} characters

Next steps:
  1. Use for self-supervised training:
     python scripts/train_self_supervised.py \\
       --input_documents {output_path} \\
       --document_split_strategy half \\
       --batch_size 32 \\
       --epochs 2

  2. Or combine with other sources:
     python scripts/combine_documents.py \\
       --inputs {output_path} data/raw/wiki_100k.txt \\
       --output data/raw/combined_documents.json
""")


def main():
    parser = argparse.ArgumentParser(
        description="Extract documents from MS MARCO or similar datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python scripts/data/extract_documents.py \\
    --input data/processed/msmarco/train.json \\
    --output data/raw/msmarco_documents.json

  # With deduplication
  python scripts/data/extract_documents.py \\
    --input data/processed/msmarco/train.json \\
    --output data/raw/msmarco_documents.json \\
    --deduplicate

  # Filter by length
  python scripts/data/extract_documents.py \\
    --input data/processed/msmarco/train.json \\
    --output data/raw/msmarco_documents.json \\
    --min_length 100 \\
    --max_length 1500

  # Extract from different field
  python scripts/data/extract_documents.py \\
    --input data/processed/custom.json \\
    --output data/raw/custom_documents.json \\
    --field_name document
"""
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with query-document pairs'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file (array of documents)'
    )
    
    parser.add_argument(
        '--deduplicate',
        action='store_true',
        help='Remove duplicate documents (by content hash)'
    )
    
    parser.add_argument(
        '--min_length',
        type=int,
        default=0,
        help='Minimum document length in characters (default: 0)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Maximum document length in characters (default: no limit, truncate if exceeded)'
    )
    
    parser.add_argument(
        '--field_name',
        type=str,
        default='positive',
        help='Field name containing documents (default: positive)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {args.input}")
        return
    
    extract_documents(
        input_path=str(input_path),
        output_path=args.output,
        deduplicate=args.deduplicate,
        min_length=args.min_length,
        max_length=args.max_length,
        field_name=args.field_name
    )


if __name__ == '__main__':
    main()


