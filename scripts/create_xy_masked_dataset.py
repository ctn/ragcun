#!/usr/bin/env python3
"""
Create large X/Y Text/Masked-Text dataset for self-supervised JEPA training.

This script generates simple pairs of (original_text, masked_text) from:
- Documents (from JSON list)
- Queries (from query-document pairs)
- Or both

Output format: [{x: original_text, y: masked_text}, ...]

Usage:
    # From documents only
    python scripts/create_xy_masked_dataset.py \
        --input_documents data/raw/msmarco_documents.json \
        --output data/processed/xy_masked_documents.json \
        --num_variants 5 \
        --min_length 50 \
        --max_length 2000

    # From query-document pairs (extract queries and documents)
    python scripts/create_xy_masked_dataset.py \
        --input_pairs data/processed/msmarco/train.json \
        --output data/processed/xy_masked_pairs.json \
        --use_queries \
        --use_documents \
        --num_variants 3
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mask_text_span(
    text: str,
    mask_ratio: float = 0.2,
    mask_token: str = "[MASK]"
) -> str:
    """Mask a contiguous span of text."""
    words = text.split()
    if len(words) <= 1:
        return text
    
    num_to_mask = max(1, int(len(words) * mask_ratio))
    max_start = len(words) - num_to_mask
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + num_to_mask
    
    masked_words = words[:start_idx] + [mask_token] + words[end_idx:]
    return " ".join(masked_words)


def mask_text_tokens(
    text: str,
    mask_ratio: float = 0.2,
    mask_token: str = "[MASK]"
) -> str:
    """Mask random tokens (words) in text."""
    words = text.split()
    if len(words) <= 1:
        return text
    
    num_to_mask = max(1, int(len(words) * mask_ratio))
    indices_to_mask = random.sample(range(len(words)), num_to_mask)
    
    masked_words = [
        mask_token if i in indices_to_mask else word
        for i, word in enumerate(words)
    ]
    return " ".join(masked_words)


def mask_text_prefix(
    text: str,
    mask_ratio: float = 0.3
) -> str:
    """Mask the prefix (beginning) of text."""
    words = text.split()
    if len(words) <= 1:
        return text
    
    num_to_keep = max(1, int(len(words) * (1 - mask_ratio)))
    return " ".join(words[-num_to_keep:])


def mask_text_suffix(
    text: str,
    mask_ratio: float = 0.3
) -> str:
    """Mask the suffix (end) of text."""
    words = text.split()
    if len(words) <= 1:
        return text
    
    num_to_keep = max(1, int(len(words) * (1 - mask_ratio)))
    return " ".join(words[:num_to_keep])


def create_masked_variant(
    text: str,
    mask_ratio: float = 0.2,
    strategy: str = "mixed"
) -> str:
    """Create a single masked variant of text."""
    if strategy == "span":
        return mask_text_span(text, mask_ratio)
    elif strategy == "tokens":
        return mask_text_tokens(text, mask_ratio)
    elif strategy == "prefix":
        return mask_text_prefix(text, mask_ratio)
    elif strategy == "suffix":
        return mask_text_suffix(text, mask_ratio)
    elif strategy == "mixed":
        strategies = ["span", "tokens", "prefix", "suffix"]
        chosen = random.choice(strategies)
        if chosen == "span":
            return mask_text_span(text, mask_ratio)
        elif chosen == "tokens":
            return mask_text_tokens(text, mask_ratio)
        elif chosen == "prefix":
            return mask_text_prefix(text, mask_ratio)
        else:  # suffix
            return mask_text_suffix(text, mask_ratio)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def process_documents(
    documents: List[str],
    num_variants: int = 3,
    mask_ratio: float = 0.2,
    strategy: str = "mixed",
    min_length: int = 50,
    max_length: int = 2000,
    deduplicate: bool = True
) -> List[Dict[str, str]]:
    """Process documents to create X/Y pairs."""
    pairs = []
    seen_texts: Set[str] = set()
    
    logger.info(f"Processing {len(documents):,} documents...")
    
    for i, doc in enumerate(documents):
        # Filter by length
        if not (min_length <= len(doc) <= max_length):
            continue
        
        # Deduplicate
        if deduplicate:
            if doc in seen_texts:
                continue
            seen_texts.add(doc)
        
        # Create masked variants
        for _ in range(num_variants):
            masked = create_masked_variant(doc, mask_ratio, strategy)
            pairs.append({
                'x': doc,  # Original
                'y': masked  # Masked
            })
        
        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i+1:,}/{len(documents):,} documents, created {len(pairs):,} pairs")
    
    logger.info(f"Created {len(pairs):,} X/Y pairs from {len(documents):,} documents")
    return pairs


def process_pairs(
    pairs_data: List[Dict],
    use_queries: bool = True,
    use_documents: bool = True,
    num_variants: int = 3,
    mask_ratio: float = 0.2,
    strategy: str = "mixed",
    min_length: int = 50,
    max_length: int = 2000,
    deduplicate: bool = True
) -> List[Dict[str, str]]:
    """Process query-document pairs to extract texts and create X/Y pairs."""
    xy_pairs = []
    seen_texts: Set[str] = set()
    
    logger.info(f"Processing {len(pairs_data):,} query-document pairs...")
    
    for i, example in enumerate(pairs_data):
        texts_to_process = []
        
        if use_queries and 'query' in example:
            query = example['query']
            if min_length <= len(query) <= max_length:
                texts_to_process.append(query)
        
        if use_documents:
            if 'positive' in example:
                pos_doc = example['positive']
                if min_length <= len(pos_doc) <= max_length:
                    texts_to_process.append(pos_doc)
            
            if 'negative' in example:
                neg_doc = example['negative']
                if min_length <= len(neg_doc) <= max_length:
                    texts_to_process.append(neg_doc)
        
        # Create X/Y pairs for each text
        for text in texts_to_process:
            # Deduplicate
            if deduplicate:
                if text in seen_texts:
                    continue
                seen_texts.add(text)
            
            # Create masked variants
            for _ in range(num_variants):
                masked = create_masked_variant(text, mask_ratio, strategy)
                xy_pairs.append({
                    'x': text,  # Original
                    'y': masked  # Masked
                })
        
        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i+1:,}/{len(pairs_data):,} examples, created {len(xy_pairs):,} pairs")
    
    logger.info(f"Created {len(xy_pairs):,} X/Y pairs from {len(pairs_data):,} examples")
    return xy_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Create large X/Y Text/Masked-Text dataset for self-supervised training"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_documents',
        type=str,
        help='Input JSON file with list of documents'
    )
    input_group.add_argument(
        '--input_pairs',
        type=str,
        help='Input JSON file with query-document pairs'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path for X/Y pairs'
    )
    
    # Processing options
    parser.add_argument(
        '--num_variants',
        type=int,
        default=3,
        help='Number of masked variants per text (default: 3)'
    )
    parser.add_argument(
        '--mask_ratio',
        type=float,
        default=0.2,
        help='Fraction of text to mask (0.0-1.0, default: 0.2)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='mixed',
        choices=['span', 'tokens', 'prefix', 'suffix', 'mixed'],
        help='Masking strategy (default: mixed)'
    )
    
    # Filtering options
    parser.add_argument(
        '--min_length',
        type=int,
        default=50,
        help='Minimum text length to include (characters, default: 50)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2000,
        help='Maximum text length to include (characters, default: 2000)'
    )
    parser.add_argument(
        '--no_deduplicate',
        action='store_true',
        help='Disable deduplication of texts'
    )
    
    # For input_pairs only
    parser.add_argument(
        '--use_queries',
        action='store_true',
        help='Include queries from pairs (only with --input_pairs)'
    )
    parser.add_argument(
        '--use_documents',
        action='store_true',
        help='Include documents from pairs (only with --input_pairs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Process based on input type
    if args.input_documents:
        logger.info(f"Loading documents from {args.input_documents}")
        with open(args.input_documents, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if not isinstance(documents, list):
            raise ValueError("Input must be a JSON array of document strings")
        
        xy_pairs = process_documents(
            documents,
            num_variants=args.num_variants,
            mask_ratio=args.mask_ratio,
            strategy=args.strategy,
            min_length=args.min_length,
            max_length=args.max_length,
            deduplicate=not args.no_deduplicate
        )
    
    elif args.input_pairs:
        if not args.use_queries and not args.use_documents:
            raise ValueError("Must specify --use_queries and/or --use_documents with --input_pairs")
        
        logger.info(f"Loading pairs from {args.input_pairs}")
        with open(args.input_pairs, 'r', encoding='utf-8') as f:
            pairs_data = json.load(f)
        
        xy_pairs = process_pairs(
            pairs_data,
            use_queries=args.use_queries,
            use_documents=args.use_documents,
            num_variants=args.num_variants,
            mask_ratio=args.mask_ratio,
            strategy=args.strategy,
            min_length=args.min_length,
            max_length=args.max_length,
            deduplicate=not args.no_deduplicate
        )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(xy_pairs):,} X/Y pairs to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(xy_pairs, f, indent=2, ensure_ascii=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Done! File size: {file_size_mb:.1f} MB")
    logger.info(f"   Total pairs: {len(xy_pairs):,}")
    logger.info(f"   Format: [{{x: original_text, y: masked_text}}, ...]")


if __name__ == '__main__':
    main()

