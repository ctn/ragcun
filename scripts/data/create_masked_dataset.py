#! /usr/bin/env python3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
"""
Create masked datasets for JEPA training.

This script creates masked variants of queries to:
1. Augment training data (more examples from same query-doc pairs)
2. Improve robustness (handle incomplete/partial queries)
3. Better generalization for JEPA predictor

Usage:
    python scripts/create_masked_dataset.py \
        --input data/processed/msmarco_smoke/train.json \
        --output data/processed/msmarco_smoke_masked/train.json \
        --mask_ratio 0.2 \
        --num_variants 3
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mask_query_span(
    query: str,
    mask_ratio: float = 0.2,
    mask_token: str = "[MASK]"
) -> str:
    """
    Mask a contiguous span of the query.
    
    Args:
        query: Original query text
        mask_ratio: Fraction of query to mask (0.0-1.0)
        mask_token: Token to use for masking
        
    Returns:
        Masked query string
    """
    words = query.split()
    if len(words) <= 1:
        return query  # Can't mask single word
    
    # Calculate number of words to mask
    num_to_mask = max(1, int(len(words) * mask_ratio))
    
    # Random start position
    max_start = len(words) - num_to_mask
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + num_to_mask
    
    # Create masked version
    masked_words = words[:start_idx] + [mask_token] + words[end_idx:]
    return " ".join(masked_words)


def mask_query_tokens(
    query: str,
    mask_ratio: float = 0.2,
    mask_token: str = "[MASK]"
) -> str:
    """
    Mask random tokens (words) in the query.
    
    Args:
        query: Original query text
        mask_ratio: Fraction of tokens to mask
        mask_token: Token to use for masking
        
    Returns:
        Masked query string
    """
    words = query.split()
    if len(words) <= 1:
        return query
    
    num_to_mask = max(1, int(len(words) * mask_ratio))
    
    # Randomly select indices to mask
    indices_to_mask = random.sample(range(len(words)), num_to_mask)
    
    # Create masked version
    masked_words = [
        mask_token if i in indices_to_mask else word
        for i, word in enumerate(words)
    ]
    return " ".join(masked_words)


def mask_query_prefix(
    query: str,
    mask_ratio: float = 0.3
) -> str:
    """
    Mask the prefix (beginning) of the query.
    
    Useful for simulating queries where user hasn't finished typing.
    
    Args:
        query: Original query text
        mask_ratio: Fraction to mask from beginning
        
    Returns:
        Masked query string
    """
    words = query.split()
    if len(words) <= 1:
        return query
    
    num_to_keep = max(1, int(len(words) * (1 - mask_ratio)))
    return " ".join(words[-num_to_keep:])


def mask_query_suffix(
    query: str,
    mask_ratio: float = 0.3
) -> str:
    """
    Mask the suffix (end) of the query.
    
    Args:
        query: Original query text
        mask_ratio: Fraction to mask from end
        
    Returns:
        Masked query string
    """
    words = query.split()
    if len(words) <= 1:
        return query
    
    num_to_keep = max(1, int(len(words) * (1 - mask_ratio)))
    return " ".join(words[:num_to_keep])


def create_masked_variants(
    query: str,
    num_variants: int = 3,
    mask_ratio: float = 0.2,
    strategy: str = "mixed"
) -> List[str]:
    """
    Create multiple masked variants of a query.
    
    Args:
        query: Original query
        num_variants: Number of masked variants to create
        mask_ratio: Fraction to mask
        strategy: "span", "tokens", "prefix", "suffix", or "mixed"
        
    Returns:
        List of masked query strings (includes original)
    """
    variants = [query]  # Always include original
    
    for _ in range(num_variants):
        if strategy == "span":
            masked = mask_query_span(query, mask_ratio)
        elif strategy == "tokens":
            masked = mask_query_tokens(query, mask_ratio)
        elif strategy == "prefix":
            masked = mask_query_prefix(query, mask_ratio)
        elif strategy == "suffix":
            masked = mask_query_suffix(query, mask_ratio)
        elif strategy == "mixed":
            # Randomly choose strategy
            strategies = ["span", "tokens", "prefix", "suffix"]
            chosen = random.choice(strategies)
            if chosen == "span":
                masked = mask_query_span(query, mask_ratio)
            elif chosen == "tokens":
                masked = mask_query_tokens(query, mask_ratio)
            elif chosen == "prefix":
                masked = mask_query_prefix(query, mask_ratio)
            else:  # suffix
                masked = mask_query_suffix(query, mask_ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        variants.append(masked)
    
    return variants


def process_dataset(
    input_path: str,
    output_path: str,
    mask_ratio: float = 0.2,
    num_variants: int = 3,
    strategy: str = "mixed",
    keep_original: bool = True,
    seed: int = 42
) -> None:
    """
    Process dataset to create masked variants.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        mask_ratio: Fraction of query to mask
        num_variants: Number of masked variants per query
        strategy: Masking strategy
        keep_original: Whether to keep original query-doc pairs
        seed: Random seed
    """
    random.seed(seed)
    
    logger.info(f"Loading dataset from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Create masked variants
    augmented_data = []
    
    for i, example in enumerate(data):
        query = example['query']
        positive = example['positive']
        negative = example.get('negative', None)
        
        # Create masked variants
        if keep_original:
            variants = create_masked_variants(
                query, num_variants, mask_ratio, strategy
            )
        else:
            variants = create_masked_variants(
                query, num_variants, mask_ratio, strategy
            )[1:]  # Skip original
        
        # Create examples for each variant
        for variant_query in variants:
            new_example = {
                'query': variant_query,
                'positive': positive,
            }
            if negative is not None:
                new_example['negative'] = negative
            
            augmented_data.append(new_example)
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} examples")
    
    logger.info(f"Created {len(augmented_data)} examples (from {len(data)} original)")
    logger.info(f"Augmentation factor: {len(augmented_data) / len(data):.2f}x")
    
    # Save output
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Create masked datasets for JEPA training"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file with query-positive pairs'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--mask_ratio',
        type=float,
        default=0.2,
        help='Fraction of query to mask (0.0-1.0, default: 0.2)'
    )
    parser.add_argument(
        '--num_variants',
        type=int,
        default=3,
        help='Number of masked variants per query (default: 3)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='mixed',
        choices=['span', 'tokens', 'prefix', 'suffix', 'mixed'],
        help='Masking strategy (default: mixed)'
    )
    parser.add_argument(
        '--no_original',
        action='store_true',
        help='Exclude original queries (only masked variants)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_path=args.input,
        output_path=args.output,
        mask_ratio=args.mask_ratio,
        num_variants=args.num_variants,
        strategy=args.strategy,
        keep_original=not args.no_original,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

