#!/usr/bin/env python3
"""
Download MS MARCO dataset for training.

MS MARCO is the standard benchmark for passage ranking with 500K+ training pairs.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    from datasets import load_dataset
    from tqdm import tqdm
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    # Don't exit here - allow importing for testing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_msmarco_example(example: Dict) -> Optional[Dict[str, str]]:
    """
    Convert MS MARCO format to training format.
    
    Args:
        example: MS MARCO example with query and passages
        
    Returns:
        Dictionary with query, positive, and negative passages
        None if example cannot be formatted
    """
    try:
        query = example['query']
        passages = example['passages']
        
        # Find positive passage
        is_selected = passages['is_selected']
        passage_texts = passages['passage_text']
        
        if 1 not in is_selected:
            return None
            
        positive_idx = is_selected.index(1)
        positive = passage_texts[positive_idx]
        
        # Find negative passage (first non-selected)
        negatives = [
            text for text, selected in zip(passage_texts, is_selected)
            if not selected
        ]
        
        if not negatives:
            # If no explicit negative, use a different passage
            negative = passage_texts[1 if positive_idx == 0 else 0]
        else:
            negative = negatives[0]
        
        return {
            'query': query,
            'positive': positive,
            'negative': negative
        }
    except (KeyError, IndexError, TypeError) as e:
        logger.debug(f"Error formatting example: {e}")
        return None


def download_msmarco(
    output_dir: str,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
    split_ratio: float = 1.0
) -> None:
    """
    Download and format MS MARCO dataset.
    
    Args:
        output_dir: Directory to save processed data
        max_train_samples: Limit training samples (None = all)
        max_dev_samples: Limit dev samples (None = all)
        split_ratio: Fraction of training data to use (0.0-1.0)
    """
    if not DEPS_AVAILABLE:
        logger.error("Required packages not installed")
        logger.error("Install: pip install datasets tqdm")
        sys.exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("📥 Downloading MS MARCO from HuggingFace...")
    logger.info("This may take 10-30 minutes depending on your connection...")
    
    # Load training data
    try:
        if max_train_samples:
            logger.info(f"Loading {max_train_samples:,} training samples...")
            dataset = load_dataset(
                'ms_marco',
                'v1.1',
                split=f'train[:{max_train_samples}]',
                trust_remote_code=True
            )
        else:
            logger.info("Loading full training set...")
            dataset = load_dataset(
                'ms_marco',
                'v1.1',
                split='train',
                trust_remote_code=True
            )
        
        logger.info(f"✅ Loaded {len(dataset):,} training examples")
        
        # Load dev data
        if max_dev_samples:
            dev_dataset = load_dataset(
                'ms_marco',
                'v1.1',
                split=f'validation[:{max_dev_samples}]',
                trust_remote_code=True
            )
        else:
            dev_dataset = load_dataset(
                'ms_marco',
                'v1.1',
                split='validation',
                trust_remote_code=True
            )
        
        logger.info(f"✅ Loaded {len(dev_dataset):,} dev examples")
        
    except Exception as e:
        logger.error(f"Failed to download MS MARCO: {e}")
        logger.error("Make sure you have internet connection and datasets library installed")
        sys.exit(1)
    
    # Format training data
    logger.info("\n📝 Formatting training data...")
    train_data = []
    skipped = 0
    
    for example in tqdm(dataset, desc="Processing training"):
        formatted = format_msmarco_example(example)
        if formatted:
            train_data.append(formatted)
        else:
            skipped += 1
    
    if split_ratio < 1.0:
        original_size = len(train_data)
        train_data = train_data[:int(len(train_data) * split_ratio)]
        logger.info(f"Using {len(train_data):,} / {original_size:,} training examples (ratio: {split_ratio})")
    
    logger.info(f"Formatted {len(train_data):,} training examples (skipped {skipped:,})")
    
    # Format dev data
    logger.info("📝 Formatting dev data...")
    dev_data = []
    skipped = 0
    
    for example in tqdm(dev_dataset, desc="Processing dev"):
        formatted = format_msmarco_example(example)
        if formatted:
            dev_data.append(formatted)
        else:
            skipped += 1
    
    logger.info(f"Formatted {len(dev_data):,} dev examples (skipped {skipped:,})")
    
    # Save
    train_path = output_path / 'train.json'
    dev_path = output_path / 'dev.json'
    
    logger.info(f"\n💾 Saving to {output_path}/")
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(dev_path, 'w') as f:
        json.dump(dev_data, f, indent=2)
    
    # Calculate sizes
    train_size_mb = train_path.stat().st_size / 1e6
    dev_size_mb = dev_path.stat().st_size / 1e6
    
    logger.info(f"""
✅ MS MARCO download complete!

Files created:
  - {train_path} ({len(train_data):,} pairs, {train_size_mb:.1f} MB)
  - {dev_path} ({len(dev_data):,} pairs, {dev_size_mb:.1f} MB)

Total size: {train_size_mb + dev_size_mb:.1f} MB

Next steps:
  1. Train: python scripts/train.py --train_data {train_path} --val_data {dev_path}
  2. Or use wrapper: ./scripts/train_smart_hybrid.sh
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download MS MARCO dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download full dataset (500K+ pairs, ~2GB)
  python scripts/download_msmarco.py --output_dir data/processed/msmarco

  # Download subset for testing (10K pairs)
  python scripts/download_msmarco.py \\
    --output_dir data/processed/msmarco_10k \\
    --max_train_samples 10000

  # Download 100K for faster training
  python scripts/download_msmarco.py \\
    --output_dir data/processed/msmarco_100k \\
    --max_train_samples 100000

  # Use 50% of training data
  python scripts/download_msmarco.py \\
    --output_dir data/processed/msmarco \\
    --split_ratio 0.5
"""
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/msmarco',
        help='Output directory (default: data/processed/msmarco)'
    )
    
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help='Limit number of training samples (default: all ~502K)'
    )
    
    parser.add_argument(
        '--max_dev_samples',
        type=int,
        default=None,
        help='Limit number of dev samples (default: all ~6.9K)'
    )
    
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=1.0,
        help='Fraction of training data to use, 0.0-1.0 (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not 0.0 < args.split_ratio <= 1.0:
        parser.error("split_ratio must be between 0.0 and 1.0")
    
    download_msmarco(
        args.output_dir,
        args.max_train_samples,
        args.max_dev_samples,
        args.split_ratio
    )


if __name__ == '__main__':
    main()

