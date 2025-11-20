#!/usr/bin/env python3
"""
Download Wikipedia passages for unsupervised pre-training.

Wikipedia is used for unsupervised contrastive learning before fine-tuning
on MS MARCO.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
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


def clean_text(text: str, max_length: int = 500) -> Optional[str]:
    """
    Clean and truncate Wikipedia text.
    
    Args:
        text: Raw Wikipedia text
        max_length: Maximum characters to keep
        
    Returns:
        Cleaned text or None if too short
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate
    text = text[:max_length].strip()
    
    # Filter very short passages
    if len(text) < 50:
        return None
    
    return text


def download_wikipedia(
    output_file: str,
    num_passages: int = 100000,
    max_length: int = 500,
    language: str = 'en',
    date: str = '20220301'
) -> None:
    """
    Download Wikipedia passages.
    
    Args:
        output_file: Output file path
        num_passages: Number of passages to download
        max_length: Maximum characters per passage
        language: Wikipedia language code
        date: Wikipedia dump date
    """
    if not DEPS_AVAILABLE:
        logger.error("Required packages not installed")
        logger.error("Install: pip install datasets tqdm")
        sys.exit(1)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📥 Downloading {num_passages:,} Wikipedia passages...")
    logger.info(f"Language: {language}, Date: {date}")
    logger.info("This may take 5-30 minutes depending on size...")
    
    # Load Wikipedia with streaming
    try:
        wiki = load_dataset(
            'wikipedia',
            f'{date}.{language}',
            split='train',
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load Wikipedia: {e}")
        logger.error("Available dates: 20220301, 20230901")
        sys.exit(1)
    
    passages = []
    skipped = 0
    
    logger.info("Processing passages...")
    pbar = tqdm(total=num_passages, desc="Downloading")
    
    for doc in wiki:
        if len(passages) >= num_passages:
            break
        
        # Extract text
        try:
            text = doc['text']
        except KeyError:
            skipped += 1
            continue
        
        # Clean and truncate
        cleaned = clean_text(text, max_length)
        
        if cleaned:
            passages.append(cleaned)
            pbar.update(1)
        else:
            skipped += 1
    
    pbar.close()
    
    if len(passages) < num_passages:
        logger.warning(f"Only got {len(passages):,} passages (wanted {num_passages:,})")
    
    # Save
    logger.info(f"\n💾 Saving to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(passages))
    
    # Stats
    size_mb = output_path.stat().st_size / 1e6
    avg_length = sum(len(p) for p in passages) / len(passages)
    
    logger.info(f"""
✅ Wikipedia download complete!

File created: {output_path}
Passages: {len(passages):,}
Size: {size_mb:.1f} MB
Avg length: {avg_length:.0f} characters
Skipped: {skipped:,} (too short or empty)

Next steps:
  1. Generate training pairs:
     python scripts/prepare_data.py \\
       --documents {output_path} \\
       --generate_pairs \\
       --num_pairs {len(passages)} \\
       --output data/processed/wiki/data.json \\
       --output_dir data/processed/wiki

  2. Train unsupervised:
     python scripts/train/isotropic.py \\
       --train_data data/processed/wiki/train.json \\
       --val_data data/processed/wiki/val.json
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia passages for unsupervised training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1K passages, ~0.5 MB, 1 min)
  python scripts/download_wiki.py \\
    --num_passages 1000 \\
    --output data/raw/wiki_1k.txt

  # Small unsupervised (100K passages, ~50 MB, 5 min)
  python scripts/download_wiki.py \\
    --num_passages 100000 \\
    --output data/raw/wiki_100k.txt

  # Medium unsupervised (1M passages, ~500 MB, 30 min)
  python scripts/download_wiki.py \\
    --num_passages 1000000 \\
    --output data/raw/wiki_1m.txt

  # Short passages for faster training
  python scripts/download_wiki.py \\
    --num_passages 100000 \\
    --max_length 300 \\
    --output data/raw/wiki_100k_short.txt
"""
    )
    
    parser.add_argument(
        '--num_passages',
        type=int,
        default=100000,
        help='Number of passages to download (default: 100000)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=500,
        help='Max characters per passage (default: 500)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/wiki_100k.txt',
        help='Output file path (default: data/raw/wiki_100k.txt)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Wikipedia language code (default: en)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default='20220301',
        help='Wikipedia dump date (default: 20220301)'
    )
    
    args = parser.parse_args()
    
    download_wikipedia(
        args.output,
        args.num_passages,
        args.max_length,
        args.language,
        args.date
    )


if __name__ == '__main__':
    main()

