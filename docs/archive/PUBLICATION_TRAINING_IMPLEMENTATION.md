# Publication Training - Implementation Scripts

This document contains ready-to-run scripts for the training strategies documented in `PUBLICATION_TRAINING_GUIDE.md`.

---

## 🚀 Quick Start: Smart Hybrid (Recommended)

**Total time**: 2-3 days | **Cost**: ~$30 | **Paper quality**: Top conference ⭐⭐⭐⭐⭐

```bash
# Run this complete pipeline
cd /home/ubuntu/ragcun
./scripts/train_for_publication.sh smart_hybrid
```

---

## 📥 Data Download Scripts

### Script 1: Download MS MARCO

Create `scripts/download_msmarco.py`:

```python
#!/usr/bin/env python3
"""
Download MS MARCO dataset for training.
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_msmarco(output_dir: str, max_samples: int = None):
    """Download and format MS MARCO dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading MS MARCO from HuggingFace...")
    print("This may take 10-20 minutes...")
    
    # Load training data
    if max_samples:
        dataset = load_dataset('ms_marco', 'v1.1', split=f'train[:{max_samples}]')
    else:
        dataset = load_dataset('ms_marco', 'v1.1', split='train')
    
    print(f"✅ Loaded {len(dataset)} training examples")
    
    # Load dev data
    dev_dataset = load_dataset('ms_marco', 'v1.1', split='validation')
    print(f"✅ Loaded {len(dev_dataset)} dev examples")
    
    # Format for training
    def format_example(example):
        """Convert MS MARCO format to training format."""
        query = example['query']
        # Use first positive passage
        positive = example['passages']['passage_text'][
            example['passages']['is_selected'].index(1)
        ]
        # Use first negative passage
        negatives = [
            p for p, selected in zip(
                example['passages']['passage_text'],
                example['passages']['is_selected']
            ) if not selected
        ]
        negative = negatives[0] if negatives else positive
        
        return {
            'query': query,
            'positive': positive,
            'negative': negative
        }
    
    # Process training data
    print("\n📝 Formatting training data...")
    train_data = []
    for example in tqdm(dataset):
        try:
            train_data.append(format_example(example))
        except:
            continue
    
    # Process dev data
    print("📝 Formatting dev data...")
    dev_data = []
    for example in tqdm(dev_dataset):
        try:
            dev_data.append(format_example(example))
        except:
            continue
    
    # Save
    train_path = output_path / 'train.json'
    dev_path = output_path / 'dev.json'
    
    print(f"\n💾 Saving to {output_path}/")
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(dev_path, 'w') as f:
        json.dump(dev_data, f, indent=2)
    
    print(f"""
✅ MS MARCO download complete!

Files created:
  - {train_path} ({len(train_data):,} pairs)
  - {dev_path} ({len(dev_data):,} pairs)

Total size: ~{(train_path.stat().st_size + dev_path.stat().st_size) / 1e9:.1f} GB
    """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/processed/msmarco',
                       help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of training samples (for testing)')
    
    args = parser.parse_args()
    download_msmarco(args.output_dir, args.max_samples)
```

Make it executable:
```bash
chmod +x scripts/download_msmarco.py
```

Usage:
```bash
# Download full MS MARCO (500K pairs, ~2GB)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# Or download subset for testing (10K pairs)
python scripts/download_msmarco.py --output_dir data/processed/msmarco_10k --max_samples 10000
```

### Script 2: Download Wikipedia

Create `scripts/download_wiki.py`:

```python
#!/usr/bin/env python3
"""
Download Wikipedia passages for unsupervised pre-training.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_wikipedia(output_file: str, num_passages: int = 100000, max_length: int = 500):
    """Download Wikipedia passages."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {num_passages:,} Wikipedia passages...")
    print("This may take 5-15 minutes...")
    
    # Load Wikipedia with streaming
    wiki = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)
    
    passages = []
    for i, doc in enumerate(tqdm(wiki, total=num_passages)):
        if i >= num_passages:
            break
        
        # Extract text and truncate
        text = doc['text'][:max_length].strip()
        
        # Skip very short passages
        if len(text) < 50:
            continue
        
        passages.append(text)
    
    # Save
    print(f"\n💾 Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(passages))
    
    print(f"""
✅ Wikipedia download complete!

File created: {output_path}
Passages: {len(passages):,}
Size: {output_path.stat().st_size / 1e6:.1f} MB
    """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_passages', type=int, default=100000,
                       help='Number of passages to download')
    parser.add_argument('--max_length', type=int, default=500,
                       help='Max characters per passage')
    parser.add_argument('--output', type=str, default='data/raw/wiki_100k.txt',
                       help='Output file path')
    
    args = parser.parse_args()
    download_wikipedia(args.output, args.num_passages, args.max_length)
```

Make it executable:
```bash
chmod +x scripts/download_wiki.py
```

Usage:
```bash
# Download 100K passages (Quick Prototype)
python scripts/download_wiki.py --num_passages 100000 --output data/raw/wiki_100k.txt

# Download 1M passages (Medium Scale)
python scripts/download_wiki.py --num_passages 1000000 --output data/raw/wiki_1m.txt
```

---

## 🎓 Training Scripts

### Master Training Script

Create `scripts/train_for_publication.sh`:

```bash
#!/bin/bash
# Master script for publication-ready training

set -e

STRATEGY="${1:-smart_hybrid}"

echo "============================================"
echo "Publication Training: $STRATEGY"
echo "============================================"
echo ""

# Load HF token
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

case $STRATEGY in
    quick)
        echo "🚀 Quick Prototype (20 hours)"
        ./scripts/train_quick_prototype.sh
        ;;
    medium)
        echo "🎓 Medium Scale (6-9 days)"
        ./scripts/train_medium_scale.sh
        ;;
    smart_hybrid)
        echo "🔥 Smart Hybrid (2-3 days) - RECOMMENDED"
        ./scripts/train_smart_hybrid.sh
        ;;
    *)
        echo "❌ Unknown strategy: $STRATEGY"
        echo "Usage: $0 [quick|medium|smart_hybrid]"
        exit 1
        ;;
esac
```

### Quick Prototype Training

Create `scripts/train_quick_prototype.sh`:

```bash
#!/bin/bash
# Quick prototype training (~20 hours)

set -e

echo "Step 1: Download Wikipedia (100K passages)"
python scripts/download_wiki.py \
    --num_passages 100000 \
    --output data/raw/wiki_100k.txt

echo ""
echo "Step 2: Generate training pairs"
python scripts/prepare_data.py \
    --documents data/raw/wiki_100k.txt \
    --generate_pairs \
    --num_pairs 100000 \
    --add_negatives \
    --split 0.8 0.1 0.1 \
    --output data/processed/wiki100k/data.json \
    --output_dir data/processed/wiki100k

echo ""
echo "Step 3: Unsupervised pre-training (~6 hours)"
python scripts/train.py \
    --train_data data/processed/wiki100k/train.json \
    --val_data data/processed/wiki100k/val.json \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dim 512 \
    --freeze_early_layers \
    --mixed_precision \
    --output_dir checkpoints/quick_prototype \
    --log_interval 100

echo ""
echo "Step 4: Download MS MARCO subset (100K)"
python scripts/download_msmarco.py \
    --output_dir data/processed/msmarco_100k \
    --max_samples 100000

echo ""
echo "Step 5: Supervised fine-tuning (~8 hours)"
python scripts/train.py \
    --train_data data/processed/msmarco_100k/train.json \
    --val_data data/processed/msmarco_100k/dev.json \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --lambda_isotropy 0.5 \
    --output_dir checkpoints/quick_finetune \
    --load_checkpoint checkpoints/quick_prototype/best_model.pt \
    --mixed_precision

echo ""
echo "✅ Quick prototype training complete!"
echo "Model: checkpoints/quick_finetune/best_model.pt"
echo ""
echo "Next: Evaluate with ./scripts/evaluate_beir.sh checkpoints/quick_finetune/best_model.pt"
```

### Smart Hybrid Training (RECOMMENDED)

Create `scripts/train_smart_hybrid.sh`:

```bash
#!/bin/bash
# Smart Hybrid training (~2-3 days)
# Trains only projection layer with frozen pre-trained base

set -e

echo "============================================"
echo "Smart Hybrid Training"
echo "============================================"
echo ""
echo "Strategy: Train Gaussian projection on frozen pre-trained base"
echo "Base model: sentence-transformers/all-mpnet-base-v2"
echo "Trainable params: ~1M (projection only)"
echo "Training time: ~48 hours on T4"
echo ""

# Check if model supports frozen base
if ! grep -q "freeze_base" ragcun/model.py; then
    echo "⚠️  Model doesn't support frozen base yet."
    echo "Updating model.py..."
    # Model update would go here, or prompt user
    echo "Please update ragcun/model.py with freeze_base support"
    echo "See PUBLICATION_TRAINING_GUIDE.md, Strategy 3, Step 1"
    exit 1
fi

echo "Step 1: Download MS MARCO (full dataset)"
if [ ! -f "data/processed/msmarco/train.json" ]; then
    python scripts/download_msmarco.py --output_dir data/processed/msmarco
else
    echo "✅ MS MARCO already downloaded"
fi

echo ""
echo "Step 2: Train Gaussian projection (~48 hours on T4)"
echo "Starting training at $(date)"
echo ""

python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base \
    --epochs 3 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --warmup_steps 1000 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dim 512 \
    --mixed_precision \
    --output_dir checkpoints/smart_hybrid \
    --save_interval 1 \
    --log_interval 100

echo ""
echo "✅ Smart Hybrid training complete!"
echo "Finished at $(date)"
echo ""
echo "Model: checkpoints/smart_hybrid/best_model.pt"
echo "Trainable params: ~1.2M"
echo ""
echo "Next: Evaluate with ./scripts/evaluate_beir.sh checkpoints/smart_hybrid/best_model.pt"
```

Make all scripts executable:
```bash
chmod +x scripts/train_for_publication.sh
chmod +x scripts/train_quick_prototype.sh
chmod +x scripts/train_smart_hybrid.sh
```

---

## 📊 BEIR Evaluation Scripts

### Script 1: Install BEIR

```bash
# Install BEIR evaluation framework
pip install beir
```

### Script 2: BEIR Evaluation Script

Create `scripts/evaluate_beir.py`:

```python
#!/usr/bin/env python3
"""
Evaluate model on BEIR benchmark.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragcun.model import IsotropicGaussianEncoder
import torch
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All 18 BEIR datasets
BEIR_DATASETS = [
    'msmarco', 'trec-covid', 'nfcorpus', 'nq', 'hotpotqa',
    'fiqa', 'arguana', 'webis-touche2020', 'cqadupstack',
    'quora', 'dbpedia-entity', 'scidocs', 'fever',
    'climate-fever', 'scifact', 'germanquad', 'germandpr'
]

class GaussianRetriever:
    """Wrapper for BEIR evaluation."""
    
    def __init__(self, model_path: str, device='cuda'):
        self.device = device
        logger.info(f"Loading model from {model_path}")
        self.model = IsotropicGaussianEncoder.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def encode_queries(self, queries, batch_size=32, **kwargs):
        """Encode queries for BEIR."""
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                batch_size=batch_size,
                convert_to_numpy=True
            )
        return embeddings
    
    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        """Encode corpus for BEIR."""
        # Corpus is dict with 'text' field
        texts = [doc.get('title', '') + ' ' + doc.get('text', '') for doc in corpus]
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress=True
            )
        return embeddings

def evaluate_beir(model_path: str, datasets=None, output_file=None):
    """Evaluate on BEIR datasets."""
    
    if datasets is None or datasets == ['all']:
        datasets = BEIR_DATASETS
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    retriever = GaussianRetriever(model_path, device)
    
    # Results
    all_results = {}
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {dataset_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Download dataset
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            data_path = util.download_and_unzip(url, "datasets")
            
            # Load data
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
            
            # Encode
            logger.info("Encoding corpus...")
            corpus_embeddings = retriever.encode_corpus(list(corpus.values()))
            
            logger.info("Encoding queries...")
            query_embeddings = retriever.encode_queries(list(queries.values()))
            
            # Compute distances (Euclidean for Gaussian embeddings)
            logger.info("Computing similarities...")
            # Negative Euclidean distance (higher = more similar)
            distances = -np.linalg.norm(
                query_embeddings[:, None, :] - corpus_embeddings[None, :, :],
                axis=2
            )
            
            # Retrieve and evaluate
            retriever_eval = EvaluateRetrieval()
            
            # Convert to BEIR format
            results = {}
            corpus_ids = list(corpus.keys())
            query_ids = list(queries.keys())
            
            for i, qid in enumerate(query_ids):
                scores = distances[i]
                top_indices = np.argsort(scores)[::-1][:1000]  # Top 1000
                results[qid] = {
                    corpus_ids[idx]: float(scores[idx])
                    for idx in top_indices
                }
            
            # Compute metrics
            ndcg, _map, recall, precision = retriever_eval.evaluate(qrels, results, [1, 3, 5, 10, 100, 1000])
            mrr = retriever_eval.evaluate_custom(qrels, results, [10, 100], metric="mrr")
            
            # Store results
            all_results[dataset_name] = {
                'ndcg@10': ndcg['NDCG@10'],
                'map@100': _map['MAP@100'],
                'recall@100': recall['Recall@100'],
                'mrr@10': mrr['MRR@10']
            }
            
            logger.info(f"Results for {dataset_name}:")
            logger.info(f"  NDCG@10:    {ndcg['NDCG@10']:.4f}")
            logger.info(f"  MAP@100:    {_map['MAP@100']:.4f}")
            logger.info(f"  Recall@100: {recall['Recall@100']:.4f}")
            logger.info(f"  MRR@10:     {mrr['MRR@10']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Compute average
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        avg_ndcg = np.mean([r['ndcg@10'] for r in valid_results.values()])
        all_results['average'] = {
            'ndcg@10': avg_ndcg,
            'num_datasets': len(valid_results)
        }
        logger.info(f"\n{'='*60}")
        logger.info(f"Average NDCG@10: {avg_ndcg:.4f} (across {len(valid_results)} datasets)")
        logger.info(f"{'='*60}")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✅ Results saved to {output_path}")
    
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--datasets', nargs='+', default=['all'],
                       help='BEIR datasets to evaluate (default: all)')
    parser.add_argument('--output_file', type=str, default='results/beir_results.json')
    
    args = parser.parse_args()
    evaluate_beir(args.model_path, args.datasets, args.output_file)
```

### Script 3: BEIR Evaluation Wrapper

Create `scripts/evaluate_beir.sh`:

```bash
#!/bin/bash
# Wrapper script for BEIR evaluation

set -e

MODEL_PATH="${1:-checkpoints/smart_hybrid/best_model.pt}"
OUTPUT_FILE="${2:-results/beir_results.json}"

echo "============================================"
echo "BEIR Benchmark Evaluation"
echo "============================================"
echo ""
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

# Evaluate
python scripts/evaluate_beir.py \
    --model_path "$MODEL_PATH" \
    --datasets all \
    --output_file "$OUTPUT_FILE"

echo ""
echo "✅ BEIR evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
```

Make executable:
```bash
chmod +x scripts/evaluate_beir.py
chmod +x scripts/evaluate_beir.sh
```

---

## 📈 Results Analysis Scripts

### Generate Paper Results Table

Create `scripts/generate_paper_results.py`:

```python
#!/usr/bin/env python3
"""
Generate LaTeX tables for paper from BEIR results.
"""

import argparse
import json
from pathlib import Path

def generate_latex_table(results_file, output_file):
    """Generate LaTeX results table."""
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Get average
    avg_ndcg = results.get('average', {}).get('ndcg@10', 0)
    
    # Select representative datasets for table
    datasets = ['msmarco', 'nfcorpus', 'scifact', 'fiqa', 'arguana', 'hotpotqa']
    
    latex = r"""
\begin{table}[t]
\centering
\caption{Results on BEIR benchmark (NDCG@10)}
\label{tab:beir_results}
\begin{tabular}{lcccccc|c}
\toprule
Model & MS MARCO & NFCorpus & SciFact & FiQA & ArguAna & HotpotQA & Avg \\
\midrule
"""
    
    # Add baseline rows (you'll fill these in)
    latex += r"BM25       & 22.8 & 32.5 & 66.5 & 23.6 & 31.5 & 60.3 & 40.6 \\" + "\n"
    latex += r"MPNet-Base & 33.4 & 34.8 & 67.9 & 32.4 & 44.2 & 63.2 & 46.3 \\" + "\n"
    latex += r"Contriever & 35.6 & 32.9 & 69.3 & 31.8 & 46.1 & 65.1 & 46.8 \\" + "\n"
    
    # Add your model
    latex += r"\midrule" + "\n"
    latex += r"\textbf{Ours (Gaussian)} & "
    
    values = []
    for ds in datasets:
        if ds in results and 'ndcg@10' in results[ds]:
            val = results[ds]['ndcg@10'] * 100
            values.append(f"{val:.1f}")
        else:
            values.append("--")
    
    latex += " & ".join(values) + f" & \\textbf{{{avg_ndcg*100:.1f}}} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ LaTeX table saved to {output_path}")
    print("\nCopy this into your paper:")
    print(latex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True,
                       help='BEIR results JSON file')
    parser.add_argument('--output', type=str, default='paper/results_table.tex',
                       help='Output LaTeX file')
    
    args = parser.parse_args()
    generate_latex_table(args.results, args.output)
```

---

## 🚦 Complete Pipeline Example

### Run Everything

```bash
# 1. Setup
cd /home/ubuntu/ragcun
export HF_TOKEN="your_token_here"

# 2. Train (Smart Hybrid - RECOMMENDED)
./scripts/train_for_publication.sh smart_hybrid

# This will:
# - Download MS MARCO (~2 hours)
# - Train projection layer (~48 hours)
# - Save checkpoints every epoch

# 3. Evaluate on BEIR (~3 hours)
./scripts/evaluate_beir.sh \
    checkpoints/smart_hybrid/best_model.pt \
    results/smart_hybrid_beir.json

# 4. Generate paper results
python scripts/generate_paper_results.py \
    --results results/smart_hybrid_beir.json \
    --output paper/results_table.tex

# 5. View results
cat results/smart_hybrid_beir.json
```

### Monitor Training

```bash
# In separate terminal, monitor progress
watch -n 5 nvidia-smi

# View training log
tail -f training.log

# Check checkpoint sizes
du -sh checkpoints/smart_hybrid/
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 4 --gradient_accumulation_steps 8

# Enable gradient checkpointing
--gradient_checkpointing
```

### Slow Training

```bash
# Enable mixed precision (2x speedup)
--mixed_precision

# Check GPU utilization
nvidia-smi dmon -s u
# Should be >80% utilization
```

### Download Errors

```bash
# Set HuggingFace cache
export HF_HOME=/path/to/large/disk/.cache/huggingface

# Use mirror if main site is slow
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📞 Quick Reference

### File Structure After Setup

```
ragcun/
├── data/
│   ├── raw/
│   │   ├── wiki_100k.txt        # Wikipedia passages
│   │   └── wiki_1m.txt          # (optional)
│   └── processed/
│       ├── msmarco/             # MS MARCO training data
│       │   ├── train.json
│       │   └── dev.json
│       └── wiki100k/            # (optional unsupervised)
├── checkpoints/
│   ├── smart_hybrid/            # Your trained model
│   │   ├── best_model.pt
│   │   ├── final_model.pt
│   │   └── train_config.json
│   └── quick_prototype/         # (optional)
├── results/
│   ├── beir_results.json        # BEIR evaluation
│   └── smart_hybrid_beir.json
└── paper/
    └── results_table.tex        # LaTeX table for paper
```

### Key Commands

```bash
# Download data
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# Train
./scripts/train_for_publication.sh smart_hybrid

# Evaluate
./scripts/evaluate_beir.sh checkpoints/smart_hybrid/best_model.pt

# Generate paper results
python scripts/generate_paper_results.py --results results/beir_results.json
```

---

**Ready to start?**

```bash
# Start with Smart Hybrid (recommended)
cd /home/ubuntu/ragcun
./scripts/train_for_publication.sh smart_hybrid
```

**Estimated completion**: 2-3 days  
**Expected BEIR NDCG@10**: 48-50%  
**Paper quality**: Top conference ⭐⭐⭐⭐⭐

