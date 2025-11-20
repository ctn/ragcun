# Data Preparation Guide

## 📥 MS MARCO Dataset

**Required for training**

### Quick Download

```bash
# Download full MS MARCO (500K training pairs)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# Time: ~1 hour
# Size: ~2GB
```

### Output

```
data/processed/msmarco/
├── train.json  (~2GB, 502,939 examples)
└── dev.json    (~50MB, 6,980 examples)
```

### Data Format

```json
[
  {
    "query": "what is machine learning",
    "positive": "Machine learning is a branch of artificial intelligence...",
    "negative": "The weather forecast shows rain tomorrow..."
  },
  ...
]
```

### Options

```bash
# Small subset for testing
python scripts/download_msmarco.py \
    --output_dir data/processed/msmarco_test \
    --max_train_samples 10000

# Custom split ratio
python scripts/download_msmarco.py \
    --output_dir data/processed/msmarco \
    --split_ratio 0.5  # Use 50% of data
```

---

## 🌐 Wikipedia (Optional)

**For unsupervised pre-training**

```bash
# Download Wikipedia passages
python scripts/download_wiki.py \
    --num_passages 100000 \
    --output data/raw/wiki_100k.txt
```

**Note:** MS MARCO alone is sufficient for publication-quality results.

---

## 🧪 Test Data

For quick testing:

```bash
# Create tiny dataset (100 examples)
python << 'EOF'
import json
test_data = [
    {
        "query": "test query",
        "positive": "relevant document",
        "negative": "irrelevant document"
    }
] * 100

with open('data/processed/test_data.json', 'w') as f:
    json.dump(test_data, f)
EOF
```

---

## 📊 BEIR Datasets

**Auto-downloaded during evaluation**

```bash
# BEIR datasets download automatically when needed
python scripts/eval/beir.py \
    --model_path checkpoints/model.pt \
    --datasets scifact nfcorpus  # Auto-downloads these
```

**Cached in:** `~/.beir/datasets/`

---

## 🔧 Custom Data

### Format Requirements

Your data must be JSON with this structure:

```python
[
  {
    "query": str,     # Query text
    "positive": str,  # Relevant document
    "negative": str   # Irrelevant document
  },
  ...
]
```

### Create Custom Dataset

```python
import json

# Your data
custom_data = [
    {
        "query": "Your query",
        "positive": "Relevant document",
        "negative": "Irrelevant document"
    },
    # ... more examples
]

# Save
with open('data/processed/custom/train.json', 'w') as f:
    json.dump(custom_data, f)

# Use in training
python scripts/train/isotropic.py \
    --train_data data/processed/custom/train.json \
    ...
```

---

## 📁 Directory Structure

```
data/
├── raw/                    # Raw downloaded data
│   └── wiki_100k.txt
├── processed/              # Processed training data
│   ├── msmarco/
│   │   ├── train.json
│   │   └── dev.json
│   └── custom/
└── beir/                   # BEIR datasets (auto-cached)
    ├── scifact/
    ├── nfcorpus/
    └── ...
```

---

## ✅ Data Checklist

Before training:

- [ ] MS MARCO downloaded
- [ ] Data files exist: `train.json`, `dev.json`
- [ ] Format verified (query/positive/negative keys)
- [ ] File sizes reasonable (~2GB for full MS MARCO)
- [ ] No corrupted files

Verify:
```bash
# Check files exist
ls -lh data/processed/msmarco/

# Verify format
python -c "
import json
with open('data/processed/msmarco/train.json') as f:
    data = json.load(f)
    print(f'✅ {len(data):,} examples')
    print(f'Keys: {list(data[0].keys())}')
"
```

---

## 🚀 Quick Start

```bash
# 1. Download MS MARCO
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 2. Verify
ls -lh data/processed/msmarco/

# 3. Start training
python scripts/train/isotropic.py --train_data data/processed/msmarco/train.json ...
```

That's it! MS MARCO is all you need.

