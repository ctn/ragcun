# How to Run Data Preparation

## 🚀 Quick Start (Choose One)

### Option 1: Quick Test Data (50 pairs)
```bash
./scripts/1_prepare_data_quick.sh
```
**Creates**:
- `data/processed/train.json` (35 pairs)
- `data/processed/val.json` (7 pairs)
- `data/processed/test_eval.json` (8 queries)

**Time**: < 1 minute
**Use case**: Testing, learning, debugging

---

### Option 2: Full Production Data (1000 pairs)
```bash
./scripts/4_prepare_data_full.sh
```
**Creates**:
- `data/processed/train.json` (800 pairs)
- `data/processed/val.json` (100 pairs)
- `data/processed/test_eval.json` (100 queries)

**Time**: < 1 minute
**Use case**: Real training, production

---

## 📋 Step-by-Step Examples

### Example 1: Quick Test (Default Settings)
```bash
# Navigate to project
cd /home/ubuntu/ragcun

# Run quick prep
./scripts/1_prepare_data_quick.sh

# Output:
# ============================================
# Quick Data Preparation (50 pairs)
# ============================================
#
# Configuration:
#   Input: data/raw/sample_docs.txt
#   Output directory: data/processed
#   Number of pairs: 50
#
# Preparing data...
# ✅ Quick data preparation complete!
#
# Files created:
#   - data/processed/train.json
#   - data/processed/val.json
#   - data/processed/test_eval.json
```

### Example 2: Full Production Data
```bash
cd /home/ubuntu/ragcun

./scripts/4_prepare_data_full.sh

# Output:
# ============================================
# Full Data Preparation (1000 pairs)
# ============================================
#
# Combining tech and science documents...
# ✅ Combined 61 documents
#
# Generating training pairs from combined documents...
# ✅ Full data preparation complete!
```

---

## 🎛️ Advanced Options

### Custom Number of Pairs

**Quick with 100 pairs:**
```bash
./scripts/1_prepare_data_quick.sh data/raw/sample_docs.txt data/processed 100
```

**Full with 2000 pairs:**
```bash
./scripts/4_prepare_data_full.sh data/processed 2000
```

### Use Different Input Files

**From tech docs only:**
```bash
python scripts/prepare_data.py \
    --documents data/raw/tech_docs.txt \
    --generate_pairs \
    --num_pairs 500 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**From science docs only:**
```bash
python scripts/prepare_data.py \
    --documents data/raw/science_docs.txt \
    --generate_pairs \
    --num_pairs 250 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**From your own documents:**
```bash
# 1. Add your documents
echo "Your document text here" > data/raw/my_docs.txt

# 2. Generate pairs
python scripts/prepare_data.py \
    --documents data/raw/my_docs.txt \
    --generate_pairs \
    --num_pairs 500 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

### Use Pre-made Training Pairs

**From curated pairs:**
```bash
python scripts/prepare_data.py \
    --input data/raw/training_pairs.json \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed
```

### Custom CSV Input

If you have a CSV file with query-document pairs:

```bash
# CSV format: query,positive,negative
python scripts/prepare_data.py \
    --csv_file data/my_pairs.csv \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

---

## 🔍 Verify Your Data

After preparation, check what was created:

```bash
# List created files
ls -lh data/processed/

# Check number of examples
echo "Train pairs:" && cat data/processed/train.json | python -c "import sys,json; print(len(json.load(sys.stdin)))"
echo "Val pairs:" && cat data/processed/val.json | python -c "import sys,json; print(len(json.load(sys.stdin)))"

# View first training example
cat data/processed/train.json | python -m json.tool | head -20
```

---

## 📊 What Gets Created

After running data prep, you'll have:

```
data/processed/
├── train.json          # Training pairs (70-80% of data)
├── val.json           # Validation pairs (10-15% of data)
├── test.json          # Test pairs (training format, 10-15%)
└── test_eval.json     # Test data (evaluation format, for metrics)
```

### File Formats

**Training/Val format** (`train.json`, `val.json`):
```json
[
  {
    "query": "What is Python used for?",
    "positive": "Python is a programming language...",
    "negative": "Java is an object-oriented language..."
  }
]
```

**Evaluation format** (`test_eval.json`):
```json
{
  "corpus": ["doc1", "doc2", "doc3"],
  "queries": ["query1", "query2"],
  "relevance": [[0], [1, 2]]
}
```

---

## 🎯 Common Use Cases

### Use Case 1: Quick Test
```bash
./scripts/1_prepare_data_quick.sh
# Then: ./scripts/2_train_quick.sh
```

### Use Case 2: Production Training
```bash
./scripts/4_prepare_data_full.sh
# Then: ./scripts/5_train_full.sh
```

### Use Case 3: Custom Dataset Size
```bash
./scripts/4_prepare_data_full.sh data/processed 5000
# Creates 5000 pairs from all documents
```

### Use Case 4: Multiple Datasets
```bash
# Tech only
python scripts/prepare_data.py \
    --documents data/raw/tech_docs.txt \
    --generate_pairs --num_pairs 500 \
    --output_dir data/processed/tech

# Science only
python scripts/prepare_data.py \
    --documents data/raw/science_docs.txt \
    --generate_pairs --num_pairs 250 \
    --output_dir data/processed/science

# Combined
./scripts/4_prepare_data_full.sh data/processed/combined 1000
```

---

## 🔧 Python Script Direct Usage

For maximum control, use the Python script directly:

```bash
python scripts/prepare_data.py --help
```

**Full example:**
```bash
python scripts/prepare_data.py \
    --documents data/raw/combined_docs.txt \
    --generate_pairs \
    --num_pairs 1000 \
    --add_negatives \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed \
    --seed 42
```

**Parameters**:
- `--documents PATH` - Text file with one document per line
- `--generate_pairs` - Generate synthetic query-doc pairs
- `--num_pairs N` - Number of pairs to generate
- `--add_negatives` - Include hard negative examples
- `--split TRAIN VAL TEST` - Split ratios (must sum to 1.0)
- `--output_dir PATH` - Where to save files
- `--seed N` - Random seed for reproducibility

---

## ✅ Checklist

Before running data prep:
- [ ] Navigate to project: `cd /home/ubuntu/ragcun`
- [ ] Check data exists: `ls -l data/raw/`
- [ ] Decide: Quick (50) or Full (1000)?

After running data prep:
- [ ] Check files created: `ls -l data/processed/`
- [ ] Verify file sizes are non-zero
- [ ] (Optional) View a sample: `head -20 data/processed/train.json`

---

## 🐛 Troubleshooting

### Error: "Input file not found"
```bash
# Check if data files exist
ls -l data/raw/

# If missing, they should be there already, but you can regenerate
# The sample_docs.txt should exist
ls -l data/raw/sample_docs.txt
```

### Error: "No documents found"
```bash
# Make sure the file has content
wc -l data/raw/sample_docs.txt

# Should show 10 lines or more
```

### Want to start fresh?
```bash
# Remove old processed data
rm -rf data/processed/*

# Run prep again
./scripts/1_prepare_data_quick.sh
```

### Generated pairs seem wrong?
```bash
# View the data
cat data/processed/train.json | python -m json.tool | less

# Regenerate with different seed
python scripts/prepare_data.py \
    --documents data/raw/sample_docs.txt \
    --generate_pairs --num_pairs 50 \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed \
    --seed 123
```

---

## 🎓 Next Steps

After data preparation:

1. **Train a model**:
   ```bash
   ./scripts/2_train_quick.sh   # or
   ./scripts/5_train_full.sh
   ```

2. **Or run complete pipeline**:
   ```bash
   ./scripts/99_pipeline_quick.sh   # includes data prep
   ```

---

## 📚 Related Documentation

- **Python script help**: `python scripts/prepare_data.py --help`
- **Shell script docs**: `scripts/README.md`
- **Data summary**: `DATA_SUMMARY.md`
- **Workflow guide**: `scripts/WORKFLOW.md`

---

## 💡 Pro Tips

1. **Start small**: Use `1_prepare_data_quick.sh` first to test
2. **Check output**: Always verify files were created
3. **Use pipelines**: `99_pipeline_quick.sh` does everything
4. **Save configs**: Document your data prep parameters
5. **Version data**: Keep different datasets in separate directories

Happy data prepping! 🎉
