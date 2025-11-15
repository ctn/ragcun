# RAGCUN Directory Structure Guide

This guide explains the purpose and contents of each directory in the RAGCUN project.

## 📁 Project Layout

```
ragcun/
├── ragcun/              # Main source code (Python package)
├── data/                # All data files (raw, processed, embeddings)
├── config/              # Configuration files
├── examples/            # Example scripts
├── notebooks/           # Jupyter notebooks (Colab-ready)
├── external/            # External dependencies
├── tests/               # Unit tests (empty, ready for tests)
├── checkpoints/         # Model checkpoints (created during training)
└── results/             # Evaluation results (created during eval)
```

---

## 📦 Source Code

### `ragcun/` - Main Package

**Purpose**: Core Python package with model and retrieval implementations

**Contents**:
- `__init__.py` - Package initialization, exports main classes
- `model.py` - `GaussianEmbeddingGemma` model class
- `retriever.py` - `GaussianRetriever` for document retrieval

**Key Classes**:
1. **GaussianEmbeddingGemma**:
   - Wraps Google's EmbeddingGemma-300M
   - Projects to isotropic Gaussian space
   - Uses LeJEPA SIGReg loss
   - Outputs unnormalized embeddings

2. **GaussianRetriever**:
   - Document retrieval with Euclidean distance
   - FAISS indexing for speed
   - Saves/loads document indices

**Import from**:
```python
from ragcun import GaussianEmbeddingGemma, GaussianRetriever
```

---

## 💾 Data Directories

### `data/` - All Data Files

**Purpose**: Central location for all data (input, processed, outputs)

**Structure**:
```
data/
├── raw/              # Raw input documents
├── processed/        # Processed training data
└── embeddings/       # Saved embeddings and models
```

#### `data/raw/` - Raw Input Data

**Purpose**: Store original, unprocessed documents

**Contents**:
- `sample_docs.txt` - Sample documents (10 tech topics)
- `README.md` - Usage instructions
- Your documents (PDFs, TXT, DOCX, etc.)

**Usage**:
```bash
# Add your documents here
cp mydocs.txt data/raw/
python scripts/prepare_data.py --documents data/raw/mydocs.txt ...
```

**Note**: Files here are not tracked by git (in `.gitignore`)

#### `data/processed/` - Processed Training Data

**Purpose**: Store prepared training data in JSON format

**Expected Files** (created by `prepare_data.py`):
- `train.json` - Training pairs (query, positive, negative)
- `val.json` - Validation pairs
- `test.json` - Test pairs (training format)
- `test_eval.json` - Test data (evaluation format)
- `retriever_index.pkl` - Saved retriever index (optional)

**Format**:
```json
[
  {
    "query": "What is machine learning?",
    "positive": "ML is a subset of AI...",
    "negative": "Python is a language..."  // optional
  }
]
```

**Note**: Files here are not tracked by git

#### `data/embeddings/` - Saved Embeddings

**Purpose**: Store computed embeddings and trained models

**Typical Contents**:
- Trained model weights (`.pt` files)
- Pre-computed document embeddings
- FAISS indices

**Usage**:
```python
retriever = GaussianRetriever(
    model_path='data/embeddings/my_model.pt'
)
```

---

## ⚙️ Configuration

### `config/` - Configuration Files

**Purpose**: Store training and model configurations

**Contents**:
- `train_example.json` - Example training configuration

**Config Format**:
```json
{
  "train_data": "data/processed/train.json",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 2e-5,
  "output_dim": 512
}
```

**Usage**:
```bash
# Load config and override specific params
python scripts/train.py --epochs 5 --batch_size 16
```

---

## 📚 Examples & Notebooks

### `examples/` - Example Scripts

**Purpose**: Runnable examples showing how to use RAGCUN

**Contents**:
- `retrieval_example.py` - Basic retrieval example

**Run examples**:
```bash
python examples/retrieval_example.py
```

**What it does**:
1. Loads a trained model (or uses untrained for demo)
2. Adds sample documents
3. Performs retrieval with example queries
4. Explains Euclidean distance vs cosine similarity

### `notebooks/` - Jupyter Notebooks

**Purpose**: Interactive notebooks for Google Colab

**Contents** (5 notebooks):

1. **`lejepa_training.ipynb`** (41KB)
   - Complete training pipeline
   - LeJEPA SIGReg loss implementation
   - Model evaluation
   - **Use this for**: Training in Colab with GPU/TPU

2. **`lejepa_training_tpu.ipynb`** (16KB)
   - TPU-optimized training
   - Google Cloud TPU setup
   - **Use this for**: Training on Colab TPUs

3. **`evaluate_rag.ipynb`** (12KB)
   - RAG system evaluation
   - Retrieval metrics (Recall, MRR, NDCG, MAP)
   - **Use this for**: Evaluating retrieval performance

4. **`evaluate_isotropy.ipynb`** (10KB)
   - Embedding quality analysis
   - Isotropy measurements
   - Visualization of embedding distribution
   - **Use this for**: Understanding embedding properties

5. **`document_processing.ipynb`** (9.5KB)
   - Document loading and preprocessing
   - Text cleaning and chunking
   - **Use this for**: Preparing documents for RAG

**Opening in Colab**:
```
https://colab.research.google.com/github/ctn/ragcun/blob/main/notebooks/[notebook_name]
```

**Note**: These notebooks are Colab-optimized but work in local Jupyter too

---

## 🧪 Testing

### `tests/` - Unit Tests

**Purpose**: Automated tests for code quality

**Current Status**: Empty (ready for tests)

**Structure** (to be added):
```
tests/
├── test_model.py         # Model tests
├── test_retriever.py     # Retriever tests
├── test_training.py      # Training pipeline tests
└── test_evaluation.py    # Evaluation tests
```

**Running tests** (when added):
```bash
pytest tests/
pytest tests/test_model.py -v
```

---

## 🔧 External Dependencies

### `external/` - Third-party Code

**Purpose**: External libraries or research code

**Contents**:
- `lejepa/` - LeJEPA reference implementation (currently empty)

**Typical Use**:
- Reference implementations
- Research code
- Dependencies not in pip

**Note**: This directory may contain third-party code with different licenses

---

## 📊 Output Directories

These directories are created automatically during training/evaluation:

### `checkpoints/` - Model Checkpoints

**Created by**: `train.py`

**Contents** (after training):
- `best_model.pt` - Best model by validation loss
- `final_model.pt` - Final model after all epochs
- `checkpoint_epoch_N.pt` - Periodic checkpoints
- `train_config.json` - Training configuration

**Size**: Each checkpoint ~300-600MB (depends on model)

**Usage**:
```python
model = GaussianEmbeddingGemma.from_pretrained(
    'checkpoints/best_model.pt'
)
```

### `results/` - Evaluation Results

**Created by**: `evaluate.py`

**Contents** (after evaluation):
- `eval_results.json` - Evaluation metrics
- `quickstart_results.json` - Quickstart demo results

**Format**:
```json
{
  "metrics": {
    "Recall@10": 0.7543,
    "MRR": 0.6234,
    "NDCG@10": 0.6891,
    "MAP@100": 0.5678
  }
}
```

---

## 🚫 Ignored Directories

These are not tracked in git:

- `.venv/` - Python virtual environment
- `.git/` - Git repository metadata
- `__pycache__/` - Python bytecode cache
- `*.egg-info/` - Package metadata
- `.claude/` - Claude Code configuration

---

## 📈 Typical Data Flow

```
1. Raw Data
   data/raw/mydocs.txt
   ↓
2. Preparation
   prepare_data.py
   ↓
3. Processed Data
   data/processed/train.json
   ↓
4. Training
   train.py
   ↓
5. Checkpoints
   checkpoints/best_model.pt
   ↓
6. Evaluation
   evaluate.py
   ↓
7. Results
   results/eval_results.json
```

---

## 💡 Best Practices

### Data Organization

1. **Keep raw data safe**: Never modify files in `data/raw/`
2. **Version control**: Track code, not data
3. **Document processing**: Keep preprocessing scripts in `examples/`
4. **Separate concerns**: Train/val/test in separate files

### Model Management

1. **Name checkpoints**: Use descriptive names
   ```
   checkpoints/epoch3_lr2e5_bs8.pt
   ```

2. **Save configs**: Always save training config with model
   ```python
   torch.save({'model': model, 'config': config}, path)
   ```

3. **Track experiments**: Use subdirectories
   ```
   checkpoints/
   ├── experiment1/
   ├── experiment2/
   └── baseline/
   ```

### Code Organization

1. **Import from package**: Use `from ragcun import ...`
2. **Add tests**: Put new tests in `tests/`
3. **Document changes**: Update README when adding features
4. **Use examples**: Add new examples to `examples/`

---

## 🔍 Quick Reference

### Find a File

```bash
# Search for a pattern in code
grep -r "GaussianEmbedding" ragcun/

# Find all Python files
find . -name "*.py" -not -path "./.venv/*"

# Find notebooks
ls notebooks/*.ipynb
```

### Clean Up

```bash
# Remove checkpoints
rm -rf checkpoints/

# Remove processed data
rm -rf data/processed/*

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

### Disk Usage

```bash
# Check directory sizes
du -sh */

# Find large files
find . -type f -size +100M
```

---

## 📝 Summary Table

| Directory | Purpose | Git Tracked | Created By | Size |
|-----------|---------|-------------|------------|------|
| `ragcun/` | Source code | ✅ Yes | Manual | ~10KB |
| `data/raw/` | Input docs | ❌ No | User | Varies |
| `data/processed/` | Training data | ❌ No | `prepare_data.py` | ~1-100MB |
| `data/embeddings/` | Models/embeddings | ❌ No | User/training | ~300-600MB |
| `config/` | Configs | ✅ Yes | Manual | ~1KB |
| `examples/` | Examples | ✅ Yes | Manual | ~10KB |
| `notebooks/` | Jupyter notebooks | ✅ Yes | Manual | ~100KB |
| `external/` | Third-party code | ⚠️ Maybe | Manual | Varies |
| `tests/` | Unit tests | ✅ Yes | Manual | Empty |
| `checkpoints/` | Model weights | ❌ No | `train.py` | ~300-600MB |
| `results/` | Eval results | ❌ No | `evaluate.py` | ~1-10KB |

---

## 🎯 Next Steps

1. **New User**: Start with `./quickstart.sh`
2. **Add Data**: Put documents in `data/raw/`
3. **Train Model**: Use `train.py` or notebooks
4. **Evaluate**: Use `evaluate.py` for metrics
5. **Production**: Import `ragcun` package in your code

For more details, see:
- `README.md` - Project overview
- `TRAINING_GUIDE.md` - Training instructions
- `SCRIPTS_README.md` - Script reference
