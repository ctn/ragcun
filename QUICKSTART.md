# RAGCUN Quickstart Guide

Get started training your RAG model in minutes!

## ⚡ Super Quick Start (5 minutes)

Run the complete pipeline with one command:

```bash
cd /home/ubuntu/ragcun
./scripts/pipeline_quick.sh
```

This will:
1. ✅ Prepare 50 training examples
2. ✅ Train for 1 epoch (~3 minutes on T4 GPU)
3. ✅ Evaluate the model
4. ✅ Save results

**Done!** Your model is ready at `checkpoints/quick/final_model.pt`

---

## 🚀 Production Training (1-2 hours)

For real training with full dataset:

```bash
cd /home/ubuntu/ragcun
./scripts/pipeline_full.sh
```

This will:
1. ✅ Prepare 1000 training examples from 61 documents
2. ✅ Train for 3 epochs with optimal settings
3. ✅ Evaluate on test set
4. ✅ Save best model

**Result**: Production-ready model at `checkpoints/full/best_model.pt`

---

## 📊 What's Available

### Shell Scripts (Ready to Run!)

**Quick Workflows**:
```bash
./scripts/pipeline_quick.sh      # Complete test pipeline (~5 min)
./scripts/pipeline_full.sh       # Complete production pipeline (~1-2 hrs)
```

**Data Preparation**:
```bash
./scripts/prepare_data_quick.sh  # Generate 50 pairs for testing
./scripts/prepare_data_full.sh   # Generate 1000 pairs for production
```

**Training**:
```bash
./scripts/train_quick.sh         # Quick training (1 epoch)
./scripts/train_full.sh          # Full training (3 epochs)
./scripts/train_custom.sh        # Custom hyperparameters
```

**Evaluation**:
```bash
./scripts/eval.sh                # Evaluate single model
./scripts/eval_all.sh            # Evaluate all checkpoints
```

**Utilities**:
```bash
./scripts/gpu_info.sh            # Check GPU status
./scripts/monitor_training.sh    # Monitor training in real-time
./scripts/hyperparameter_search.sh  # Grid search experiments
```

### Data Available

- **tech_docs.txt**: 41 documents on programming, ML, DevOps, cloud
- **science_docs.txt**: 20 documents on biology, physics, chemistry
- **training_pairs.json**: 20 hand-crafted query-document pairs
- **sample_docs.txt**: 10 documents for quick tests

**Total**: 61+ documents ready for training!

---

## 🎯 Common Tasks

### Check Your GPU
```bash
./scripts/gpu_info.sh
```

### Train a Model
```bash
# Quick test (5 min)
./scripts/prepare_data_quick.sh
./scripts/train_quick.sh

# Production (1-2 hrs)
./scripts/prepare_data_full.sh
./scripts/train_full.sh
```

### Evaluate Your Model
```bash
./scripts/eval.sh checkpoints/best_model.pt
```

### Monitor Training Progress
```bash
# Terminal 1: Start training
./scripts/train_full.sh

# Terminal 2: Monitor
./scripts/monitor_training.sh
```

### Use Your Trained Model
```python
from ragcun import GaussianRetriever

# Load model
retriever = GaussianRetriever(
    model_path='checkpoints/best_model.pt',
    use_gpu=True
)

# Add documents
retriever.add_documents([
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "Quantum computing uses qubits"
])

# Retrieve
results = retriever.retrieve("What is ML?", top_k=5)
for doc, distance in results:
    print(f"[{distance:.3f}] {doc}")
```

---

## 📁 Project Structure

```
ragcun/
├── scripts/              # Shell scripts (start here!)
│   ├── pipeline_quick.sh    # Quick end-to-end pipeline
│   ├── pipeline_full.sh     # Full production pipeline
│   ├── train_*.sh           # Training scripts
│   ├── eval*.sh             # Evaluation scripts
│   ├── prepare_data_*.sh    # Data preparation
│   └── *.py                 # Python implementations
├── data/
│   ├── raw/              # Input documents
│   │   ├── tech_docs.txt
│   │   ├── science_docs.txt
│   │   └── training_pairs.json
│   └── processed/        # Generated training data (created by scripts)
├── checkpoints/          # Trained models (created by training)
├── results/              # Evaluation results (created by eval)
└── ragcun/               # Python package
    ├── model.py          # GaussianEmbeddingGemma
    └── retriever.py      # GaussianRetriever
```

---

## 🎓 Step-by-Step Tutorial

### Step 1: Check Your Environment

```bash
# Navigate to project
cd /home/ubuntu/ragcun

# Check GPU
./scripts/gpu_info.sh

# Expected output:
# GPU Details:
#   Name: Tesla T4
#   Memory: 15360 MiB
#   ...
```

### Step 2: Quick Test Run

```bash
# Run quick pipeline (5 minutes)
./scripts/pipeline_quick.sh

# This creates:
#   - data/processed/quick/train.json
#   - checkpoints/quick/final_model.pt
#   - results/quick_results.json
```

### Step 3: View Results

```bash
# View evaluation results
cat results/quick_results.json

# Expected metrics:
# {
#   "metrics": {
#     "MRR": 0.45,
#     "Recall@10": 0.65,
#     "NDCG@10": 0.58
#   }
# }
```

### Step 4: Production Training

```bash
# Prepare full dataset (1000 pairs)
./scripts/prepare_data_full.sh

# Train with optimal settings (1-2 hours)
./scripts/train_full.sh

# Evaluate
./scripts/eval.sh checkpoints/full/best_model.pt
```

### Step 5: Use Your Model

Create `test_retrieval.py`:
```python
from ragcun import GaussianRetriever

# Load trained model
retriever = GaussianRetriever('checkpoints/full/best_model.pt')

# Add your documents
docs = [
    "Python is great for machine learning",
    "Docker containers are lightweight",
    "React is a JavaScript library"
]
retriever.add_documents(docs)

# Query
results = retriever.retrieve("programming language for ML", top_k=3)

for doc, dist in results:
    print(f"Distance {dist:.3f}: {doc}")
```

Run it:
```bash
python test_retrieval.py
```

---

## 💡 Tips for Success

### GPU Memory
- **T4 GPU (15GB)**: Use batch size 8 (recommended)
- **Smaller GPU**: Use batch size 4
- **OOM errors**: Reduce batch size

```bash
# Safe settings for T4
./scripts/train_full.sh data/processed/train.json data/processed/val.json checkpoints 8

# Conservative (if OOM)
./scripts/train_full.sh data/processed/train.json data/processed/val.json checkpoints 4
```

### Training Time Estimates

| Dataset | Epochs | Batch Size | GPU | Time |
|---------|--------|------------|-----|------|
| 50 pairs | 1 | 8 | T4 | ~3 min |
| 500 pairs | 3 | 8 | T4 | ~30 min |
| 1000 pairs | 3 | 8 | T4 | ~1-2 hrs |
| 1000 pairs | 5 | 8 | T4 | ~2-3 hrs |

### When to Use Each Script

| Goal | Script | Why |
|------|--------|-----|
| Test everything works | `pipeline_quick.sh` | Fast, end-to-end validation |
| Train production model | `pipeline_full.sh` | Best settings, full data |
| Experiment with hyperparameters | `train_custom.sh` | Full control |
| Compare multiple models | `hyperparameter_search.sh` | Automated grid search |
| Debug issues | `train_quick.sh` | Fast iteration |

---

## 🐛 Troubleshooting

### Script not found
```bash
cd /home/ubuntu/ragcun
ls scripts/  # Verify scripts exist
```

### Permission denied
```bash
chmod +x scripts/*.sh
```

### Training data not found
```bash
# Prepare data first
./scripts/prepare_data_quick.sh
```

### CUDA out of memory
```bash
# Reduce batch size
./scripts/train_custom.sh --train_data data/processed/train.json --batch_size 4
```

### Training seems stuck
```bash
# Check GPU usage
./scripts/gpu_info.sh

# Monitor training
./scripts/monitor_training.sh

# Check logs
tail -f training.log
```

---

## 📚 Next Steps

1. **Read the guides**:
   - `scripts/README.md` - All shell scripts
   - `TRAINING_GUIDE.md` - Detailed training guide
   - `DATA_SUMMARY.md` - Data documentation

2. **Explore notebooks**: `notebooks/` (for Google Colab)

3. **Try examples**: `examples/retrieval_example.py`

4. **Add your data**:
   ```bash
   # Add documents to data/raw/
   echo "Your document text" > data/raw/my_docs.txt

   # Generate training pairs
   ./scripts/prepare_data_full.sh

   # Train
   ./scripts/train_full.sh
   ```

5. **Experiment**:
   ```bash
   # Try different hyperparameters
   ./scripts/hyperparameter_search.sh

   # Compare results
   ./scripts/eval_all.sh checkpoints/hypersearch
   ```

---

## 🎉 You're Ready!

Start with the quick pipeline to test everything:

```bash
./scripts/pipeline_quick.sh
```

Then move to production training:

```bash
./scripts/pipeline_full.sh
```

**Questions?** Check the documentation in the root directory!

Happy training! 🚀
