# Training Scripts Overview

This directory contains Python programs for training, evaluating, and managing your RAGCUN models on GPU machines.

## 🚀 Quick Start

Run the quickstart script to see everything in action:

```bash
./quickstart.sh
```

This will:
1. Prepare sample training data
2. Train a model for 2 epochs
3. Evaluate the model
4. Show you how to use it

## 📁 Available Scripts

### 1. `train.py` - Model Training

Train IsotropicGaussianEncoder with LeJEPA SIGReg loss.

**Basic usage:**
```bash
python scripts/train.py --train_data data/processed/train.json --epochs 3
```

**Key features:**
- LeJEPA SIGReg loss (contrastive + isotropy + regularization)
- Automatic mixed precision training (FP16)
- Learning rate warmup and cosine annealing
- Checkpoint saving and validation
- Comprehensive logging

**Key arguments:**
- `--train_data`: Path to training JSON
- `--val_data`: Path to validation JSON (optional)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size (default: 8 for T4 GPU)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--output_dir`: Where to save checkpoints (default: checkpoints/)
- `--freeze_early_layers`: Freeze first 4 transformer layers
- `--lambda_isotropy`: Weight for isotropy loss (default: 1.0)
- `--lambda_reg`: Weight for regularization (default: 0.1)
- `--margin`: Contrastive loss margin (default: 1.0)

**Output:**
- `checkpoints/best_model.pt` - Best model by validation loss
- `checkpoints/final_model.pt` - Final model after all epochs
- `checkpoints/checkpoint_epoch_N.pt` - Periodic checkpoints
- `checkpoints/train_config.json` - Training configuration
- `training.log` - Detailed logs

### 2. `evaluate.py` - Model Evaluation

Evaluate trained models on retrieval metrics.

**Basic usage:**
```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pt --test_data data/processed/test_eval.json
```

**Metrics computed:**
- **Recall@K**: K ∈ {1, 5, 10, 20, 50, 100}
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **MAP@100**: Mean Average Precision

**Key arguments:**
- `--model_path`: Path to trained model checkpoint
- `--test_data`: Path to test data JSON
- `--output_file`: Save results to JSON (optional)
- `--batch_size`: Batch size for encoding (default: 32)
- `--top_k`: Number of documents to retrieve (default: 100)

**Test data format:**
```json
{
  "corpus": ["doc1", "doc2", ...],
  "queries": ["query1", "query2", ...],
  "relevance": [[0, 5], [1, 3], ...]
}
```

### 3. `prepare_data.py` - Data Preparation

Prepare training data from various sources.

**Generate from documents:**
```bash
python scripts/prepare_data.py \
    --documents data/docs.txt \
    --generate_pairs \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**From CSV:**
```bash
python scripts/prepare_data.py \
    --csv_file data/pairs.csv \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**Key features:**
- Supports multiple input formats (text files, CSV, directories)
- Generates synthetic query-document pairs
- Automatic train/val/test splitting
- Hard negative mining
- Validation and format checking

**Key arguments:**
- `--documents`: Text file with one document per line
- `--csv_file`: CSV with query, positive, negative columns
- `--input_dir`: Directory of text files
- `--generate_pairs`: Generate synthetic pairs
- `--split`: Train/val/test ratios (e.g., 0.8 0.1 0.1)
- `--output_dir`: Output directory for splits

**Training data format:**
```json
[
  {
    "query": "What is machine learning?",
    "positive": "Machine learning is...",
    "negative": "Python is..."  // optional
  },
  ...
]
```

### 4. `quickstart.sh` - One-Command Demo

Complete end-to-end demo:

```bash
./quickstart.sh
```

Runs:
1. Data preparation (100 pairs from sample docs)
2. Training (2 epochs)
3. Evaluation
4. Shows usage examples

Perfect for:
- First-time setup
- Verifying installation
- Learning the workflow
- Quick experiments

## 📊 Example Workflow

### Complete Training Pipeline

```bash
# Step 1: Prepare your data
echo "doc1
doc2
doc3" > data/raw/mydocs.txt

python scripts/prepare_data.py \
    --documents data/raw/mydocs.txt \
    --generate_pairs \
    --num_pairs 1000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed

# Step 2: Train the model
python scripts/train.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 \
    --batch_size 8 \
    --freeze_early_layers \
    --output_dir checkpoints

# Step 3: Evaluate
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/processed/test_eval.json \
    --output_file results/eval.json

# Step 4: Use the model
python << EOF
from ragcun import IsotropicRetriever

retriever = IsotropicRetriever('checkpoints/best_model.pt', use_gpu=True)
retriever.add_documents(["doc1", "doc2", "doc3"])
results = retriever.retrieve("my query", top_k=5)

for doc, dist in results:
    print(f"{dist:.3f}: {doc}")
EOF
```

## 🔧 Configuration

Example configuration (`config/train_example.json`):

```json
{
  "train_data": "data/processed/train.json",
  "val_data": "data/processed/val.json",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 2e-5,
  "output_dim": 512,
  "freeze_early_layers": true,
  "lambda_isotropy": 1.0,
  "lambda_reg": 0.1,
  "margin": 1.0
}
```

Use with:
```bash
python scripts/train.py @config/train_example.json  # Not implemented yet
# For now, pass args directly
```

## 💡 Tips & Best Practices

### GPU Memory Management (T4 with 15GB)

- **Batch size 8**: ~8GB (safe, recommended)
- **Batch size 16**: ~12GB (near limit)
- **Batch size 32**: May OOM

If OOM occurs:
```bash
python scripts/train.py --batch_size 4  # Reduce batch size
```

### Training Tips

1. **Start small**: Test with 100-1000 examples first
2. **Use validation**: Always provide `--val_data`
3. **Monitor logs**: Watch `training.log` and console output
4. **Save often**: Default saves every epoch
5. **Freeze layers**: Use `--freeze_early_layers` to preserve knowledge

### Key Metrics to Watch

**During training:**
- Total loss should decrease
- Pos distance should be small (< 2.0)
- Embedding std should be ~1.0

**During evaluation:**
- Recall@10 > 0.7 is good
- MRR > 0.5 is good
- NDCG@10 > 0.6 is good

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
python scripts/train.py --batch_size 4
```

### "Loss not decreasing"
- Check data quality
- Reduce learning rate: `--learning_rate 1e-5`
- Increase warmup: `--warmup_steps 200`

### "Model not loading"
- Check `output_dim` matches training
- Verify checkpoint file exists
- Check CUDA availability

### "ImportError: No module named 'ragcun'"
```bash
pip install -e .
```

## 📚 Additional Resources

- **Detailed Guide**: See `TRAINING_GUIDE.md`
- **API Reference**: See `README.md`
- **Examples**: See `examples/` directory
- **Issues**: https://github.com/ctn/ragcun/issues

## 🎯 Common Use Cases

### 1. Train on Custom Dataset

```bash
# Prepare your data
python scripts/prepare_data.py \
    --csv_file mydata.csv \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed

# Train
python scripts/train.py \
    --train_data data/processed/train.json \
    --epochs 5 \
    --batch_size 8
```

### 2. Fine-tune Existing Model

```bash
# Load checkpoint and continue training
python scripts/train.py \
    --train_data new_data.json \
    --learning_rate 1e-5 \
    --epochs 2 \
    --freeze_early_layers
```

### 3. Evaluate Multiple Checkpoints

```bash
for ckpt in checkpoints/*.pt; do
    echo "Evaluating $ckpt"
    python scripts/evaluate.py \
        --model_path "$ckpt" \
        --test_data data/processed/test_eval.json \
        --output_file "results/$(basename $ckpt .pt).json"
done
```

### 4. Hyperparameter Search

```bash
for margin in 0.5 1.0 2.0; do
    for lambda_iso in 0.5 1.0 1.5; do
        python scripts/train.py \
            --train_data data/processed/train.json \
            --margin $margin \
            --lambda_isotropy $lambda_iso \
            --output_dir "checkpoints/m${margin}_li${lambda_iso}"
    done
done
```

## 🚀 Getting Started Now

1. **Verify installation:**
   ```bash
   python -c "import ragcun; print('✅ RAGCUN installed')"
   ```

2. **Run quickstart:**
   ```bash
   ./quickstart.sh
   ```

3. **Check results:**
   ```bash
   cat results/quickstart_results.json
   ```

4. **Try with your data:**
   ```bash
   python scripts/prepare_data.py --documents your_docs.txt ...
   python scripts/train.py --train_data data/processed/train.json ...
   ```

Happy training! 🎉
