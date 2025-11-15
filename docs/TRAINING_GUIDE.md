# Training Guide for RAGCUN

This guide explains how to train and evaluate your GaussianEmbeddingGemma model on GPU machines.

## Quick Start

```bash
# 1. Prepare your data
python scripts/prepare_data.py --documents data/docs.txt --generate_pairs --split 0.8 0.1 0.1 --output_dir data/processed

# 2. Train the model
python scripts/train.py --train_data data/processed/train.json --val_data data/processed/val.json --epochs 3

# 3. Evaluate the model
python scripts/evaluate.py --model_path checkpoints/best_model.pt --test_data data/processed/test_eval.json
```

## Detailed Instructions

### 1. Data Preparation

The model requires training data in the format of query-positive-negative triplets.

#### Option A: From Raw Documents

If you have a collection of documents, generate synthetic pairs:

```bash
python scripts/prepare_data.py \
    --documents data/raw/documents.txt \
    --generate_pairs \
    --num_pairs 10000 \
    --add_negatives \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

This will create:
- `data/processed/train.json` - Training data
- `data/processed/val.json` - Validation data
- `data/processed/test_eval.json` - Test data (evaluation format)

#### Option B: From CSV

If you already have query-document pairs:

```bash
# CSV format: query,positive,negative
python scripts/prepare_data.py \
    --csv_file data/pairs.csv \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

#### Option C: Manual Format

Create JSON files manually with this format:

```json
[
  {
    "query": "What is machine learning?",
    "positive": "Machine learning is a subset of AI that enables systems to learn from data.",
    "negative": "Python is a programming language."
  },
  ...
]
```

### 2. Training

#### Basic Training

```bash
python scripts/train.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 \
    --batch_size 8 \
    --output_dir checkpoints
```

#### Advanced Training Options

```bash
python scripts/train.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --output_dim 512 \
    --freeze_early_layers \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --margin 1.0 \
    --output_dir checkpoints \
    --mixed_precision
```

#### Key Hyperparameters

- **`--batch_size`**: Start with 8 for T4 GPU (15GB). Increase if you have more memory.
- **`--learning_rate`**: Default 2e-5 works well. Lower (1e-5) for fine-tuning.
- **`--epochs`**: 3-5 epochs usually sufficient. Monitor validation loss.
- **`--freeze_early_layers`**: Freeze first 4 transformer layers to preserve general knowledge.
- **`--lambda_isotropy`**: Weight for isotropy loss (encourages uniform distribution).
- **`--lambda_reg`**: Weight for variance regularization.
- **`--margin`**: Margin for contrastive loss (distance between pos/neg pairs).
- **`--mixed_precision`**: Use FP16 for faster training (experimental).

#### Training Output

The script will save:
- `checkpoints/checkpoint_epoch_N.pt` - Checkpoint every N epochs
- `checkpoints/best_model.pt` - Best model based on validation loss
- `checkpoints/final_model.pt` - Final model after all epochs
- `checkpoints/train_config.json` - Training configuration
- `training.log` - Detailed training logs

#### Monitoring Training

Watch the logs for:
- **Total Loss**: Should decrease over time
- **Contrastive Loss**: Measures query-doc similarity
- **Isotropy Loss**: Measures uniformity of embedding space
- **Pos Distance (mean)**: Average distance between query and positive doc
- **Embedding Std**: Standard deviation of embeddings (target: ~1.0)

### 3. Evaluation

Evaluate your trained model on a test set:

```bash
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/processed/test_eval.json \
    --batch_size 32 \
    --top_k 100 \
    --output_file results/eval_results.json
```

#### Evaluation Metrics

The script computes:

- **Recall@K**: Fraction of relevant docs in top-K results
  - Recall@1, @5, @10, @20, @50, @100
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant doc
- **NDCG@10**: Normalized Discounted Cumulative Gain (position-aware)
- **MAP@100**: Mean Average Precision

#### Interpreting Results

Good performance benchmarks:
- **Recall@10 > 0.7**: Model finds relevant docs in top 10
- **MRR > 0.5**: First relevant doc is typically in top 2-3
- **NDCG@10 > 0.6**: Good ranking quality

### 4. Using the Trained Model

After training, use your model for retrieval:

```python
from ragcun import GaussianRetriever

# Load trained model
retriever = GaussianRetriever(
    model_path='checkpoints/best_model.pt',
    embedding_dim=512,
    use_gpu=True
)

# Add documents
documents = [
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
    # ... more documents
]
retriever.add_documents(documents)

# Retrieve
results = retriever.retrieve("What is ML?", top_k=5)
for doc, distance in results:
    print(f"Distance: {distance:.3f} - {doc}")
```

## GPU Memory Management

### T4 GPU (15GB)

- Batch size 8: ~8GB memory (safe)
- Batch size 16: ~12GB memory (near limit)
- Batch size 32: May OOM (Out of Memory)

### Reducing Memory Usage

If you encounter OOM errors:

```bash
# Reduce batch size
python scripts/train.py --batch_size 4 ...

# Freeze more layers
python scripts/train.py --freeze_early_layers ...

# Use gradient accumulation (simulate larger batch)
# (not implemented yet - coming soon)
```

### Monitoring GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi
```

## Troubleshooting

### Issue: OOM during training

**Solution**: Reduce `--batch_size` to 4 or lower

### Issue: Loss not decreasing

**Solutions**:
- Check your data format and quality
- Reduce learning rate: `--learning_rate 1e-5`
- Increase warmup: `--warmup_steps 200`
- Ensure positives are actually relevant to queries

### Issue: High isotropy loss

**Solution**: This is expected initially. It should decrease as training progresses.

### Issue: Embeddings collapse (std → 0)

**Solutions**:
- Increase `--lambda_reg` to 0.5 or 1.0
- Check your data has enough diversity
- Reduce `--lambda_isotropy`

## Advanced Topics

### Custom Loss Weights

Tune loss component weights:

```bash
python scripts/train.py \
    --lambda_isotropy 1.5 \  # Stronger isotropy
    --lambda_reg 0.2 \       # More variance preservation
    --margin 0.5             # Smaller margin
```

### Resume Training

```python
# Load checkpoint and continue training
checkpoint = torch.load('checkpoints/checkpoint_epoch_2.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Fine-tuning

If you have a pre-trained model:

```bash
python scripts/train.py \
    --model_path checkpoints/pretrained.pt \  # Not implemented yet
    --learning_rate 1e-5 \                     # Lower LR
    --epochs 2 \                                # Fewer epochs
    --freeze_early_layers                       # Preserve knowledge
```

## Example Workflow

Complete workflow example:

```bash
# Step 1: Prepare data (from 10k documents)
python scripts/prepare_data.py \
    --documents data/raw/docs.txt \
    --generate_pairs \
    --num_pairs 10000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed

# Step 2: Train (3 epochs)
python scripts/train.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 \
    --batch_size 8 \
    --output_dir checkpoints

# Step 3: Evaluate
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/processed/test_eval.json \
    --output_file results/results.json

# Step 4: Use in production
python -c "
from ragcun import GaussianRetriever
retriever = GaussianRetriever('checkpoints/best_model.pt')
# ... your code
"
```

## Best Practices

1. **Start Small**: Begin with a small dataset (1k examples) to validate your pipeline
2. **Monitor Validation**: Always use validation data to detect overfitting
3. **Save Checkpoints**: Training can be interrupted - save regularly
4. **Log Everything**: Keep detailed logs for debugging and analysis
5. **Version Control**: Track your training configs and hyperparameters
6. **Evaluate Often**: Run evaluation after each training run
7. **GPU Utilization**: Monitor `nvidia-smi` to ensure efficient GPU usage

## Citation

If you use RAGCUN in your research:

```bibtex
@software{ragcun2024,
  title={RAGCUN: LeJEPA Isotropic Gaussian Embeddings for RAG},
  author={RAGCUN Team},
  year={2024},
  url={https://github.com/ctn/ragcun}
}
```

## Support

- Issues: https://github.com/ctn/ragcun/issues
- Documentation: https://github.com/ctn/ragcun
