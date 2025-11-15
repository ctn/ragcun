# Shell Scripts Reference

This directory contains shell scripts to automate common workflows for training and evaluating RAGCUN models.

## 🚀 Quick Start Scripts

### One-Command Workflows

#### `pipeline_quick.sh` - Test Everything (~5 min)
Complete pipeline for quick testing:
```bash
./scripts/pipeline_quick.sh
```
**Does**: Prep 50 pairs → Train 1 epoch → Evaluate

#### `pipeline_full.sh` - Full Training (~1-2 hours)
Complete production pipeline:
```bash
./scripts/pipeline_full.sh
```
**Does**: Prep 1000 pairs → Train 3 epochs → Evaluate

---

## 📊 Data Preparation Scripts

### `prepare_data_quick.sh` - Quick Test Data
Generate small dataset for testing:
```bash
./scripts/prepare_data_quick.sh [input_file] [output_dir] [num_pairs]

# Examples:
./scripts/prepare_data_quick.sh                                    # 50 pairs from sample_docs.txt
./scripts/prepare_data_quick.sh data/raw/tech_docs.txt             # 50 pairs from tech docs
./scripts/prepare_data_quick.sh data/raw/tech_docs.txt data/processed 100  # 100 pairs
```

### `prepare_data_full.sh` - Production Data
Generate full dataset from all documents:
```bash
./scripts/prepare_data_full.sh [output_dir] [num_pairs]

# Examples:
./scripts/prepare_data_full.sh                     # 1000 pairs to data/processed
./scripts/prepare_data_full.sh data/processed 2000 # 2000 pairs
```
**Combines**: tech_docs.txt + science_docs.txt (61 documents)

### `generate_training_data.sh` - Generate All Variants
Create multiple dataset variants:
```bash
./scripts/generate_training_data.sh
```
**Creates**:
- `data/processed/tech_only/` - 500 pairs from tech docs
- `data/processed/science_only/` - 250 pairs from science docs
- `data/processed/combined/` - 1000 pairs from all docs
- `data/processed/premade/` - 20 curated pairs

---

## 🎓 Training Scripts

### `train_quick.sh` - Quick Training (1 epoch, ~3 min)
Fast training for testing:
```bash
./scripts/train_quick.sh [train_data] [val_data] [output_dir] [batch_size]

# Examples:
./scripts/train_quick.sh                                      # Use defaults
./scripts/train_quick.sh data/processed/train.json            # Custom train data
./scripts/train_quick.sh data/processed/train.json data/processed/val.json checkpoints/test 4  # All params
```

**Settings**:
- 1 epoch
- Batch size: 8 (default)
- Learning rate: 2e-5
- Output: `checkpoints/quick/`

### `train_full.sh` - Full Training (3 epochs, ~1-2 hours)
Production training with recommended settings:
```bash
./scripts/train_full.sh [train_data] [val_data] [output_dir] [batch_size] [epochs]

# Examples:
./scripts/train_full.sh                                       # Use defaults
./scripts/train_full.sh data/processed/train.json             # 3 epochs, batch 8
./scripts/train_full.sh data/processed/train.json data/processed/val.json checkpoints/prod 8 5  # 5 epochs
```

**Settings**:
- 3 epochs (default)
- Batch size: 8 (default)
- Learning rate: 2e-5
- Freezes early layers
- Full loss components
- Output: `checkpoints/full/`

### `train_custom.sh` - Custom Hyperparameters
Fine-grained control over training:
```bash
./scripts/train_custom.sh --train_data PATH [OPTIONS]

# Required:
--train_data PATH              Training data file

# Optional:
--val_data PATH                Validation data file
--output_dir PATH              Output directory (default: checkpoints/custom)
--batch_size N                 Batch size (default: 8)
--epochs N                     Number of epochs (default: 3)
--lr FLOAT                     Learning rate (default: 2e-5)
--lambda_iso FLOAT             Isotropy loss weight (default: 1.0)
--lambda_reg FLOAT             Regularization weight (default: 0.1)
--margin FLOAT                 Contrastive margin (default: 1.0)
--freeze_layers                Freeze early transformer layers
-h, --help                     Show help

# Examples:
./scripts/train_custom.sh --train_data data/processed/train.json --epochs 5 --batch_size 16

./scripts/train_custom.sh \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --lr 1e-5 \
    --lambda_iso 1.5 \
    --freeze_layers
```

---

## 📈 Evaluation Scripts

### `eval.sh` - Evaluate Single Model
Evaluate a trained model:
```bash
./scripts/eval.sh [model_path] [test_data] [output_file] [batch_size]

# Examples:
./scripts/eval.sh                                          # Use defaults
./scripts/eval.sh checkpoints/best_model.pt                # Specific model
./scripts/eval.sh checkpoints/best_model.pt data/processed/test_eval.json results/eval.json
```

**Metrics**: Recall@K, MRR, NDCG@10, MAP@100

### `eval_all.sh` - Evaluate All Checkpoints
Evaluate all models in a directory:
```bash
./scripts/eval_all.sh [checkpoint_dir] [test_data] [output_dir]

# Examples:
./scripts/eval_all.sh                                      # Evaluate all in checkpoints/
./scripts/eval_all.sh checkpoints/full                     # Specific checkpoint dir
./scripts/eval_all.sh checkpoints/hypersearch data/processed/test_eval.json results/comparison
```

**Output**: Individual results + summary comparison

---

## 🔬 Advanced Scripts

### `hyperparameter_search.sh` - Grid Search
Test multiple hyperparameter combinations:
```bash
./scripts/hyperparameter_search.sh [train_data] [val_data] [test_data]

# Example:
./scripts/hyperparameter_search.sh data/processed/train.json data/processed/val.json data/processed/test_eval.json
```

**Tests**:
- Learning rates: 1e-5, 2e-5, 5e-5
- Batch sizes: 4, 8, 16
- Isotropy weights: 0.5, 1.0, 1.5

**Time**: 3-6 hours on T4 GPU
**Output**: `checkpoints/hypersearch/` + `results/hypersearch/`

---

## 🛠️ Utility Scripts

### `gpu_info.sh` - GPU Status & Recommendations
Display GPU info and batch size recommendations:
```bash
./scripts/gpu_info.sh
```

**Shows**:
- GPU model and memory
- Current utilization
- Recommended batch sizes
- Running training processes

### `monitor_training.sh` - Real-time Monitor
Monitor training progress in real-time:
```bash
./scripts/monitor_training.sh
```

**Displays** (updates every 2 seconds):
- GPU utilization and memory
- Recent training logs
- Process status

**Usage**: Run in a separate terminal while training

---

## 📋 Common Workflows

### Workflow 1: Quick Test (5 minutes)
```bash
# 1. Prepare small dataset
./scripts/prepare_data_quick.sh

# 2. Quick training
./scripts/train_quick.sh

# 3. Evaluate
./scripts/eval.sh checkpoints/quick/final_model.pt
```

### Workflow 2: Full Training (1-2 hours)
```bash
# 1. Prepare full dataset
./scripts/prepare_data_full.sh

# 2. Train with best settings
./scripts/train_full.sh

# 3. Evaluate
./scripts/eval.sh checkpoints/full/best_model.pt
```

### Workflow 3: Complete Pipeline (Automated)
```bash
# Quick pipeline (5 min)
./scripts/pipeline_quick.sh

# OR full pipeline (1-2 hours)
./scripts/pipeline_full.sh
```

### Workflow 4: Hyperparameter Tuning
```bash
# 1. Prepare data once
./scripts/prepare_data_full.sh

# 2. Run grid search (3-6 hours)
./scripts/hyperparameter_search.sh

# 3. Compare all results
./scripts/eval_all.sh checkpoints/hypersearch
```

### Workflow 5: Monitor Long Training
```bash
# Terminal 1: Start training
./scripts/train_full.sh

# Terminal 2: Monitor progress
./scripts/monitor_training.sh
```

---

## 🎯 Quick Reference Table

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `pipeline_quick.sh` | End-to-end test | ~5 min | Quick checkpoint + results |
| `pipeline_full.sh` | Production pipeline | ~1-2 hrs | Best checkpoint + results |
| `prepare_data_quick.sh` | Small dataset | <1 min | 50 pairs |
| `prepare_data_full.sh` | Full dataset | <1 min | 1000 pairs |
| `train_quick.sh` | Fast training | ~3 min | 1 epoch checkpoint |
| `train_full.sh` | Full training | ~1-2 hrs | 3 epoch checkpoint |
| `train_custom.sh` | Custom training | Varies | Custom checkpoint |
| `eval.sh` | Single evaluation | ~1 min | Metrics JSON |
| `eval_all.sh` | Batch evaluation | Varies | All metrics + summary |
| `hyperparameter_search.sh` | Grid search | ~3-6 hrs | Multiple checkpoints |
| `gpu_info.sh` | GPU status | Instant | Info display |
| `monitor_training.sh` | Live monitoring | Continuous | Real-time display |

---

## 💡 Tips & Best Practices

### GPU Memory Management

Check available memory before training:
```bash
./scripts/gpu_info.sh
```

**T4 GPU (15GB) Recommendations**:
- Batch size 4: ~6GB (safe)
- Batch size 8: ~8GB (recommended)
- Batch size 16: ~12GB (near limit)
- Batch size 32: Will OOM

### Training Tips

1. **Start small**: Use `pipeline_quick.sh` to test everything works
2. **Monitor GPU**: Run `gpu_info.sh` to check memory before training
3. **Watch progress**: Use `monitor_training.sh` in a separate terminal
4. **Save logs**: Training logs are saved to `training.log`

### Data Preparation

1. **Quick test**: `prepare_data_quick.sh` for 50 pairs
2. **Production**: `prepare_data_full.sh` for 1000 pairs
3. **Custom**: Use Python script directly for fine control

### Evaluation

1. **Single model**: `eval.sh checkpoints/best_model.pt`
2. **Compare all**: `eval_all.sh checkpoints/`
3. **View results**: Use `cat results/*.json` or `jq`

---

## 🔧 Customization

### Modify Default Batch Size

Edit any training script and change:
```bash
BATCH_SIZE="${4:-8}"  # Change 8 to your preferred default
```

### Add Custom Experiments

Copy and modify `train_custom.sh`:
```bash
cp scripts/train_custom.sh scripts/train_my_experiment.sh
# Edit to add your custom logic
```

### Change Data Paths

All scripts accept path arguments:
```bash
./scripts/train_full.sh /path/to/train.json /path/to/val.json /path/to/output
```

---

## 🐛 Troubleshooting

### "No such file or directory"
```bash
# Make sure you're in project root
cd /home/ubuntu/ragcun

# Check script exists
ls -l scripts/
```

### "Permission denied"
```bash
# Make executable
chmod +x scripts/*.sh
```

### "Training data not found"
```bash
# Prepare data first
./scripts/prepare_data_quick.sh
# OR
./scripts/prepare_data_full.sh
```

### "CUDA out of memory"
```bash
# Use smaller batch size
./scripts/train_full.sh data/processed/train.json data/processed/val.json checkpoints 4
# OR use train_custom.sh
./scripts/train_custom.sh --train_data data/processed/train.json --batch_size 4
```

### Monitor not updating
```bash
# Make sure training is running
ps aux | grep train.py

# Try manual monitoring
tail -f training.log
```

---

## 📚 Related Documentation

- **Python Scripts**: See Python files in `scripts/` directory
- **Training Guide**: See `TRAINING_GUIDE.md`
- **Data Guide**: See `DATA_SUMMARY.md`
- **Directory Structure**: See `DIRECTORY_GUIDE.md`

---

## 🎓 Examples

### Example 1: Quick Test
```bash
./scripts/pipeline_quick.sh
```

### Example 2: Custom Training
```bash
./scripts/prepare_data_full.sh
./scripts/train_custom.sh \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-5 \
    --freeze_layers
./scripts/eval.sh checkpoints/custom/best_model.pt
```

### Example 3: Compare Multiple Runs
```bash
# Train with different settings
./scripts/train_custom.sh --train_data data/processed/train.json --output_dir checkpoints/run1 --lr 1e-5
./scripts/train_custom.sh --train_data data/processed/train.json --output_dir checkpoints/run2 --lr 2e-5
./scripts/train_custom.sh --train_data data/processed/train.json --output_dir checkpoints/run3 --lr 5e-5

# Evaluate all
./scripts/eval_all.sh checkpoints
```

Happy training! 🚀
