# Script Workflow Guide

The scripts are numbered to show the typical order of use.

## 🚀 Quick Reference

```
0️⃣  Start Here
1️⃣-3️⃣  Quick Test Path (5 minutes)
4️⃣-5️⃣  Production Path (1-2 hours)
6️⃣-9️⃣  Advanced Tools
98-99  Complete Pipelines (automated)
```

---

## 📋 Script Numbers Explained

### 0️⃣ Setup & Information

**`0_gpu_info.sh`** - Check GPU status first
```bash
./scripts/0_gpu_info.sh
```
**Purpose**: Check GPU availability, memory, and get batch size recommendations

---

## 🎯 Beginner Path: Quick Test (5 minutes)

Follow these 3 steps in order:

### 1️⃣ Prepare Quick Data
**`1_prepare_data_quick.sh`** - Generate 50 test pairs
```bash
./scripts/1_prepare_data_quick.sh
```
**Creates**: `data/processed/quick/` with train/val/test files

### 2️⃣ Train Quick Model
**`2_train_quick.sh`** - Train for 1 epoch (~3 min)
```bash
./scripts/2_train_quick.sh
```
**Creates**: `checkpoints/quick/final_model.pt`

### 3️⃣ Evaluate Model
**`3_eval.sh`** - Test the model
```bash
./scripts/3_eval.sh checkpoints/quick/final_model.pt
```
**Creates**: `results/eval_results.json`

---

## 🚀 Production Path (1-2 hours)

After testing with the quick path:

### 4️⃣ Prepare Full Dataset
**`4_prepare_data_full.sh`** - Generate 1000 pairs
```bash
./scripts/4_prepare_data_full.sh
```
**Creates**: `data/processed/` with 800 train, 100 val, 100 test pairs

### 5️⃣ Train Full Model
**`5_train_full.sh`** - Train for 3 epochs with best settings
```bash
./scripts/5_train_full.sh
```
**Creates**: `checkpoints/full/best_model.pt` and `final_model.pt`

### (3️⃣) Evaluate (reuse)
**`3_eval.sh`** - Evaluate the trained model
```bash
./scripts/3_eval.sh checkpoints/full/best_model.pt
```

---

## 🔧 Advanced Tools

### 6️⃣ Custom Training
**`6_train_custom.sh`** - Full control over hyperparameters
```bash
./scripts/6_train_custom.sh \
    --train_data data/processed/train.json \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-5 \
    --freeze_layers
```

### 7️⃣ Batch Evaluation
**`7_eval_all.sh`** - Evaluate all checkpoints
```bash
./scripts/7_eval_all.sh checkpoints/
```
**Use case**: Compare multiple trained models

### 8️⃣ Monitor Training
**`8_monitor_training.sh`** - Real-time training monitor
```bash
# Terminal 1: Start training
./scripts/5_train_full.sh

# Terminal 2: Monitor progress
./scripts/8_monitor_training.sh
```
**Updates**: Every 2 seconds with GPU stats and logs

### 9️⃣ Hyperparameter Search
**`9_hyperparameter_search.sh`** - Automated grid search
```bash
./scripts/9_hyperparameter_search.sh
```
**Tests**: Different learning rates, batch sizes, isotropy weights
**Time**: 3-6 hours

---

## 🎁 Complete Pipelines (Automated)

### 99 - Quick Pipeline (5 minutes)
**`99_pipeline_quick.sh`** - Complete test workflow
```bash
./scripts/99_pipeline_quick.sh
```
**Runs**: 1 → 2 → 3 automatically

### 98 - Full Pipeline (1-2 hours)
**`98_pipeline_full.sh`** - Complete production workflow
```bash
./scripts/98_pipeline_full.sh
```
**Runs**: 4 → 5 → 3 automatically

---

## 📊 Typical Workflows

### Workflow 1: First Time User
```bash
# Step 0: Check GPU
./scripts/0_gpu_info.sh

# Steps 1-3: Quick test
./scripts/1_prepare_data_quick.sh
./scripts/2_train_quick.sh
./scripts/3_eval.sh checkpoints/quick/final_model.pt

# OR use automated pipeline
./scripts/99_pipeline_quick.sh
```

### Workflow 2: Production Training
```bash
# Steps 4-5: Full training
./scripts/4_prepare_data_full.sh
./scripts/5_train_full.sh
./scripts/3_eval.sh checkpoints/full/best_model.pt

# OR use automated pipeline
./scripts/98_pipeline_full.sh
```

### Workflow 3: Experiment & Compare
```bash
# Prepare data once
./scripts/4_prepare_data_full.sh

# Run experiments with different settings
./scripts/6_train_custom.sh --train_data data/processed/train.json --lr 1e-5 --output_dir checkpoints/exp1
./scripts/6_train_custom.sh --train_data data/processed/train.json --lr 2e-5 --output_dir checkpoints/exp2
./scripts/6_train_custom.sh --train_data data/processed/train.json --lr 5e-5 --output_dir checkpoints/exp3

# Compare all results
./scripts/7_eval_all.sh checkpoints/
```

### Workflow 4: Monitored Training
```bash
# Terminal 1: Check GPU first
./scripts/0_gpu_info.sh

# Terminal 1: Start training
./scripts/5_train_full.sh

# Terminal 2: Monitor
./scripts/8_monitor_training.sh
```

### Workflow 5: Hyperparameter Optimization
```bash
# Prepare data
./scripts/4_prepare_data_full.sh

# Run automated grid search
./scripts/9_hyperparameter_search.sh

# View results
./scripts/7_eval_all.sh checkpoints/hypersearch/
```

---

## 🎯 Decision Tree

```
START
  |
  ├─ Never used before?
  │   └─ Run: ./scripts/99_pipeline_quick.sh
  │
  ├─ Quick test successful?
  │   └─ Run: ./scripts/98_pipeline_full.sh
  │
  ├─ Want custom settings?
  │   └─ Use: ./scripts/6_train_custom.sh
  │
  ├─ Optimize hyperparameters?
  │   └─ Run: ./scripts/9_hyperparameter_search.sh
  │
  └─ Compare multiple models?
      └─ Use: ./scripts/7_eval_all.sh
```

---

## 📖 Script Reference

| # | Script | Purpose | Time |
|---|--------|---------|------|
| 0 | `0_gpu_info.sh` | Check GPU status | Instant |
| 1 | `1_prepare_data_quick.sh` | 50 training pairs | <1 min |
| 2 | `2_train_quick.sh` | Quick training test | ~3 min |
| 3 | `3_eval.sh` | Evaluate any model | ~1 min |
| 4 | `4_prepare_data_full.sh` | 1000 training pairs | <1 min |
| 5 | `5_train_full.sh` | Production training | ~1-2 hrs |
| 6 | `6_train_custom.sh` | Custom hyperparameters | Varies |
| 7 | `7_eval_all.sh` | Batch evaluation | Varies |
| 8 | `8_monitor_training.sh` | Real-time monitor | Continuous |
| 9 | `9_hyperparameter_search.sh` | Grid search | ~3-6 hrs |
| 98 | `98_pipeline_full.sh` | Full pipeline | ~1-2 hrs |
| 99 | `99_pipeline_quick.sh` | Quick pipeline | ~5 min |

---

## 💡 Tips

### When to Use Each Script

**Start with 99**: If you're new, start with the complete quick pipeline
```bash
./scripts/99_pipeline_quick.sh
```

**Progress to 98**: When ready for production, use the full pipeline
```bash
./scripts/98_pipeline_full.sh
```

**Use 0 first**: Always check GPU before heavy training
```bash
./scripts/0_gpu_info.sh
```

**Use 8 during 5**: Monitor long training sessions
```bash
# Terminal 1
./scripts/5_train_full.sh

# Terminal 2
./scripts/8_monitor_training.sh
```

### Recommended Learning Path

1. **Day 1**: Run `99_pipeline_quick.sh` to understand the workflow
2. **Day 2**: Run `98_pipeline_full.sh` for real training
3. **Day 3**: Experiment with `6_train_custom.sh`
4. **Day 4**: Optimize with `9_hyperparameter_search.sh`

---

## 🚦 Status Indicators

When you run scripts, look for these indicators:

- ✅ Success
- ⚠️  Warning (non-critical)
- ❌ Error (critical)

Example output:
```bash
$ ./scripts/1_prepare_data_quick.sh
============================================
Quick Data Preparation (50 pairs)
============================================

Configuration:
  Input: data/raw/sample_docs.txt
  Output directory: data/processed
  Number of pairs: 50

Preparing data...
✅ Quick data preparation complete!
```

---

## 🎓 Next Steps

After running the basic workflows:

1. **Read detailed docs**:
   - `scripts/README.md` - Full script reference
   - `TRAINING_GUIDE.md` - Training deep dive
   - `DATA_SUMMARY.md` - Data documentation

2. **Try examples**:
   - `examples/retrieval_example.py` - Use your trained model

3. **Add your data**:
   - Put documents in `data/raw/`
   - Run `./scripts/4_prepare_data_full.sh`

4. **Explore notebooks**:
   - `notebooks/` - Jupyter notebooks for Colab

---

## 🎉 Summary

**For beginners**: Run scripts 99 → 98 → done!

**For experimentation**: Use scripts 1-9 individually

**For production**: Use script 98 or 5

The numbering makes the learning curve smoother! 🚀
