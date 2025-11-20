# 🚀 Quick Training Start Guide

## TL;DR - Start Training Now

```bash
# 1. Test training works (1 minute)
python scripts/train/isotropic.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 1 \
    --batch_size 4 \
    --output_dim 128 \
    --output_dir checkpoints/test

# 2. If successful, run full training
python scripts/train/isotropic.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 \
    --batch_size 8 \
    --output_dim 512 \
    --mixed_precision \
    --output_dir checkpoints/full \
    --device cuda
```

---

## ✅ What Was Fixed

| Issue | Status | Impact |
|-------|--------|--------|
| Test failure | ✅ Fixed | Tests passing |
| Mixed precision | ✅ Fixed | 2x faster training |
| Learning rate scheduler | ✅ Fixed | Better convergence |
| Input validation | ✅ Fixed | Clear error messages |
| Training data | ✅ Generated | 43 train / 9 val / 10 test |
| Memory management | ✅ Added | Prevents OOM |

---

## 📊 Training Data Ready

```bash
$ ls -lh data/processed/
-rw-rw-r-- 1 ubuntu ubuntu 4.3K Nov 15 14:45 train.json      # 43 examples
-rw-rw-r-- 1 ubuntu ubuntu  600 Nov 15 14:45 val.json        # 9 examples  
-rw-rw-r-- 1 ubuntu ubuntu  771 Nov 15 14:45 test_eval.json  # 10 examples
```

---

## 🎯 Training Commands

### Quick Test (CPU, 1 min)
```bash
python scripts/train/isotropic.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 1 --batch_size 4 --output_dim 128 \
    --output_dir checkpoints/test --device cpu
```

### Full Training (GPU, ~30 min)
```bash
python scripts/train/isotropic.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 --batch_size 8 --output_dim 512 \
    --mixed_precision --output_dir checkpoints/full \
    --device cuda
```

### Smart Hybrid (Faster, ~15 min)
```bash
python scripts/train/isotropic.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 --batch_size 16 --output_dim 512 \
    --freeze_base --projection_learning_rate 5e-4 \
    --mixed_precision --output_dir checkpoints/hybrid \
    --device cuda
```

---

## 📈 Expected Output

```
✅ Input files validated
Using device: cuda
GPU: NVIDIA A100-SXM4-40GB
Loaded 43 training examples
Loaded 9 training examples
✅ Mixed precision training enabled (FP16)
Starting training...

============================================================
Epoch 1/3
============================================================
Epoch 1: 100%|██████████| 6/6 [00:23<00:00]

Training metrics:
  Total Loss: 2.1234
  Pos Distance (mean): 1.234
  Embedding Std: 0.987

Validation metrics:
  Total Loss: 2.0123
  ✅ Saved best model to checkpoints/full/best_model.pt
```

---

## 🔍 Monitor Training

```bash
# Watch training log
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor checkpoints
ls -lht checkpoints/full/
```

---

## 🎓 After Training

### Evaluate Model
```bash
python scripts/evaluate.py \
    --model_path checkpoints/full/best_model.pt \
    --test_data data/processed/test_eval.json \
    --output_file results/evaluation.json
```

### Use for Retrieval
```python
from ragcun import IsotropicRetriever

retriever = IsotropicRetriever(
    model_path='checkpoints/full/best_model.pt',
    embedding_dim=512
)

retriever.add_documents([
    "Python is a programming language.",
    "Machine learning is AI.",
    "NLP processes human language."
])

results = retriever.retrieve("What is ML?", top_k=3)
for doc, distance in results:
    print(f"[{distance:.3f}] {doc}")
```

---

## ⚠️ Troubleshooting

### Out of Memory
```bash
# Try in order:
--batch_size 4          # Smaller batches
--output_dim 256        # Smaller embeddings  
--freeze_base           # Train projection only
```

### Training Too Slow
```bash
--mixed_precision       # FP16 (2x faster)
--freeze_base           # Train less parameters
--batch_size 16         # Larger batches
```

### Bad Results
```bash
# More data needed (recommend 1000+ examples)
python scripts/prepare_data.py \
    --input_dir data/raw/your_docs/ \
    --generate_pairs --num_pairs 10000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed/more
```

---

## 📚 More Information

- **Full Details**: See `TRAINING_READINESS_FIXES.md`
- **Training Guide**: See `docs/TRAINING_GUIDE.md`
- **Scripts README**: See `scripts/README.md`

---

## ✅ Pre-Flight Checklist

- [x] Code fixes applied
- [x] Tests passing
- [x] Training data ready (43 examples)
- [ ] GPU available (recommended)
- [ ] Enough disk space (1GB+)

---

**Status**: 🟢 **READY TO TRAIN**

**Next**: Run the quick test command above ↑

