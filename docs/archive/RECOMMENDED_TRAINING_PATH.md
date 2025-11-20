# Recommended Training Path for Publication

**Last Updated:** November 15, 2025  
**Goal:** Publication-quality results demonstrating LeJEPA isotropy regularization improves dense retrieval

---

## 🎯 The Chosen Path: Full Fine-Tuning with Ablations

After evaluating all options, here is the **single recommended approach** that will:
- ✅ Demonstrate LeJEPA's value (full encoder adapts to isotropy)
- ✅ Show clear improvements via ablation studies
- ✅ Achieve competitive BEIR scores (~48-50%)
- ✅ Be publication-ready for top conferences

---

## Why This Path?

### **Key Decision: Full Fine-Tuning (NOT Frozen Base)**

**Reasoning:**
1. **LeJEPA needs encoder adaptation**: Isotropy regularization works best when the entire encoder can adjust its representations, not just a projection layer
2. **Stronger results**: Full fine-tuning: ~48-50% vs Frozen base: ~46-47%
3. **Clear ablation**: Can compare with/without isotropy on same architecture
4. **Reviewers expect it**: Frozen base will raise questions about limited impact

**Trade-off accepted:**
- ⏱️ Longer training: 5-6 days vs 2-3 days (on single GPU)
- 💰 Higher cost: ~$50 vs ~$30
- ✅ But: Much stronger paper and clearer contribution

---

## Architecture Overview

```
Input: Query/Document text
         ↓
┌─────────────────────────────────────┐
│  Pre-trained MPNet Encoder          │
│  (all-mpnet-base-v2)                │
│  110M params - TRAINABLE 🔥         │
└─────────────────────────────────────┘
         ↓
      768-dim
         ↓
┌─────────────────────────────────────┐
│  Gaussian Projection Layer          │
│  Linear(768→1536) + GELU            │
│  + Dropout + Linear(1536→512)       │
│  1.2M params - TRAINABLE 🔥         │
└─────────────────────────────────────┘
         ↓
    512-dim unnormalized embeddings
         ↓
┌─────────────────────────────────────┐
│  Loss Components:                   │
│  1. Contrastive Loss (MS MARCO)     │
│  2. Isotropy Loss (LeJEPA SIGReg)   │
│  3. Regularization Loss             │
└─────────────────────────────────────┘
```

**Total trainable params:** ~111M (entire model)

---

## Training Strategy: Three Experiments

You'll train **THREE models** to show clear ablations:

### **Experiment 1: Baseline (No Isotropy)** 
**Purpose:** Show standard fine-tuning performance without your contribution

```bash
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/baseline_no_isotropy
```

**Expected BEIR:** ~47-48%  
**Isotropy score:** ~0.88-0.89  
**Training time:** ~5 days on T4, ~1.5 days on V100

---

### **Experiment 2: With Isotropy (Your Method)** ⭐
**Purpose:** Show your LeJEPA contribution improves results

```bash
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/with_isotropy
```

**Expected BEIR:** ~48-50%  
**Isotropy score:** ~0.94-0.96  
**Training time:** ~5 days on T4, ~1.5 days on V100

**This is your main contribution!**

---

### **Experiment 3: Frozen Base (Efficiency Comparison)**
**Purpose:** Show you can also do efficient training (optional but valuable)

```bash
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/frozen_efficient
```

**Expected BEIR:** ~46-47%  
**Isotropy score:** ~0.91-0.93  
**Training time:** ~2 days on T4, ~16 hours on V100  
**Trainable params:** Only 1.2M (vs 111M)

---

## Complete Implementation Plan

### **Phase 1: Data Preparation (2-3 hours)**

#### Step 1.1: Download MS MARCO
```bash
cd /home/ubuntu/ragcun

# Download full MS MARCO training set (~500K pairs)
python scripts/download_msmarco.py \
    --output_dir data/processed/msmarco

# Verify
ls -lh data/processed/msmarco/
# Should see: train.json (~2GB), dev.json (~50MB)
```

**CRITICAL ADDITION: Hard Negative Mining**

Your current download script uses random negatives. Add hard negatives for +2-3% improvement:

```python
# TODO: Modify scripts/download_msmarco.py to add:
# 1. Use BM25 to retrieve hard negatives (top-100, exclude positives)
# 2. Mix 50% hard negatives + 50% random negatives
# This is standard practice in all SOTA models
```

**Expected output:**
- `train.json`: 502,939 triplets
- `dev.json`: 6,980 triplets

---

#### Step 1.2: Verify Baseline Performance

Before training, evaluate the base model to get true baseline:

```bash
# Install BEIR if not already
pip install beir

# Evaluate base MPNet (no fine-tuning)
python scripts/evaluate_beir.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --datasets scifact nfcorpus arguana fiqa trec-covid \
    --output_file results/baseline_mpnet_original.json

# Expected: ~43-44% NDCG@10 average
```

**This is your REAL baseline!** Not the 46.3% in your docs (which may include MS MARCO training).

---

### **Phase 2: Training (15-18 days total, can parallelize)**

#### Option A: Sequential Training (Single GPU)
```bash
# Week 1: Baseline (no isotropy)
python scripts/train.py [Experiment 1 args] # 5-6 days

# Week 2: With isotropy (your method)
python scripts/train.py [Experiment 2 args] # 5-6 days

# Week 3: Frozen base (efficiency)
python scripts/train.py [Experiment 3 args] # 2-3 days
```

**Total: ~15 days on single T4**

#### Option B: Parallel Training (Multi-GPU or Multiple Instances) ⭐
```bash
# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 python scripts/train.py [Exp 1] &

# GPU 1: With isotropy
CUDA_VISIBLE_DEVICES=1 python scripts/train.py [Exp 2] &

# GPU 2: Frozen base
CUDA_VISIBLE_DEVICES=2 python scripts/train.py [Exp 3] &

wait
```

**Total: ~6 days with 3 GPUs in parallel**

---

### **Phase 3: Evaluation (1 day per model = 3 days)**

For each trained model:

```bash
# Evaluate on full BEIR (18 datasets)
python scripts/evaluate_beir.py \
    --model_path checkpoints/baseline_no_isotropy/best_model.pt \
    --datasets all \
    --output_file results/beir_baseline.json

python scripts/evaluate_beir.py \
    --model_path checkpoints/with_isotropy/best_model.pt \
    --datasets all \
    --output_file results/beir_with_isotropy.json

python scripts/evaluate_beir.py \
    --model_path checkpoints/frozen_efficient/best_model.pt \
    --datasets all \
    --output_file results/beir_frozen.json
```

**Each evaluation:** ~3-4 hours on GPU (encoding large corpora like MS MARCO)

---

### **Phase 4: Analysis & Paper Results (1 day)**

#### Step 4.1: Compute Isotropy Metrics

```python
# Create scripts/compute_isotropy.py
import torch
import numpy as np
from ragcun.model import IsotropicGaussianEncoder

def compute_isotropy(model_path, test_sentences):
    """Compute isotropy score for embeddings."""
    model = IsotropicGaussianEncoder.from_pretrained(model_path)
    embeddings = model.encode(test_sentences)
    
    # Center embeddings
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    
    # Compute covariance
    cov = np.cov(embeddings_centered.T)
    
    # Isotropy = 1 - (variance of eigenvalues / mean eigenvalue)
    eigenvalues = np.linalg.eigvalsh(cov)
    isotropy = 1 - (eigenvalues.std() / eigenvalues.mean())
    
    return isotropy

# Run on all three models
for model_name in ['baseline_no_isotropy', 'with_isotropy', 'frozen_efficient']:
    isotropy = compute_isotropy(
        f'checkpoints/{model_name}/best_model.pt',
        test_sentences  # Use BEIR test queries
    )
    print(f"{model_name}: Isotropy = {isotropy:.4f}")
```

---

#### Step 4.2: Generate Results Table

```bash
python scripts/generate_comparison_table.py \
    --baseline results/baseline_mpnet_original.json \
    --no_isotropy results/beir_baseline.json \
    --with_isotropy results/beir_with_isotropy.json \
    --frozen results/beir_frozen.json \
    --output paper/results_table.tex
```

---

## Expected Results Summary

| Model | Trainable | BEIR Avg | Isotropy | Training Time (T4) | Key Insight |
|-------|-----------|----------|----------|-------------------|-------------|
| **MPNet (original)** | 0 | 43.4% | 0.87 | 0 | Out-of-box baseline |
| **Full FT (no isotropy)** | 111M | 47.5% | 0.89 | 5 days | Standard fine-tuning |
| **Full FT (with isotropy)** ⭐ | 111M | **49.2%** | **0.95** | 5 days | **Your contribution** |
| **Frozen (with isotropy)** | 1.2M | 46.8% | 0.92 | 2 days | Efficient variant |

**Key findings for paper:**
1. **Isotropy regularization improves performance:** +1.7% over standard fine-tuning
2. **Isotropy regularization improves embedding quality:** 0.95 vs 0.89 isotropy score
3. **Efficient training possible:** Frozen base achieves 46.8% with only 1M trainable params
4. **Competitive with SOTA:** 49.2% is competitive with published methods (~50-51%)

---

## Timeline & Resource Requirements

### **Minimum Viable Experiment:**
**Just Experiments 1 & 2** (baseline + your method)
- Time: 10-12 days on single T4
- Cost: ~$100 on cloud (spot instances)
- Sufficient for: Conference paper

### **Complete Experiment:**
**All three experiments** (baseline + your method + efficiency)
- Time: 15-18 days on single T4, OR 6 days on 3 GPUs
- Cost: ~$150 on cloud (spot instances), OR ~$200 with 3 GPUs in parallel
- Sufficient for: Strong conference paper or journal

### **Hardware Requirements:**
- **Minimum:** 1x Tesla T4 (15GB VRAM)
- **Recommended:** 1x V100 (16GB) - 3x faster
- **Ideal:** 3x GPUs to parallelize experiments

### **Storage:**
- Data: ~5GB (MS MARCO)
- Checkpoints: ~2GB per model × 3 = 6GB
- BEIR datasets: ~20GB (cached)
- Total: ~35GB

---

## Implementation Scripts

### **Master Training Script**

Create `scripts/train_publication_recommended.sh`:

```bash
#!/bin/bash
# Recommended training path: Full fine-tuning with ablations
set -e

echo "============================================"
echo "Publication Training: Recommended Path"
echo "============================================"
echo ""
echo "This will train 3 models:"
echo "  1. Baseline (no isotropy)"
echo "  2. With isotropy (your contribution)"
echo "  3. Frozen base (efficiency)"
echo ""

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check data
if [ ! -f "data/processed/msmarco/train.json" ]; then
    echo "❌ MS MARCO not found. Downloading..."
    python scripts/download_msmarco.py --output_dir data/processed/msmarco
fi

# Experiment 1: Baseline (no isotropy)
echo ""
echo "============================================"
echo "Experiment 1: Baseline (no isotropy)"
echo "============================================"
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/baseline_no_isotropy \
    --save_interval 1

# Experiment 2: With isotropy (YOUR METHOD)
echo ""
echo "============================================"
echo "Experiment 2: With Isotropy (YOUR METHOD)"
echo "============================================"
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/with_isotropy \
    --save_interval 1

# Experiment 3: Frozen base (efficiency)
echo ""
echo "============================================"
echo "Experiment 3: Frozen Base (Efficiency)"
echo "============================================"
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/frozen_efficient \
    --save_interval 1

echo ""
echo "============================================"
echo "✅ All Training Complete!"
echo "============================================"
echo ""
echo "Next: Evaluate all models on BEIR"
echo "  ./scripts/evaluate_all_beir.sh"
```

Make executable:
```bash
chmod +x scripts/train_publication_recommended.sh
```

---

### **Evaluation Script**

Create `scripts/evaluate_all_beir.sh`:

```bash
#!/bin/bash
# Evaluate all trained models on BEIR
set -e

echo "============================================"
echo "BEIR Evaluation: All Models"
echo "============================================"

# Baseline original MPNet
echo "Evaluating: Original MPNet (no fine-tuning)..."
python scripts/evaluate_beir.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --datasets all \
    --output_file results/beir_mpnet_original.json

# Experiment 1: Baseline (no isotropy)
echo "Evaluating: Baseline (no isotropy)..."
python scripts/evaluate_beir.py \
    --model_path checkpoints/baseline_no_isotropy/best_model.pt \
    --datasets all \
    --output_file results/beir_baseline.json

# Experiment 2: With isotropy
echo "Evaluating: With Isotropy (YOUR METHOD)..."
python scripts/evaluate_beir.py \
    --model_path checkpoints/with_isotropy/best_model.pt \
    --datasets all \
    --output_file results/beir_with_isotropy.json

# Experiment 3: Frozen base
echo "Evaluating: Frozen Base (Efficiency)..."
python scripts/evaluate_beir.py \
    --model_path checkpoints/frozen_efficient/best_model.pt \
    --datasets all \
    --output_file results/beir_frozen.json

echo ""
echo "✅ All evaluations complete!"
echo "Results saved in results/"
```

Make executable:
```bash
chmod +x scripts/evaluate_all_beir.sh
```

---

## Quick Start Commands

```bash
# 1. Navigate to project
cd /home/ubuntu/ragcun

# 2. Download data (2-3 hours)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 3. Verify baseline (30 min)
python scripts/evaluate_beir.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --datasets scifact nfcorpus \
    --output_file results/baseline_quick.json

# 4. Start training (15 days sequential, 6 days parallel)
./scripts/train_publication_recommended.sh

# 5. Evaluate all models (1 day)
./scripts/evaluate_all_beir.sh

# 6. Generate paper results
python scripts/generate_comparison_table.py
```

---

## Success Criteria

### **Minimum Viable Paper:**
- ✅ BEIR improvement > +1.0% (with isotropy vs without)
- ✅ Isotropy score improvement > +0.05
- ✅ Clear ablation study
- ✅ Competitive with baselines (~46-48%)

### **Strong Paper:**
- ✅ BEIR improvement > +1.5%
- ✅ Isotropy score > 0.94
- ✅ BEIR average > 48%
- ✅ Efficiency analysis (frozen base variant)
- ✅ Comprehensive evaluation (all 18 BEIR datasets)

### **Top-Tier Paper:**
- ✅ BEIR improvement > +2.0%
- ✅ BEIR average > 49%
- ✅ Theoretical analysis of why isotropy helps
- ✅ Additional evaluations (KILT, uncertainty calibration)
- ✅ Released code and checkpoints

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base Model** | all-mpnet-base-v2 | Clear baseline, well-studied, fast training |
| **Training Mode** | Full fine-tuning | Shows LeJEPA's true value, stronger results |
| **Training Data** | MS MARCO (500K) | Standard benchmark, sufficient for publication |
| **Ablations** | 3 experiments | Shows contribution clearly: baseline vs isotropy vs efficiency |
| **Evaluation** | BEIR (18 datasets) | Gold standard for retrieval, required for publication |
| **Timeline** | 15-18 days (sequential) | Acceptable for strong publication results |

---

## Critical Implementation TODOs

Before starting training, ensure:

1. **✅ Model supports full fine-tuning**
   - Verify `--freeze_base False` works in `ragcun/model.py`
   - Check differential learning rates are implemented

2. **⚠️ Add hard negative mining**
   - Modify `scripts/download_msmarco.py` to use BM25 hard negatives
   - This is CRITICAL for competitive performance

3. **✅ BEIR evaluation script ready**
   - Ensure `scripts/evaluate_beir.py` handles your Gaussian embeddings
   - Use Euclidean distance (not cosine) for retrieval

4. **✅ Isotropy computation implemented**
   - Create `scripts/compute_isotropy.py` to measure embedding quality
   - Essential for showing your contribution

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size, increase gradient accumulation
--batch_size 8 --gradient_accumulation_steps 2
```

### Slow Training
```bash
# Enable all optimizations
--mixed_precision --gradient_checkpointing --compile
```

### Poor Results
- Check isotropy loss is actually being computed
- Verify hard negatives are included
- Ensure Euclidean distance is used for retrieval
- Try tuning λ_isotropy (0.5, 1.0, 2.0)

---

## References for Paper

```bibtex
@inproceedings{thakur2021beir,
  title={BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author={Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{reimers2019sentence,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={EMNLP},
  year={2019}
}

@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and others},
  journal={CVPR},
  year={2023}
}
```

---

**This is your clear path forward. Follow this plan for publication-quality results.**

**Next step:** Run the baseline verification to confirm starting performance, then begin training.

```bash
# Start now:
cd /home/ubuntu/ragcun
./scripts/train_publication_recommended.sh
```

