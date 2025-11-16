# Most Promising Hyperparameters for Isotropy Regularization

## Current Results Analysis
- **Pure isotropy:** -4.1% vs baseline (losing semantic knowledge)
- **Embedding std:** 0.014 (very small, embeddings too compact)
- **Loss:** Near zero (weak gradients, minimal learning)
- **Key issue:** No contrastive loss = semantic knowledge degradation

---

## 🎯 Top 3 Most Promising Configurations

### 1. **Small Contrastive Component** ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**Why:** Pure isotropy loses semantics. Small contrastive loss preserves knowledge while still regularizing.

```bash
--lambda_contrastive 0.1 \
--lambda_isotropy 1.0 \
--lambda_reg 0.0 \
--margin 0.1 \
--projection_learning_rate 5e-4 \
--epochs 2
```

**Expected impact:** 
- Preserves baseline performance (or better)
- Still enforces isotropy
- Balances semantics + regularization

**Rationale:**
- λ_contrastive=0.1 is 10% of full contrastive, enough to preserve semantics
- margin=0.1 is smaller, works better with isotropy
- Lower LR (5e-4) = more stable training

---

### 2. **Balanced Contrastive + Isotropy** ⭐⭐⭐⭐

**Why:** More contrastive signal to ensure semantics are preserved.

```bash
--lambda_contrastive 0.5 \
--lambda_isotropy 1.0 \
--lambda_reg 0.0 \
--margin 0.1 \
--projection_learning_rate 5e-4 \
--epochs 2
```

**Expected impact:**
- Strong semantic preservation
- Moderate isotropy regularization
- Likely matches or beats baseline

**Trade-off:** Less isotropy regularization, but better retrieval performance

---

### 3. **Higher Isotropy Weight (if contrastive works)** ⭐⭐⭐

**Why:** Once semantics are preserved, we can push isotropy harder.

```bash
--lambda_contrastive 0.1 \
--lambda_isotropy 2.0 \
--lambda_reg 0.0 \
--margin 0.1 \
--projection_learning_rate 5e-4 \
--epochs 2
```

**Expected impact:**
- Strong isotropy regularization
- Still preserves semantics (via contrastive)
- Better embedding distribution

**Use this AFTER #1 or #2 works**

---

## 🔧 Supporting Hyperparameters (Use with above)

### Learning Rate (Critical)
```bash
--projection_learning_rate 5e-4  # Lower = more stable
```
**Why:** Current 1e-3 might be too high, causing instability. Lower LR preserves base embeddings better.

### Epochs
```bash
--epochs 2  # or 3
```
**Why:** 1 epoch might not be enough. Give it more time to learn the balance.

### Margin (Important)
```bash
--margin 0.1  # Smaller than default 1.0
```
**Why:** Smaller margin works better with isotropy. Large margin (1.0) might be too aggressive.

### Training Data Size
```bash
--train_data data/processed/msmarco/train.json  # 48K examples
```
**Why:** More data helps preserve semantics. Current 10K might be too small.

---

## 📊 Recommended Experiment Order

### Experiment 1: Small Contrastive (MUST TRY FIRST)
```bash
python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --freeze_base \
    --lambda_contrastive 0.1 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --margin 0.1 \
    --projection_learning_rate 5e-4 \
    --epochs 2 \
    --batch_size 16 \
    --output_dir "checkpoints/isotropy_contrastive_01"
```

**Expected:** Matches or beats baseline (+0% to +2%)

---

### Experiment 2: Balanced (if #1 works)
```bash
python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --freeze_base \
    --lambda_contrastive 0.5 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --margin 0.1 \
    --projection_learning_rate 5e-4 \
    --epochs 2 \
    --batch_size 16 \
    --output_dir "checkpoints/isotropy_contrastive_05"
```

**Expected:** Matches baseline, better semantic preservation

---

### Experiment 3: Scale Up (if #1 or #2 works)
```bash
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --freeze_base \
    --lambda_contrastive 0.1 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --margin 0.1 \
    --projection_learning_rate 5e-4 \
    --epochs 3 \
    --batch_size 16 \
    --output_dir "checkpoints/isotropy_contrastive_01_48k"
```

**Expected:** Better than baseline (+1% to +3%)

---

## 🚫 Hyperparameters to AVOID (Based on Results)

1. **Pure isotropy (λ_contrastive=0.0)** ❌
   - Already tested: -4.1% vs baseline
   - Loses semantic knowledge

2. **Full fine-tuning (freeze_base=False)** ❌
   - Already tested: Catastrophic forgetting
   - Base model loses pre-trained knowledge

3. **High learning rate (1e-3)** ⚠️
   - Might be too aggressive
   - Try 5e-4 first

4. **Large margin (1.0)** ⚠️
   - Might be too aggressive with isotropy
   - Try 0.1 first

5. **Variance regularization (λ_reg > 0)** ⚠️
   - Not needed with scale-invariant isotropy
   - Keep at 0.0

---

## 📈 Expected Results Progression

| Configuration | Expected NDCG@10 | vs Baseline |
|---------------|------------------|-------------|
| Pure Isotropy (current) | 0.4437 | -4.1% ❌ |
| Small Contrastive (0.1) | 0.46-0.48 | 0% to +2% ✅ |
| Balanced (0.5) | 0.47-0.49 | +1% to +3% ✅ |
| Scaled Up (48K data) | 0.48-0.50 | +2% to +5% ✅ |

---

## 🎯 Key Insight

**The critical missing piece is contrastive loss.**

Pure isotropy alone cannot preserve semantic knowledge. A small contrastive component (10-50% of full weight) should:
1. Preserve semantic relationships (via contrastive)
2. Still enforce isotropy (via isotropy loss)
3. Balance both objectives

**Start with λ_contrastive=0.1, λ_isotropy=1.0, margin=0.1, LR=5e-4, epochs=2**

