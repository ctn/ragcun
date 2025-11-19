# ASYMMETRIC PROJECTIONS FAILURE ANALYSIS

**Date:** 2025-11-18  
**Model:** AsymmetricProjectionModel  
**Status:** ❌ FAILED (Initial), 🔧 FIXING (Retraining in progress)

---

## 1. PROBLEM SUMMARY

The Asymmetric Projection model showed **catastrophic failure** with **-59.2% performance drop** compared to baseline.

| Model              | Avg NDCG@10 | vs Baseline | Status          |
|--------------------|-------------|-------------|-----------------|
| Baseline (MPNet)   | 0.4628      | -           | ✅ Reference    |
| ISO15_PRED12       | 0.4759      | +2.83%      | ✅ Works!       |
| ResPred (Epoch 1)  | 0.4416      | -4.59%      | ❌ Failed       |
| **Asymmetric (E3)**| **0.1888**  | **-59.20%** | **❌❌❌ DISASTER** |

---

## 2. ROOT CAUSE ANALYSIS

### 🔍 Discovery Process

1. **Initial Suspicion:** Device mismatch (CPU vs GPU tensors)
   - **Result:** Fixed, but no improvement

2. **Second Suspicion:** Wrong similarity metric (Euclidean vs Dot Product)
   - **Result:** Switched to dot product, but STILL bad (0.1974 NDCG@10)

3. **Breakthrough:** Examined training statistics

### 📊 The Smoking Gun

Training statistics revealed **embedding magnitude explosion**:

```
Epoch 1:
- Positive similarity: 32.25 (train), 64.65 (val)
- Negative similarity: -0.50 (train), 0.32 (val)
- Query std: 0.067, Doc std: 0.059

Epoch 2:
- Positive similarity: 88.00 (train), 134.36 (val)
- Negative similarity: -0.52 (train), -0.10 (val)
- Query std: 0.109, Doc std: 0.096

Epoch 3:
- Positive similarity: 163.60 (train), 228.52 (val)  ← EXPLODING!
- Negative similarity: 0.55 (train), 1.24 (val)
- Query std: 0.146, Doc std: 0.129
```

**Analysis:**
- Positive similarities increased by **7x** over 3 epochs (32 → 228)
- Embedding standard deviations increased by **2.5x** (0.06 → 0.17)
- The model learned to **maximize dot product by increasing magnitude**, not by learning semantic relationships!

---

## 3. WHY THIS HAPPENED

### The Failure Mode: "Cheating with Magnitude"

**Original Loss Function:**
```python
# BAD: Raw dot product allows magnitude exploitation
sim_matrix = query_emb @ pos_emb.T / temperature
contrastive_loss = F.cross_entropy(sim_matrix, labels)
```

**The Problem:**
- InfoNCE loss with **raw dot product** on **unnormalized embeddings** allows the model to "cheat"
- Instead of learning meaningful semantic distinctions, it just makes embeddings larger
- Larger embeddings → Higher dot products → Lower loss
- This is a **local optimum** that doesn't generalize to retrieval!

**Mathematical Explanation:**
```
Dot product: a · b = ||a|| ||b|| cos(θ)
             ↑       ↑       ↑
             magnitude × cos(angle)
```

With unnormalized embeddings, the model can increase similarity by:
1. **Learning good angles** (semantic meaning) ← What we want
2. **Increasing magnitude** (cheating) ← What actually happened

---

## 4. THE FIX

### ✅ Use Cosine Similarity

**Fixed Loss Function:**
```python
# GOOD: Normalize before computing similarity
query_emb_norm = F.normalize(query_emb, p=2, dim=1)
pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)
sim_matrix = query_emb_norm @ pos_emb_norm.T / temperature
contrastive_loss = F.cross_entropy(sim_matrix, labels)
```

**Why This Works:**
- Cosine similarity: `cos(θ) = (a · b) / (||a|| ||b||)`
- Normalization makes ||a|| = ||b|| = 1
- Now the model **must** learn good angles (semantic relationships)
- Magnitude is fixed at 1, so it can't cheat!

### 🔧 Changes Made

1. **Training Script** (`train_asymmetric.py`):
   - Line 84-89: Normalize embeddings before contrastive loss
   - Now uses cosine similarity instead of raw dot product

2. **Evaluation Script** (`eval_asymmetric_quick.py`):
   - Line 67-69: Normalize embeddings before computing retrieval scores
   - Ensures train/eval consistency

3. **Model** (`asymmetric_model.py`):
   - Line 116-118: Added device handling for eval mode
   - Ensures base embeddings are on correct device

---

## 5. EXPECTED OUTCOME

With cosine similarity:
- **Embedding magnitudes will stay bounded** (normalized to 1)
- **Model must learn semantic relationships** (can't cheat with magnitude)
- **Similarities will be in [-1, 1] range** (cosine values)
- **Performance should improve significantly**

### 🎯 Success Criteria

**Minimum Goal:** Beat baseline (0.4628 NDCG@10)  
**Target Goal:** Match or beat ISO15_PRED12 (0.4759 NDCG@10)  
**Stretch Goal:** > 0.48 (+3.5% over baseline)

---

## 6. TRAINING STATUS

**Current Run:**
- Started: 2025-11-18 15:28 UTC
- Model: AsymmetricProjectionModel (with cosine similarity fix)
- Dataset: MS MARCO Smoke (10K examples)
- Epochs: 3
- Batch size: 64
- Lambda values: contrastive=1.0, isotropy=1.0

**What to Monitor:**
1. **Positive similarities:** Should stay in [0.5, 1.0] range (cosine values)
2. **Embedding std:** Should stay low (~0.05-0.10, not explode)
3. **Accuracy:** Should reach ~90% (as before)
4. **Validation loss:** Should decrease smoothly

**Checkpoint:**
- Location: `checkpoints/asymmetric_smoke_*/best_model.pt`
- Will evaluate on scifact + nfcorpus when training completes

---

## 7. KEY LESSONS

### 🎓 Lesson 1: Unnormalized Contrastive Learning is Dangerous

**Never use raw dot product with unnormalized embeddings in contrastive loss.**  
Always use cosine similarity (normalize first) or explicitly normalize embeddings.

### 🎓 Lesson 2: Monitor Training Statistics

The explosion was visible in training stats from Epoch 1:
- Positive similarities growing rapidly
- Embedding standard deviations increasing

**Action:** Add monitoring for these metrics in all future training runs.

### 🎓 Lesson 3: Similarity Metric Consistency

**Train and eval must use the SAME similarity metric:**
- If training with cosine → eval with cosine
- If training with dot product → eval with dot product
- If training with Euclidean → eval with Euclidean

---

## 8. NEXT STEPS

**Immediate (Automated):**
1. ✅ Training in progress (with cosine similarity fix)
2. ⏳ Wait for training completion (~15 minutes)
3. ⏳ Evaluate on scifact + nfcorpus (2 datasets)
4. ⏳ Compare with baseline & ISO15_PRED12

**If This Fix Works:**
1. Train on full MS MARCO dataset (88K examples)
2. Evaluate on all 5 BEIR datasets
3. Write up results and compare all architectures

**If This Fix Fails:**
1. Check if isotropy loss is interfering with contrastive learning
2. Try lowering lambda_isotropy (e.g., 0.1 instead of 1.0)
3. Consider alternative architectures (Option B: Unfreeze top layers)

---

## 9. CONCLUSION

The Asymmetric Projection model's catastrophic failure was caused by **unnormalized contrastive learning** allowing the model to "cheat" by increasing embedding magnitudes rather than learning semantic relationships.

**The fix:** Use cosine similarity (normalize embeddings before computing contrastive loss).

**Current status:** Retraining in progress with fixed loss function. Results expected in ~15 minutes.

**Confidence level:** **HIGH** - This is a well-known failure mode in contrastive learning, and the fix is standard practice.

---

**END OF ANALYSIS**


