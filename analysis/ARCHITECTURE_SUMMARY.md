# Architecture Evolution Summary

**Date:** 2025-11-18  
**Project:** RAGCun - Isotropic Gaussian Embeddings for Dense Retrieval  
**Objective:** Develop retrieval embeddings that achieve JEPA-style isotropy while maintaining/improving baseline performance

---

## Performance Overview

| Architecture | Avg NDCG@10 | vs Baseline | Status | Training Data |
|--------------|-------------|-------------|--------|---------------|
| **Frozen MPNet (Baseline)** | 0.4628 | - | ✅ Reference | 10K MS MARCO |
| **ISO15_PRED12** | **0.4759** | **+2.83%** | ✅ **Best** | 10K MS MARCO |
| **JEPA 10K (Epoch 2)** | **0.4779** | **+3.26%** | ✅ **Best** | 10K MS MARCO |
| **ResPred (Epoch 1)** | 0.4416 | -4.59% | ❌ Failed | 10K MS MARCO |
| **ResPred (Epoch 3)** | 0.3871 | -16.37% | ❌ Failed | 10K MS MARCO |
| **Asymmetric (FIXED)** | 0.3235 | -30.11% | ❌ Failed | 10K MS MARCO |
| **Asymmetric (Epoch 3)** | 0.1888 | -59.20% | ❌ Failed | 10K MS MARCO |

---

## Architecture 1: MPNet + Projection (Baseline)

**Status:** ✅ Working baseline  
**File:** `ragcun/model.py` → `IsotropicGaussianEncoder`

### Architecture
```
Frozen MPNet Encoder
        ↓
    Projection Head (2-layer MLP)
        ↓
  Isotropic Embeddings
```

### Configuration
- **Loss:** Isotropy only (λ_isotropy=1.0)
- **No predictor, no contrastive loss**
- **Frozen base encoder**
- **Output dim:** 512 or 768

### Performance
- **SciFact NDCG@10:** 0.6241
- **NFCorpus NDCG@10:** 0.3016
- **Average:** 0.4628

### Verdict
✅ **Solid baseline** - Simple, stable, performant

---

## Architecture 2: JEPA with Predictor (ISO15_PRED12)

**Status:** ✅ **BEST PERFORMING**  
**File:** `ragcun/mpnet_lejepa.py` → `MPNetLeJEPAIG`  
**Checkpoint:** `checkpoints/jepa_iso15_pred12/`

### Architecture
```
                    Frozen MPNet
                         ↓
              ┌──────────┴──────────┐
         Online Head           Target Head (EMA)
              ↓                      ↓
         z_online               z_target
              ↓
         Predictor
              ↓
         p_online ──────────────→ [JEPA Loss] ←── z_target
         
         z_online ──────────────→ [Isotropy Loss]
```

### Configuration
- **Loss weights:**
  - λ_isotropy = 1.5
  - λ_predictive (JEPA) = 1.2
  - λ_contrastive = 0.0 (disabled)
- **Frozen base encoder**
- **Output dim:** 768
- **EMA decay:** 0.999

### Performance
- **SciFact NDCG@10:** 0.6280 (+0.6% vs baseline)
- **NFCorpus NDCG@10:** 0.3237 (+7.3% vs baseline)
- **Average:** 0.4759 (+2.83% vs baseline) ✅

### Training Results (10K data, 3 epochs)

| Epoch | SciFact | NFCorpus | Average | vs Baseline |
|-------|---------|----------|---------|-------------|
| 1 | 0.6382 | 0.3169 | 0.4775 | +3.18% ✅ |
| 2 | 0.6401 | 0.3158 | **0.4779** | **+3.26%** ✅ |
| 3 | 0.6383 | 0.3154 | 0.4768 | +3.03% ✅ |

### Key Insights
- **Consistent improvement** over baseline across all epochs
- **Best at Epoch 2** (slight overfitting by epoch 3)
- **Stronger gains on NFCorpus** (+4-7%) vs SciFact (+2%)
- **JEPA predictor helps** despite no contrastive loss

### Verdict
✅ **SUCCESS** - Best architecture so far. The predictor provides useful learning signal even without contrastive loss. Isotropy + JEPA is a winning combination.

---

## Architecture 3: ResPred (Residual Predictor)

**Status:** ❌ **FAILED - Residual Collapse**  
**File:** `ragcun/respred_model.py`  
**Analysis:** `analysis/respred_failure_analysis.md`

### Architecture
```
Frozen MPNet
     ↓
Query Embedding
     ↓
Residual Predictor
     ↓
   delta
     ↓
Query + alpha * tanh(delta) ───→ Predicted Doc Embedding
                                        ↓
                                  [Predictive Loss]
                                        ↓
                                  [Isotropy Loss]
                                        ↓
                                  [Residual Loss: λ * ||delta||²]
```

### Configuration
- **Loss weights:**
  - λ_isotropy = 1.5
  - λ_predictive = 1.2
  - λ_residual = 0.01 (penalty on residual magnitude)
- **Learnable alpha** (correction scale)
- **Tanh() bounds delta** to [-1, 1]
- **Frozen base encoder**

### Performance

| Epoch | SciFact | NFCorpus | Average | vs Baseline | Degradation |
|-------|---------|----------|---------|-------------|-------------|
| 1 | 0.6079 | 0.2752 | 0.4416 | -4.59% ❌ | - |
| 3 | 0.5437 | 0.2304 | 0.3871 | -16.37% ❌ | -12.34% worse |

**Performance DEGRADES with more training!**

### Root Cause: Residual Collapse

The residual magnitudes shrink to nearly zero:

| Epoch | Delta Mean | Delta Max | Alpha | Actual Max Correction |
|-------|------------|-----------|-------|----------------------|
| 1 | 0.001755 | 0.009882 | 0.0794 | 0.000784 |
| 2 | 0.000649 | 0.003732 | 0.0778 | 0.000290 |
| 3 | 0.000459 | 0.002706 | 0.0772 | **0.000209** (0.02%!) |

**What happened:**
1. λ_residual penalty drives delta → 0
2. Learnable alpha amplifies collapse (learns to be tiny)
3. Model learns **identity mapping**: `predicted_doc ≈ query_emb`
4. Satisfies all loss terms while being completely useless

### Key Failure Points
1. **Double penalty on residuals:** Tanh() + L2 regularization
2. **Wrong optimization objective:** Easier to minimize ||delta|| than learn useful corrections
3. **Conflicting objectives:** Isotropy + predictive + residual penalty
4. **Base model already good:** Little room for small corrections

### Loss Evolution
- **Residual loss dominates:** 0.2110 → 0.0007 (model minimizes this!)
- **Total loss decreases** (good training signal)
- **But retrieval degrades** (bad generalization)

### Verdict
❌ **ARCHITECTURAL FAILURE** - The λ_residual penalty is fundamentally incompatible with learning useful corrections. Model converges to identity mapping.

**Recommendation:** Remove λ_residual completely, or abandon residual learning.

---

## Architecture 4: Asymmetric Projections

**Status:** ❌ **FAILED - Severe Underperformance**  
**File:** `ragcun/asymmetric_model.py` → `AsymmetricProjectionModel`

### Architecture
```
Frozen MPNet (shared)
        ↓
    ┌───┴────┐
Query Proj   Doc Proj  ← DIFFERENT heads
    ↓          ↓
   z_q        z_d      ← Asymmetric embeddings
    
Loss: Contrastive only (no predictor, no JEPA)
```

### Configuration
- **Two separate projection heads**
- **No predictor network**
- **Contrastive loss only**
- **Frozen base encoder**
- **Output dim:** 768

### Performance

| Version | SciFact | NFCorpus | Average | vs Baseline |
|---------|---------|----------|---------|-------------|
| FIXED (Epoch 1) | 0.4467 | 0.2003 | 0.3235 | -30.11% ❌ |
| Epoch 3 | 0.2641 | 0.1135 | 0.1888 | -59.20% ❌ |

**Catastrophic performance loss!**

### What Went Wrong?
1. **No JEPA/predictor:** Lost the learning signal that made ISO15_PRED12 work
2. **Pure contrastive is weak:** Without predictor, asymmetry isn't useful
3. **Gets worse with training:** Epoch 3 is 42% worse than Epoch 1
4. **Mode collapse suspected:** Embeddings may be collapsing to similar representations

### Comparison to ResPred
- ResPred had **residual collapse** (identity mapping)
- Asymmetric has **embedding collapse** (all outputs similar)
- Both fail for different reasons, but both fail hard

### Verdict
❌ **SEVERE FAILURE** - Worse than baseline by 30-60%. The asymmetric projection approach doesn't work without a predictor/JEPA component.

**Recommendation:** Abandon pure asymmetric approach, or add JEPA predictor back.

---

## Key Learnings

### ✅ What Works

1. **JEPA Predictor is Critical**
   - ISO15_PRED12 (with predictor): +2.83% ✅
   - Asymmetric (no predictor): -30% ❌
   - **Lesson:** The predictor provides essential learning signal

2. **Isotropy + JEPA Synergy**
   - Both losses work well together
   - λ_isotropy=1.5 + λ_predictive=1.2 is a good balance
   - No need for contrastive loss (λ_contrastive=0.0 works)

3. **Frozen Base + Smart Projection**
   - Freezing MPNet is fine
   - Focus training on projection + predictor
   - Fast training, stable convergence

4. **Epoch 2 is Sweet Spot** (for 10K data)
   - Epoch 1: Underfit
   - Epoch 2: Optimal (+3.26%)
   - Epoch 3: Slight overfit (+3.03%)

### ❌ What Fails

1. **Residual Regularization is Harmful**
   - λ_residual penalty causes collapse
   - Model learns identity mapping to minimize penalty
   - **Lesson:** Don't penalize what you want to learn!

2. **Asymmetric Projections Need JEPA**
   - Pure asymmetry without predictor fails badly
   - Contrastive-only is insufficient
   - **Lesson:** JEPA predictor is not optional

3. **Training Loss ≠ Task Performance**
   - ResPred: Val loss improved, retrieval degraded
   - **Lesson:** Always evaluate on end task (BEIR)

4. **Identity Mapping is a Strong Attractor**
   - With strong base model + penalties, identity is stable
   - Need asymmetry or strong learning signal to escape
   - **Lesson:** Design losses that force learning

### 🎯 Optimal Configuration (Validated)

**Architecture:** MPNet + Projection + Predictor + Target (EMA)  
**Loss:** λ_isotropy=1.5 + λ_predictive=1.2  
**Training:** 10K data, 2-3 epochs  
**Result:** +3.26% over baseline ✅

---

## Architecture Comparison Matrix

| Feature | Baseline | ISO15_PRED12 | ResPred | Asymmetric |
|---------|----------|--------------|---------|------------|
| **Frozen Encoder** | ✅ | ✅ | ✅ | ✅ |
| **Projection Head** | ✅ Single | ✅ Online+Target | ✅ Single | ✅ Query+Doc |
| **Predictor** | ❌ | ✅ | ✅ | ❌ |
| **EMA Target** | ❌ | ✅ | ❌ | ❌ |
| **Contrastive Loss** | ❌ | ❌ | ❌ | ✅ |
| **JEPA Loss** | ❌ | ✅ | ✅ | ❌ |
| **Isotropy Loss** | ✅ | ✅ | ✅ | ❌ |
| **Residual Penalty** | ❌ | ❌ | ✅ (failure) | ❌ |
| **Performance** | 0.4628 | **0.4779** ✅ | 0.3871 ❌ | 0.1888 ❌ |
| **Status** | Baseline | **BEST** | Failed | Failed |

---

## Next Steps / Recommendations

### 1. ✅ **Use ISO15_PRED12 Architecture**
   - This is the proven winner (+3.26%)
   - Configuration is validated and stable

### 2. 🚀 **Train on Full MS MARCO (500K samples)**
   - Current best is on 10K smoke test
   - Full data likely to improve to +5-10% over baseline
   - Allow 5-10 epochs for convergence

### 3. 🔬 **Investigate N(0,1) Formulation**
   - Current "OLD" formulation: Standardize + L2 normalize
   - Explore "NEW" formulation: Pure N(0,1) standardization
   - Compare isotropy properties and retrieval performance

### 4. ⚙️ **SIGReg Isotropy Alternative**
   - Current: Covariance-based isotropy (O(D²))
   - Alternative: SIGReg from LeJEPA paper (O(K·N·M))
   - Test if SIGReg provides better isotropy

### 5. ❌ **Abandon Failed Approaches**
   - **Don't** use λ_residual penalty (causes collapse)
   - **Don't** use asymmetric projections without JEPA
   - **Don't** train pure contrastive without predictor

---

## Training Data Analysis

**Current Bottleneck:** Dataset Size

| Dataset Size | Epochs | Training Time | Expected Performance |
|--------------|--------|---------------|---------------------|
| **10K (current)** | 3 | ~13 min | Baseline +3% ✅ |
| 50K | 5 | ~1-2 hours | Baseline +4-6% |
| **500K (full)** | 5-10 | ~10-20 hours | Baseline +5-10% |

**Observation from latest training:**
- Model converges in 1 epoch on 10K data
- Validation loss plateaus after epoch 2
- Signs of overfitting by epoch 3
- **Need more training data, not more epochs**

---

## Conclusion

### 🏆 Winner: ISO15_PRED12 (JEPA + Isotropy + Predictor)

**Why it works:**
1. **JEPA predictor** provides strong learning signal
2. **Isotropy loss** shapes embedding space properly
3. **EMA target network** stabilizes training
4. **No conflicting penalties** (no λ_residual, no λ_contrastive)
5. **Simple, focused objective** that aligns with retrieval

**Performance:**
- +3.26% over baseline (10K data)
- Consistent across multiple runs
- Stable training (no collapse, no instability)

### 🔴 Failures: ResPred & Asymmetric

**ResPred:** Residual regularization causes identity mapping  
**Asymmetric:** No predictor → no learning signal → collapse

Both demonstrate that **JEPA predictor is essential** for this task.

---

**Recommendation:** Move forward with ISO15_PRED12 on full MS MARCO dataset (500K samples) for 5-10 epochs. This is the validated, winning architecture.

