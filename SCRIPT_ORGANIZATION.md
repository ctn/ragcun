# Script Organization by Model Class

## 📊 Model Class Hierarchy

```
1. IsotropicGaussianEncoder (Base)
   ├── ResidualGaussianEncoder (Inherits)
   └── (Most flexible, used by multiple training strategies)

2. AsymmetricDualEncoder (Separate projections)

3. AsymmetricWithPredictor (Separate projections + predictor)

4. MPNetLeJEPA (Dual encoder with EMA)
```

---

## 🗂️ Current Organization by Model Class

### **IsotropicGaussianEncoder Family**

**Model:** `ragcun/model.py::IsotropicGaussianEncoder`
- **Architecture:** Shared encoder + Shared projection + Optional predictor
- **Pattern:** (1, 1, 0) or (1, 1, 1)

**Training Scripts:**
- `scripts/train/isotropic.py` - Main contrastive training
  - Uses: Query-doc pairs with contrastive + isotropy + predictive losses
  - Config: `use_predictor=True` for (1,1,1)
  
- `scripts/train/xy_masked.py` - Self-supervised X/Y masked pairs
  - Uses: Original text (X) vs masked text (Y)
  - Pure predictive + isotropy (no contrastive)
  
- `scripts/train/self_supervised.py` - Document splitting
  - Uses: Split documents into part1 and part2
  - Pure predictive + isotropy (no contrastive)

**Evaluation Scripts:**
- `scripts/eval/beir.py` - Generic BEIR evaluation

**Best Models:**
- ✅ `jepa_10k` (NDCG@10: 0.4779) - **WINNER**
- ✅ `jepa_iso15_pred12` (NDCG@10: 0.4610)
- ✅ `512dim_10k` (NDCG@10: 0.4726)
- ✅ `pure_isotropy_only` (NDCG@10: 0.4562)

---

### **ResidualGaussianEncoder (Inherits IsotropicGaussianEncoder)**

**Model:** `ragcun/respred_model.py::ResidualGaussianEncoder`
- **Architecture:** Shared encoder + Shared projection + Bounded residual predictor
- **Pattern:** (1, 1, 1) with special predictor

**Training Scripts:**
- `scripts/train/residual_gaussian.py` - ResPred training
  - Uses: Residual connection with learnable alpha
  - Bounded predictions with Tanh

**Evaluation Scripts:**
- `scripts/eval/residual_gaussian_quick.py`

**Best Models:**
- ❌ `respred_*` (NDCG@10: 0.4416) - **FAILED** (identity trap)

---

### **AsymmetricDualEncoder**

**Model:** `ragcun/asymmetric_model.py::AsymmetricDualEncoder`
- **Architecture:** Shared encoder + Separate projections + No predictor
- **Pattern:** (1, 0, 0)

**Training Scripts:**
- `scripts/train/asymmetric_dual.py` - Asymmetric projections
  - Uses: Contrastive + dual isotropy losses
  - No predictor

**Evaluation Scripts:**
- `scripts/eval/asymmetric_dual_quick.py`

**Best Models:**
- ✅ `asymmetric_smoke` (NDCG@10: ~0.47) - **GOOD**

---

### **AsymmetricWithPredictor**

**Model:** `ragcun/asymmetric_predictor_model.py::AsymmetricWithPredictor`
- **Architecture:** Shared encoder + Separate projections + Predictor
- **Pattern:** (1, 0, 1)

**Training Scripts:**
- `scripts/train/asymmetric_predictor.py` - Asymmetric + predictor
  - Uses: Contrastive + dual isotropy + predictive losses
  - Best of both worlds attempt

**Evaluation Scripts:**
- `scripts/eval/asymmetric_predictor_quick.py`

**Best Models:**
- ✅ `asymmetric_pred_smoke` (NDCG@10: ~0.48) - **GOOD**

---

### **MPNetLeJEPA**

**Model:** `ragcun/mpnet_lejepa.py::MPNetLeJEPA`
- **Architecture:** Dual encoder (online + target) + Dual projections + Predictor
- **Pattern:** (0, 0, 1) - Full BYOL/JEPA style

**Training Scripts:**
- `scripts/train/mpnet_lejepa.py` - Full JEPA training
  - Uses: Online/target networks with EMA
  - Predictor on online branch only

**Evaluation Scripts:**
- (None specific - can use generic beir.py)

**Best Models:**
- ❓ Not tested yet (theory says good, but unused)

---

## 📈 Performance Ranking by Model Class

| Rank | Model Class | Best Example | NDCG@10 | Pattern |
|------|------------|--------------|---------|---------|
| 🥇 | IsotropicGaussianEncoder | jepa_10k | 0.4779 | (1,1,1) |
| 🥈 | AsymmetricWithPredictor | asymmetric_pred | ~0.48 | (1,0,1) |
| 🥉 | AsymmetricDualEncoder | asymmetric | ~0.47 | (1,0,0) |
| 4 | IsotropicGaussianEncoder | pure_isotropy | 0.4562 | (1,1,0) |
| ❌ | ResidualGaussianEncoder | respred | 0.4416 | Failed |
| ❓ | MPNetLeJEPA | - | Untested | (0,0,1) |

---

## 🎯 Recommended Usage by Model Class

### **For Best Performance:**
- Use: `IsotropicGaussianEncoder` with `use_predictor=True`
- Script: `scripts/train/isotropic.py`
- Pattern: (1, 1, 1)

### **For Explicit Query/Doc Separation:**
- Use: `AsymmetricWithPredictor` 
- Script: `scripts/train/asymmetric_predictor.py`
- Pattern: (1, 0, 1)

### **For Self-Supervised Learning:**
- Use: `IsotropicGaussianEncoder` 
- Script: `scripts/train/xy_masked.py` or `self_supervised.py`
- Pattern: (1, 1, 1)

### **For Research/Ablation:**
- Use: `MPNetLeJEPA` for full BYOL/JEPA comparison
- Script: `scripts/train/mpnet_lejepa.py`
- Pattern: (0, 0, 1)

---

## 🔄 Proposed Reorganization

```
scripts/
├── by_model/
│   ├── isotropic_gaussian/
│   │   ├── train_contrastive.py (isotropic.py)
│   │   ├── train_xy_masked.py
│   │   ├── train_self_supervised.py
│   │   └── eval_beir.py
│   │
│   ├── residual_gaussian/
│   │   ├── train_respred.py (residual_gaussian.py)
│   │   └── eval_respred.py
│   │
│   ├── asymmetric_dual/
│   │   ├── train.py (asymmetric_dual.py)
│   │   └── eval_quick.py
│   │
│   ├── asymmetric_predictor/
│   │   ├── train.py (asymmetric_predictor.py)
│   │   └── eval_quick.py
│   │
│   └── mpnet_lejepa/
│       ├── train.py (mpnet_lejepa.py)
│       └── eval.py
│
└── workflows/  (keeps existing composite scripts)
```
