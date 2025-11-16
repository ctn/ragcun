# Hyperparameter Guide for Isotropy Regularization

## Current Configuration (Pure Isotropy)
- `λ_contrastive = 0.0` (no contrastive loss)
- `λ_isotropy = 1.0` (isotropy regularization)
- `λ_reg = 0.0` (no variance constraint)
- `margin = 1.0` (not used, but available)
- `output_dim = 512`
- `base_learning_rate = 2e-5` (not used when frozen)
- `projection_learning_rate = 1e-3`
- `batch_size = 16`
- `epochs = 1`
- `freeze_base = True`
- `normalize_embeddings = True`

---

## 1. Loss Function Hyperparameters

### Loss Weights (Most Critical)
- **`--lambda_contrastive`** (default: 1.0, current: 0.0)
  - Weight for contrastive loss (pulls positive pairs closer, pushes negatives apart)
  - **Try:** 0.0, 0.1, 0.5, 1.0, 2.0
  - **Hypothesis:** Small contrastive component (0.1-0.5) might preserve semantics while still regularizing

- **`--lambda_isotropy`** (default: 1.0, current: 1.0)
  - Weight for isotropy loss (enforces isotropic covariance)
  - **Try:** 0.1, 0.5, 1.0, 2.0, 5.0
  - **Hypothesis:** Higher values (2.0-5.0) might improve isotropy but could hurt semantics

- **`--lambda_reg`** (default: 0.1, current: 0.0)
  - Weight for variance regularization (enforces std=1.0)
  - **Try:** 0.0, 0.01, 0.1, 1.0
  - **Note:** Currently disabled (scale-invariant isotropy doesn't need it)

### Contrastive Loss Hyperparameters
- **`--margin`** (default: 1.0)
  - Margin for contrastive loss (distance between positive and negative pairs)
  - **Try:** 0.05, 0.1, 0.5, 1.0, 2.0
  - **Hypothesis:** Smaller margins (0.1-0.5) might work better with isotropy

---

## 2. Training Hyperparameters

### Learning Rates
- **`--base_learning_rate`** (default: 2e-5, current: not used when frozen)
  - Learning rate for base encoder
  - **Try:** 1e-6, 5e-6, 1e-5, 2e-5, 5e-5
  - **Note:** Only relevant if `freeze_base=False`

- **`--projection_learning_rate`** (default: 1e-3, current: 1e-3)
  - Learning rate for projection layer
  - **Try:** 1e-4, 5e-4, 1e-3, 2e-3, 5e-3
  - **Hypothesis:** Lower LR (5e-4) might preserve base embeddings better

- **`--learning_rate`** (default: 2e-5)
  - Global learning rate (used if base/projection LR not specified)

### Optimization
- **`--batch_size`** (default: 8, current: 16)
  - Batch size for training
  - **Try:** 8, 16, 32, 64
  - **Trade-off:** Larger batches = more stable gradients, but slower per epoch

- **`--epochs`** (default: 3, current: 1)
  - Number of training epochs
  - **Try:** 1, 2, 3, 5, 10
  - **Hypothesis:** More epochs might help, but risk overfitting

- **`--weight_decay`** (default: 0.01)
  - L2 regularization weight
  - **Try:** 0.0, 0.001, 0.01, 0.1
  - **Note:** Helps prevent overfitting

- **`--warmup_steps`** (default: 100)
  - Number of warmup steps for learning rate schedule
  - **Try:** 0, 50, 100, 200, 500

---

## 3. Model Architecture Hyperparameters

### Embedding Dimensions
- **`--output_dim`** (default: 512, current: 512)
  - Dimension of output embeddings
  - **Try:** 256, 384, 512, 768, 1024
  - **Trade-off:** Larger = more capacity, but slower and more memory

### Projection Layer Architecture
Currently hardcoded in `ragcun/model.py`:
```python
nn.Sequential(
    nn.Linear(base_dim, base_dim * 2),  # Expansion factor: 2x
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(base_dim * 2, output_dim)
)
```

**Potential modifications:**
- **Expansion factor:** 1.5x, 2x, 3x, 4x
- **Depth:** 2 layers (current), 3 layers, 4 layers
- **Activation:** GELU (current), ReLU, Swish, Tanh
- **Dropout:** 0.0, 0.1, 0.2, 0.3

---

## 4. Training Strategy Hyperparameters

### Freezing Strategy
- **`--freeze_base`** (current: True)
  - Freeze entire base encoder
  - **Try:** True (current), False (full fine-tuning)
  - **Note:** Full fine-tuning caused catastrophic forgetting in previous experiments

- **`--freeze_early_layers`** (default: True)
  - Freeze first 4 transformer layers (only if `freeze_base=False`)
  - **Try:** True, False
  - **Hypothesis:** Partial freezing might help preserve semantics

### Normalization
- **`--no_normalize_embeddings`** (current: False, so normalize=True)
  - Whether to normalize base embeddings before projection
  - **Try:** True (current), False
  - **Hypothesis:** Removing normalization might help isotropy (embeddings already normalized)

---

## 5. Data Hyperparameters

### Dataset Size
- **Training data:** Currently 10K examples
  - **Try:** 10K, 48K (full MS MARCO), 100K, 500K
  - **Hypothesis:** More data might help pure isotropy preserve semantics

### Data Augmentation
- **Negative sampling:** Currently using hard negatives
  - **Try:** Random negatives, hard negatives, mix of both
  - **Note:** Not currently configurable, but could be added

---

## 6. Advanced Hyperparameters

### Mixed Precision Training
- **`--mixed_precision`** (default: False)
  - Use mixed precision (FP16) for faster training
  - **Try:** True, False
  - **Note:** Can speed up training 2x with minimal accuracy loss

### Gradient Clipping
Currently hardcoded: `max_norm=1.0`
- **Try:** 0.5, 1.0, 2.0, 5.0
- **Note:** Prevents gradient explosion

---

## Recommended Hyperparameter Search Strategy

### Phase 1: Loss Weight Tuning (Highest Priority)
1. **Small contrastive component:**
   ```bash
   --lambda_contrastive 0.1 --lambda_isotropy 1.0 --lambda_reg 0.0
   --lambda_contrastive 0.5 --lambda_isotropy 1.0 --lambda_reg 0.0
   ```

2. **Higher isotropy weight:**
   ```bash
   --lambda_contrastive 0.0 --lambda_isotropy 2.0 --lambda_reg 0.0
   --lambda_contrastive 0.0 --lambda_isotropy 5.0 --lambda_reg 0.0
   ```

3. **Small margin:**
   ```bash
   --lambda_contrastive 0.1 --lambda_isotropy 1.0 --margin 0.1
   ```

### Phase 2: Learning Rate Tuning
1. **Lower projection LR:**
   ```bash
   --projection_learning_rate 5e-4
   --projection_learning_rate 1e-4
   ```

2. **More epochs with lower LR:**
   ```bash
   --projection_learning_rate 5e-4 --epochs 3
   ```

### Phase 3: Architecture Tuning
1. **Larger output dimension:**
   ```bash
   --output_dim 768
   --output_dim 1024
   ```

2. **Remove normalization:**
   ```bash
   --no_normalize_embeddings
   ```

### Phase 4: Data Scaling
1. **More training data:**
   ```bash
   --train_data data/processed/msmarco/train.json  # 48K examples
   ```

---

## Quick Test Scripts

### Test 1: Small Contrastive Component
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
    --output_dir "checkpoints/isotropy_contrastive_01"
```

### Test 2: Higher Isotropy Weight
```bash
python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --freeze_base \
    --lambda_contrastive 0.0 \
    --lambda_isotropy 2.0 \
    --lambda_reg 0.0 \
    --projection_learning_rate 5e-4 \
    --epochs 2 \
    --output_dir "checkpoints/isotropy_weight_2"
```

### Test 3: Remove Normalization
```bash
python scripts/train.py \
    --train_data data/processed/msmarco_smoke/train.json \
    --val_data data/processed/msmarco_smoke/dev.json \
    --base_model "sentence-transformers/all-mpnet-base-v2" \
    --freeze_base \
    --no_normalize_embeddings \
    --lambda_contrastive 0.0 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.0 \
    --output_dir "checkpoints/isotropy_no_norm"
```

---

## Expected Impact

| Hyperparameter | Expected Impact | Priority |
|----------------|----------------|----------|
| `lambda_contrastive` (0.1-0.5) | **High** - Preserves semantics | ⭐⭐⭐⭐⭐ |
| `lambda_isotropy` (2.0-5.0) | **Medium** - Better isotropy | ⭐⭐⭐ |
| `margin` (0.1-0.5) | **Medium** - Better with isotropy | ⭐⭐⭐ |
| `projection_learning_rate` (5e-4) | **Medium** - More stable | ⭐⭐⭐ |
| `epochs` (2-3) | **Low** - More training | ⭐⭐ |
| `no_normalize_embeddings` | **Low** - Architectural change | ⭐⭐ |
| `output_dim` (768) | **Low** - More capacity | ⭐ |

---

## Key Insights from Current Results

1. **Pure isotropy (-4.1% vs baseline)** suggests we need contrastive loss to preserve semantics
2. **Frozen base** is essential (full fine-tuning caused catastrophic forgetting)
3. **Small std (0.014)** indicates embeddings are very compact - might need variance regularization
4. **Loss near zero** suggests isotropy loss might be too weak or embeddings already isotropic

**Next steps:** Try `λ_contrastive=0.1-0.5` with `λ_isotropy=1.0` to balance semantics and isotropy.

