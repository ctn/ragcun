# JEPA Training Strategy: What to Train?

## 🎯 Recommendation: **Train Only Predictor** (Freeze Encoder)

Based on your experimental findings, here's the recommended approach:

---

## Your Key Finding

> **"Full fine-tuning destroys zero-shot generalization! Frozen base + isotropy preserves it while adding task-specific learning."**

**Evidence:**
- Full fine-tuning: NDCG@10 = 0.095 (catastrophic forgetting)
- Frozen base + isotropy: NDCG@10 = 0.455 (preserves performance)
- Vanilla baseline: NDCG@10 ≈ 0.49

---

## Training Strategy Options

### ✅ **Option 1: Train Only Predictor** (RECOMMENDED)

**Architecture:**
```
Base Encoder (FROZEN) → Projection Layer (TRAINABLE) → Predictor (TRAINABLE)
```

**What gets trained:**
- ✅ Projection layer (already trainable)
- ✅ Predictor network (new)
- ❌ Base encoder (frozen)

**Implementation:**
```python
model = GaussianEmbeddingGemma(
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=True,  # Keep frozen!
    use_predictor=True
)

# Only predictor + projection are trainable
# Base encoder stays frozen (preserves pre-trained knowledge)
```

**Advantages:**
- ✅ **Preserves zero-shot generalization** (your key finding)
- ✅ **Fast training** (~15-20 minutes for 2 epochs)
- ✅ **Low risk** (no catastrophic forgetting)
- ✅ **Consistent with current approach** (frozen base works!)
- ✅ **Small memory footprint** (only ~2M trainable params)

**Disadvantages:**
- ⚠️ Encoder can't adapt to prediction task
- ⚠️ Limited capacity for learning new patterns

**Expected Performance:**
- Matches or slightly beats frozen base + isotropy (0.455)
- Potentially 0.46-0.48 with predictor helping alignment

---

### ⚠️ **Option 2: Train Both (Risky)**

**Architecture:**
```
Base Encoder (TRAINABLE) → Projection Layer (TRAINABLE) → Predictor (TRAINABLE)
```

**What gets trained:**
- ✅ Base encoder (low LR: 2e-5)
- ✅ Projection layer (high LR: 1e-3)
- ✅ Predictor (high LR: 1e-3)

**Implementation:**
```python
model = GaussianEmbeddingGemma(
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=False,  # Train encoder
    use_predictor=True
)

# Use different learning rates
optimizer = torch.optim.AdamW([
    {'params': model.base.parameters(), 'lr': 2e-5},      # Low LR
    {'params': model.projection.parameters(), 'lr': 1e-3}, # High LR
    {'params': model.predictor.parameters(), 'lr': 1e-3}   # High LR
])
```

**Advantages:**
- ✅ Encoder can adapt to prediction task
- ✅ More capacity for learning
- ✅ Potentially better performance (if it doesn't forget)

**Disadvantages:**
- ❌ **High risk of catastrophic forgetting** (your previous results)
- ❌ Slow training (~5-6 days)
- ❌ Large memory footprint (111M trainable params)
- ❌ Previous experiments showed this fails (0.095 vs 0.49)

**Expected Performance:**
- **Risk:** Could drop to 0.09-0.10 (catastrophic forgetting)
- **Best case:** 0.48-0.50 (if it works, but unlikely based on history)

---

### 🔄 **Option 3: Hybrid (Two-Phase Training)**

**Phase 1:** Train predictor only (frozen encoder)
- Fast, safe, preserves knowledge
- Get predictor working

**Phase 2:** Unfreeze encoder with very low LR
- Only if Phase 1 works well
- Use very low LR (1e-6) to minimize forgetting

**Advantages:**
- ✅ Safe initial phase
- ✅ Option to adapt encoder if needed

**Disadvantages:**
- ⚠️ More complex
- ⚠️ Phase 2 still risky

---

## 🎯 **Recommended Implementation**

### Step 1: Start with Frozen Encoder + Predictor

```python
# Model setup
model = GaussianEmbeddingGemma(
    base_model='sentence-transformers/all-mpnet-base-v2',
    output_dim=512,
    freeze_base=True,  # CRITICAL: Keep frozen!
    use_predictor=True  # NEW: Add predictor
)

# Optimizer (only predictor + projection trainable)
optimizer = torch.optim.AdamW([
    {'params': model.projection.parameters(), 'lr': 5e-4},
    {'params': model.predictor.parameters(), 'lr': 5e-4}
], weight_decay=0.01)
```

### Step 2: Loss Configuration

```python
criterion = JEPASIGRegLoss(
    lambda_predictive=0.5,    # JEPA predictive loss
    lambda_contrastive=0.1,   # Small contrastive (preserves semantics)
    lambda_isotropy=1.0,      # Isotropy regularization
    lambda_reg=0.0,           # No variance constraint
    use_stopgrad=True          # Prevent collapse
)
```

### Step 3: Training

```python
for batch in dataloader:
    # Encode (base encoder frozen, no gradients)
    query_emb = model(batch['queries'])
    doc_emb = model(batch['positives'])
    
    # Predict (predictor trainable)
    predicted_doc = model.predictor(query_emb)
    
    # Loss
    loss, _ = criterion(
        query_emb, doc_emb, 
        predicted_pos=predicted_doc
    )
    
    loss.backward()  # Only updates predictor + projection
    optimizer.step()
```

---

## 📊 Expected Results Comparison

| Strategy | Trainable Params | Training Time | Risk | Expected NDCG@10 |
|----------|-----------------|---------------|------|------------------|
| **Frozen + Predictor** | ~2M | 15-20 min | Low | **0.46-0.48** ✅ |
| Full Fine-Tune + Predictor | 111M | 5-6 days | **High** | 0.09-0.10 ❌ |
| Frozen + Isotropy (current) | ~2M | 15-20 min | Low | 0.455 |

---

## 🚨 **Why NOT Train Encoder**

Your experiments clearly showed:

1. **Full fine-tuning = catastrophic forgetting**
   - Baseline: 0.49 → Trained: 0.095 (-80%!)
   - Zero-shot generalization destroyed

2. **Frozen base = preserves knowledge**
   - Frozen + isotropy: 0.455 (only -7% vs baseline)
   - Still generalizes to new domains

3. **Predictor doesn't need encoder adaptation**
   - Predictor learns to map query → doc in latent space
   - Encoder already provides good representations
   - Predictor just needs to learn the mapping

---

## 💡 **Key Insight**

**The predictor's job is to learn the relationship between queries and documents in the latent space, not to change how the encoder represents text.**

By keeping the encoder frozen:
- ✅ Preserves pre-trained semantic knowledge
- ✅ Predictor learns query→doc mapping
- ✅ Isotropy regularizes the space
- ✅ Best of all worlds!

---

## 🎯 **Final Recommendation**

**Start with: Frozen Encoder + Train Predictor + Isotropy**

This gives you:
1. **Safety:** No risk of catastrophic forgetting
2. **Speed:** Fast training (~20 minutes)
3. **Performance:** Should match or beat current frozen+isotropy (0.455)
4. **JEPA benefits:** Predictor adds predictive learning
5. **Consistency:** Aligns with your successful approach

**Only consider unfreezing encoder if:**
- Frozen + predictor works well (0.46+)
- You have time for risky experiments
- You want to test if very low LR (1e-6) can avoid forgetting

---

## Quick Start Code

```python
# Create model with predictor
model = GaussianEmbeddingGemma(
    base_model='sentence-transformers/all-mpnet-base-v2',
    output_dim=512,
    freeze_base=True,  # Keep frozen!
    use_predictor=True  # Add predictor
)

# Only train predictor + projection
trainable_params = list(model.projection.parameters()) + \
                   list(model.predictor.parameters())

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=5e-4,
    weight_decay=0.01
)

# Loss with predictive component
criterion = JEPASIGRegLoss(
    lambda_predictive=0.5,
    lambda_contrastive=0.1,
    lambda_isotropy=1.0,
    use_stopgrad=True
)
```

**This is the safest, fastest, and most aligned with your successful experiments!**

