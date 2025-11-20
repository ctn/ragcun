# Adapting JEPA Concepts to Non-JEPA Models

## Core JEPA Concepts

### 1. **Joint Embedding Space**
- Both X and Y (e.g., image patches, or queries and documents) are embedded into the **same latent space**
- The embeddings should be semantically meaningful and comparable

### 2. **Predictor Network**
- A neural network that predicts one embedding from another: `ŷ = predictor(x)`
- In JEPA: Predicts Y from X in the latent space
- Key: Predictor operates in the **latent space**, not raw input space

### 3. **Stop-Gradient (StopGrad)**
- Prevents collapse by stopping gradients on one branch (usually the target Y)
- Forces the predictor to learn meaningful representations

### 4. **Self-Supervised Learning**
- No explicit labels needed
- Learns from the structure of the data itself

---

## Current Setup (Non-JEPA)

**What we have:**
- ✅ Joint embedding space: Queries and documents both embedded into 512-dim space
- ✅ Contrastive learning: Pull positives closer, push negatives apart
- ✅ Isotropy regularization: Enforce uniform distribution
- ❌ **No predictor network**
- ❌ **No predictive loss**

**Loss function:**
```python
L = λ_contrastive * L_contrastive + λ_isotropy * L_isotropy + λ_reg * L_reg
```

---

## How to Add JEPA Concepts

### Option 1: **Add Predictor Network** (Most Direct)

**Concept:** Add a predictor that takes query embedding and predicts the positive document embedding.

**Architecture:**
```python
class JEPAAdaptedModel(nn.Module):
    def __init__(self, base_model, output_dim=512):
        self.encoder = base_model  # Encodes both queries and docs
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, queries, documents):
        # Encode both into joint space
        query_emb = self.encoder(queries)      # (B, D)
        doc_emb = self.encoder(documents)      # (B, D)
        
        # Predict document from query
        predicted_doc = self.predictor(query_emb)  # (B, D)
        
        return query_emb, doc_emb, predicted_doc
```

**Loss Function:**
```python
class JEPALoss(nn.Module):
    def forward(self, query_emb, doc_emb, predicted_doc):
        # 1. Predictive loss (JEPA core)
        # Predict doc from query in latent space
        predictive_loss = F.mse_loss(predicted_doc, doc_emb.detach())  # StopGrad on target
        
        # 2. Contrastive loss (optional, for retrieval)
        pos_dist = torch.norm(query_emb - doc_emb, p=2, dim=1)
        contrastive_loss = torch.mean(pos_dist)  # Or margin-based
        
        # 3. Isotropy loss (LeJEPA)
        all_emb = torch.cat([query_emb, doc_emb], dim=0)
        cov = compute_covariance(all_emb)
        isotropy_loss = torch.norm(cov - I * variance, p='fro')
        
        return predictive_loss + λ_contrastive * contrastive_loss + λ_isotropy * isotropy_loss
```

**Key Design Choices:**
1. **Stop-Gradient on target:** `doc_emb.detach()` prevents collapse
2. **Predictor in latent space:** Operates on embeddings, not raw text
3. **Joint training:** Predictor + contrastive + isotropy

---

### Option 2: **Asymmetric Encoders** (JEPA-Style)

**Concept:** Use different encoders for queries and documents, with a predictor.

**Architecture:**
```python
class AsymmetricJEPA(nn.Module):
    def __init__(self, base_model, output_dim=512):
        # Two encoders (can share weights or not)
        self.query_encoder = base_model
        self.doc_encoder = base_model  # Or separate
        
        # Predictor: query → document
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, queries, documents):
        query_emb = self.query_encoder(queries)
        doc_emb = self.doc_encoder(documents)
        predicted_doc = self.predictor(query_emb)
        return query_emb, doc_emb, predicted_doc
```

**Advantages:**
- More flexible (can have different architectures)
- Closer to original JEPA (asymmetric views)

**Disadvantages:**
- More parameters
- Need to ensure embeddings are in same space

---

### Option 3: **Bidirectional Predictor** (More JEPA-Like)

**Concept:** Predict both directions (query→doc and doc→query).

**Architecture:**
```python
class BidirectionalJEPA(nn.Module):
    def __init__(self, base_model, output_dim=512):
        self.encoder = base_model
        self.predictor_q2d = nn.Sequential(...)  # Query → Doc
        self.predictor_d2q = nn.Sequential(...)   # Doc → Query
    
    def forward(self, queries, documents):
        query_emb = self.encoder(queries)
        doc_emb = self.encoder(documents)
        
        predicted_doc = self.predictor_q2d(query_emb)
        predicted_query = self.predictor_d2q(doc_emb)
        
        return query_emb, doc_emb, predicted_doc, predicted_query
```

**Loss:**
```python
L = L_predictive(q→d) + L_predictive(d→q) + L_isotropy
```

---

## Implementation for Your Codebase

### Step 1: Add Predictor to Model

**File: `ragcun/model.py`**

```python
class IsotropicGaussianEncoder(nn.Module):
    def __init__(self, ..., use_predictor=False):
        # ... existing code ...
        
        if use_predictor:
            # Predictor: query embedding → document embedding
            self.predictor = nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
        else:
            self.predictor = None
    
    def forward(self, texts, return_prediction=False, query_emb=None):
        embeddings = self.encode(texts)
        
        if return_prediction and query_emb is not None:
            predicted = self.predictor(query_emb)
            return embeddings, predicted
        
        return embeddings
```

### Step 2: Add Predictive Loss

**File: `scripts/train.py`**

```python
class JEPASIGRegLoss(nn.Module):
    def __init__(
        self,
        lambda_predictive=1.0,      # NEW: Weight for predictive loss
        lambda_contrastive=0.1,
        lambda_isotropy=1.0,
        lambda_reg=0.0,
        use_stopgrad=True,           # NEW: Stop gradient on target
        ...
    ):
        self.lambda_predictive = lambda_predictive
        self.use_stopgrad = use_stopgrad
        # ... rest of init ...
    
    def forward(self, query_emb, pos_emb, neg_emb=None, predicted_pos=None):
        # 1. Predictive loss (JEPA core)
        if predicted_pos is not None:
            target = pos_emb.detach() if self.use_stopgrad else pos_emb
            predictive_loss = F.mse_loss(predicted_pos, target)
        else:
            predictive_loss = torch.tensor(0.0, device=query_emb.device)
        
        # 2. Contrastive loss (existing)
        # ... existing contrastive code ...
        
        # 3. Isotropy loss (existing)
        # ... existing isotropy code ...
        
        total_loss = (
            self.lambda_predictive * predictive_loss +
            self.lambda_contrastive * contrastive_loss +
            self.lambda_isotropy * isotropy_loss +
            self.lambda_reg * reg_loss
        )
        
        return total_loss, loss_dict
```

### Step 3: Update Training Loop

**File: `scripts/train.py` (train_epoch function)**

```python
def train_epoch(...):
    for batch in dataloader:
        # Encode
        query_emb = model(batch['queries'])
        pos_emb = model(batch['positives'])
        
        # Predict (if predictor exists)
        if hasattr(model, 'predictor') and model.predictor is not None:
            predicted_pos = model.predictor(query_emb)
        else:
            predicted_pos = None
        
        # Compute loss
        loss, loss_dict = criterion(
            query_emb, pos_emb, batch.get('negatives'),
            predicted_pos=predicted_pos
        )
        
        # ... rest of training ...
```

---

## Key Design Decisions

### 1. **Stop-Gradient (StopGrad)**
- **Option A:** Stop gradient on target (`pos_emb.detach()`)
  - Prevents collapse
  - Forces predictor to learn meaningful representations
  - **Recommended for JEPA-style training**

- **Option B:** No stop-gradient
  - Both branches trainable
  - Might collapse without other regularization
  - **Not recommended**

### 2. **Predictor Architecture**
- **Simple:** 2-layer MLP (current recommendation)
- **Deep:** 3-4 layers (more capacity, risk of overfitting)
- **Residual:** Add skip connection (helps with gradient flow)

### 3. **Loss Weighting**
- **λ_predictive = 1.0:** Full JEPA (predictive loss dominant)
- **λ_predictive = 0.5, λ_contrastive = 0.5:** Balanced
- **λ_predictive = 0.1, λ_contrastive = 0.9:** Contrastive dominant (current setup)

### 4. **Training Strategy**
- **Phase 1:** Train predictor only (freeze encoder)
- **Phase 2:** Joint training (encoder + predictor)
- **Phase 3:** Fine-tune with isotropy

---

## Expected Benefits

### 1. **Better Representations**
- Predictor forces encoder to learn more structured representations
- Joint space becomes more semantically meaningful

### 2. **Self-Supervised Learning**
- Can learn from unlabeled data (just query-doc pairs)
- No need for hard negatives

### 3. **Theoretical Grounding**
- Aligns with JEPA's proven approach
- Better understanding of what the model learns

### 4. **Potential Performance Gains**
- Predictor might help with zero-shot generalization
- Better alignment between queries and documents

---

## Potential Challenges

### 1. **Collapse**
- **Solution:** Stop-gradient on target, isotropy regularization

### 2. **Computational Overhead**
- Predictor adds forward pass
- **Solution:** Small predictor (2 layers), efficient implementation

### 3. **Hyperparameter Tuning**
- Need to balance predictive, contrastive, and isotropy losses
- **Solution:** Start with λ_predictive=1.0, λ_contrastive=0.1, λ_isotropy=1.0

### 4. **Evaluation**
- Need to verify predictor is learning meaningful relationships
- **Solution:** Check prediction error, visualize embeddings

---

## Recommended Implementation Plan

### Phase 1: Minimal JEPA (Test)
```python
# Add predictor, use only predictive loss
λ_predictive = 1.0
λ_contrastive = 0.0
λ_isotropy = 0.0
```

### Phase 2: Hybrid JEPA (Recommended)
```python
# Combine predictive + contrastive + isotropy
λ_predictive = 0.5
λ_contrastive = 0.1
λ_isotropy = 1.0
```

### Phase 3: Full JEPA + Isotropy
```python
# Full JEPA with isotropy regularization
λ_predictive = 1.0
λ_contrastive = 0.0  # Or small (0.1)
λ_isotropy = 1.0
```

---

## Code Example: Quick Start

```python
# 1. Modify model
model = IsotropicGaussianEncoder(
    base_model='sentence-transformers/all-mpnet-base-v2',
    output_dim=512,
    freeze_base=True,
    use_predictor=True  # NEW
)

# 2. Modify loss
criterion = JEPASIGRegLoss(
    lambda_predictive=0.5,
    lambda_contrastive=0.1,
    lambda_isotropy=1.0,
    use_stopgrad=True
)

# 3. Training loop
for batch in dataloader:
    query_emb = model(batch['queries'])
    pos_emb = model(batch['positives'])
    predicted_pos = model.predictor(query_emb)
    
    loss, _ = criterion(query_emb, pos_emb, predicted_pos=predicted_pos)
    loss.backward()
    optimizer.step()
```

---

## Summary

**To apply JEPA to your non-JEPA model:**

1. ✅ **Joint embedding space** - Already have it!
2. ➕ **Add predictor network** - Predicts doc from query in latent space
3. ➕ **Add predictive loss** - MSE between predicted and actual doc embedding
4. ➕ **Use stop-gradient** - Prevents collapse
5. ✅ **Keep isotropy regularization** - Already have it!

**Key insight:** The predictor forces the encoder to learn representations that are **predictable** in the latent space, which should lead to better semantic alignment between queries and documents.

