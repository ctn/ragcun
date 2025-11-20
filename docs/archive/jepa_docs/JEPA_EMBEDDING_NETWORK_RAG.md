# Embedding Network for RAG: What Gets Used?

## 🎯 Key Distinction

**For RAG/Retrieval (Inference):**
- ✅ **Use:** Encoder + Projection Layer (the embedding network)
- ❌ **Don't use:** Predictor (training-only component)

**For Training:**
- ✅ **Use:** Encoder + Projection + Predictor (all three)
- Predictor helps train better embeddings, but is discarded at inference

---

## Architecture Breakdown

### During Training (with JEPA predictor)

```
Query Text
    ↓
Base Encoder (FROZEN) ──┐
    ↓                    │
Projection (TRAINABLE)   │
    ↓                    │
Query Embedding ─────────┼──→ Predictor (TRAINABLE) ──→ Predicted Doc Embedding
                         │                                    ↓
Document Text            │                              Loss: MSE(predicted, actual)
    ↓                    │                                    ↓
Base Encoder (FROZEN) ───┘                              Backprop to:
    ↓                                                      - Projection
Projection (TRAINABLE)                                    - Predictor
    ↓                                                      (NOT base encoder)
Document Embedding ──────────────────────────────────────→
```

### During RAG/Retrieval (Inference)

```
Query Text                    Document Text
    ↓                              ↓
Base Encoder (FROZEN)         Base Encoder (FROZEN)
    ↓                              ↓
Projection (TRAINED)          Projection (TRAINED)
    ↓                              ↓
Query Embedding ────────────── Document Embedding
    │                              │
    └──────────→ Euclidean Distance ←──────────┘
                      ↓
              Retrieve Top-K
```

**Notice:** Predictor is NOT used during retrieval!

---

## What is the "Embedding Network"?

The **embedding network** for RAG is:

```
Embedding Network = Base Encoder + Projection Layer
```

**Components:**
1. **Base Encoder** (frozen): `sentence-transformers/all-mpnet-base-v2`
   - Converts text → 768-dim normalized embeddings
   - Provides semantic understanding
   - **Frozen** (preserves pre-trained knowledge)

2. **Projection Layer** (trainable): `Linear(768→1536) → GELU → Dropout → Linear(1536→512)`
   - Converts normalized 768-dim → unnormalized 512-dim Gaussian embeddings
   - **Trained** (adapts to task + isotropy)

**Total:** ~2M trainable parameters (just projection)

---

## How It Works in RAG

### Step 1: Index Documents (One-time)

```python
# Load model (encoder + projection, NO predictor needed)
model = IsotropicGaussianEncoder.from_pretrained('checkpoints/model.pt')
model.eval()  # Inference mode

# Encode all documents
doc_embeddings = model.encode(documents)  # Uses encoder + projection

# Store in vector database
vector_db.add(doc_embeddings, documents)
```

### Step 2: Query (Real-time)

```python
# Encode query
query_embedding = model.encode([query])  # Uses encoder + projection

# Search vector DB using Euclidean distance
results = vector_db.search(query_embedding, top_k=5)
```

### Step 3: Retrieve

```python
# Vector DB computes Euclidean distances
distances = ||query_emb - doc_emb||_2

# Return closest documents
return top_k_by_distance(documents, distances)
```

---

## Why Predictor is NOT Used in RAG

**The predictor is a training tool, not an inference component.**

### During Training:
- Predictor learns: "Given query embedding, predict document embedding"
- This forces the encoder+projection to learn embeddings that are **predictable**
- Better embeddings = better retrieval

### During Inference:
- You don't need to predict - you have the actual document embeddings!
- Just compute distance: `||query_emb - doc_emb||`
- Predictor would add unnecessary computation

**Analogy:**
- **Training:** Predictor is like a "teacher" that helps the student (encoder) learn
- **Inference:** You don't need the teacher anymore - the student (encoder) can work alone

---

## Code Example: RAG Usage

```python
from ragcun import IsotropicRetriever

# Load model (encoder + projection, predictor not loaded)
retriever = IsotropicRetriever(model_path='checkpoints/model.pt')

# Add documents (uses encoder + projection)
retriever.add_documents([
    "Python is a programming language",
    "Machine learning uses algorithms",
    # ... more documents
])

# Query (uses encoder + projection)
results = retriever.retrieve("What is Python?", top_k=3)

# Results: [(document, euclidean_distance), ...]
```

**What happens internally:**

```python
# In retriever.retrieve():
query_emb = model.encode([query])  # encoder + projection only
# NOT: predicted = model.predictor(query_emb)  # ❌ Don't do this!

# Search using actual document embeddings
distances = euclidean_distance(query_emb, doc_embeddings)
```

---

## What Gets Saved in Checkpoint?

When you save a model:

```python
torch.save({
    'base_encoder': model.base.state_dict(),      # ✅ Saved (frozen)
    'projection': model.projection.state_dict(),  # ✅ Saved (trained)
    'predictor': model.predictor.state_dict()    # ✅ Saved (trained)
}, 'checkpoint.pt')
```

**For RAG inference, you need:**
- ✅ Base encoder (to encode text)
- ✅ Projection (to get Gaussian embeddings)
- ❌ Predictor (optional, not needed for retrieval)

**You can load without predictor:**

```python
# Load for RAG (predictor not needed)
model = IsotropicGaussianEncoder.from_pretrained(
    'checkpoint.pt',
    load_predictor=False  # Skip predictor
)
```

---

## Summary

| Component | Training | RAG/Retrieval | Purpose |
|-----------|----------|---------------|---------|
| **Base Encoder** | Frozen | Used | Provides semantic understanding |
| **Projection** | Trainable | Used | Converts to Gaussian embeddings |
| **Predictor** | Trainable | **NOT used** | Helps train better embeddings |

**The embedding network for RAG = Base Encoder + Projection**

**The predictor = Training-only helper (like a teacher during learning)**

---

## Key Insight

**The predictor doesn't change what embeddings you use for RAG - it changes how good those embeddings are.**

- **Without predictor:** Embeddings learned via contrastive loss only
- **With predictor:** Embeddings learned via contrastive + predictive loss (better!)

But in both cases, you use the same embedding network (encoder + projection) for retrieval.

---

## Implementation Note

If you add a predictor, make sure your `encode()` method doesn't use it:

```python
def encode(self, texts):
    # Base encoder
    base_emb = self.base.encode(texts)
    
    # Projection
    gaussian_emb = self.projection(base_emb)
    
    # DON'T do this:
    # predicted = self.predictor(gaussian_emb)  # ❌ Wrong!
    
    return gaussian_emb  # ✅ Correct
```

The predictor is only used during training in the loss function, not during inference.

