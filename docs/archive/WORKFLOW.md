# LeJEPA Embedding Project - Complete Workflow

This document outlines the complete pipeline from training to evaluation.

## 📋 Project Structure

```
ragcun/
├── Phase 1: Training (Colab)
│   └── notebooks/lejepa_training.ipynb         # Train isotropic Gaussian embeddings
│
├── Phase 2: Testing (Colab or Local)
│   └── notebooks/evaluate_isotropy.ipynb       # Verify N(0,I) distribution
│
├── Phase 3: RAG Usage (Local)
│   ├── ragcun/model.py                     # IsotropicGaussianEncoder
│   ├── ragcun/retriever.py                 # L2 distance retriever
│   └── examples/retrieval_example.py           # Usage demo
│
└── Phase 4: Evaluation (Local)
    └── notebooks/evaluate_rag.ipynb            # Compare original vs fine-tuned
```

---

## 🚀 Complete Workflow

### **Phase 1: Training** (Google Colab, 1-3 hours)

**Goal:** Fine-tune EmbeddingGemma-300M with LeJEPA loss to produce isotropic Gaussian embeddings.

**Steps:**

1. **Open Colab:**
   - Navigate to: `notebooks/lejepa_training.ipynb`
   - Upload to Google Colab
   - Set Runtime → GPU (T4, V100, or A100)

2. **Run Training:**
   ```python
   # The notebook does:
   - Loads EmbeddingGemma-300M (300M params)
   - Adds projection layer (768 → 512 dims)
   - Loads MS MARCO dataset (query-pos-neg triplets)
   - Trains with:
     * Euclidean contrastive loss (λ=1.0)
     * LeJEPA SIGReg loss (λ=0.01)
   - Verifies isotropy during training
   - Saves best checkpoint
   ```

3. **Download Model:**
   ```python
   # Last cell downloads:
   gaussian_embeddinggemma_final.pt
   ```

4. **Save Locally:**
   ```bash
   # Move to project
   mv ~/Downloads/gaussian_embeddinggemma_final.pt \
      /Users/ctn/src/ctn/ragcun/data/embeddings/
   ```

**Expected Results:**
- Training loss decreases
- Isotropy metrics improve:
  - Mean norm → ~0
  - Cov error → <5
  - Diagonal variance → ~1
  - Off-diagonal correlation → ~0

---

### **Phase 2: Testing** (Colab or Local, 30 minutes)

**Goal:** Verify that fine-tuned embeddings follow isotropic Gaussian N(0,I) distribution.

**Notebook:** `notebooks/evaluate_isotropy.ipynb`

**What it tests:**

1. **Distribution Analysis:**
   ```python
   # Compare Original vs LeJEPA:
   - Mean statistics (want ~0)
   - Variance statistics (want ~1)
   - Norm distribution (should vary, not constant)
   - Covariance structure (want identity matrix)
   ```

2. **Visual Comparison:**
   - Norm histograms
   - Dimension distributions
   - Covariance heatmaps

3. **Output:**
   ```
   data/processed/isotropy_comparison.png
   data/processed/isotropy_comparison.csv
   ```

**Success Criteria:**
- ✅ `is_isotropic = True`
- ✅ Mean norm < 0.5
- ✅ Covariance error < 10
- ✅ Diagonal ~ 1, Off-diagonal ~ 0

---

### **Phase 3: RAG Usage** (Local)

**Goal:** Use trained model for document retrieval.

**Method 1: Programmatic API**

```python
from ragcun import GaussianRetriever

# Load trained model
retriever = GaussianRetriever(
    model_path='data/embeddings/gaussian_embeddinggemma_final.pt'
)

# Add documents
documents = [
    "Python is a programming language",
    "Machine learning is AI",
    "NLP processes human language"
]
retriever.add_documents(documents)

# Retrieve with Euclidean distance
results = retriever.retrieve("What is machine learning?", top_k=3)

for doc, distance in results:
    print(f"[dist={distance:.3f}] {doc}")
```

**Method 2: Run Example Script**

```bash
cd /Users/ctn/src/ctn/ragcun
python examples/retrieval_example.py
```

**Key Differences from Traditional RAG:**

| Aspect | Traditional | LeJEPA |
|--------|------------|--------|
| **Similarity metric** | Cosine similarity | Euclidean distance (L2) |
| **Normalization** | ✅ L2 normalized | ❌ NOT normalized |
| **Magnitude meaning** | Constant (=1) | Confidence/uncertainty |
| **Score range** | [0, 1] (narrow) | [0, ∞] (wide separation) |
| **Composition** | Distorted by normalization | Natural addition works |

---

### **Phase 4: Evaluation** (Local, 1 hour)

**Goal:** Quantify improvement over original model.

**Notebook:** `notebooks/evaluate_rag.ipynb`

**Evaluation Metrics:**

1. **Recall@10:**
   - % of queries where correct doc is in top 10
   - Higher is better

2. **MRR (Mean Reciprocal Rank):**
   - Average of 1/rank of correct doc
   - Range: [0, 1], higher is better

3. **Separation:**
   - Difference between positive and negative scores
   - **KEY METRIC**: Larger separation = better discrimination

4. **Calibration:**
   - Are scores meaningful probabilities?
   - Can we set meaningful thresholds?

**Test Dataset:**
- MS MARCO dev set (500 query-doc pairs)
- Each query has 1 positive doc, 1+ negative docs

**Expected Improvements:**

```
Metric              Original    LeJEPA      Improvement
-----------------------------------------------------
Recall@10           0.78        0.82        +5.1%
MRR                 0.64        0.68        +6.3%
Separation          0.07        0.58        +8.3x  ⭐
```

**Output:**
```
data/processed/rag_performance_comparison.csv
data/processed/rag_performance.png
```

---

## 🎯 Why This Works

### **Problem with Spherical Embeddings:**

```python
# All embeddings have norm = 1.0
query_emb = [0.7, 0.5, 0.2, ...]  # normalized, ||emb|| = 1.0
doc1_emb = [0.6, 0.4, 0.3, ...]   # normalized, ||emb|| = 1.0
doc2_emb = [0.1, 0.2, 0.8, ...]   # normalized, ||emb|| = 1.0

# Cosine similarity (after normalization)
cos_sim(query, doc1) = 0.78  # Good match
cos_sim(query, doc2) = 0.71  # Bad match
# Difference: only 0.07!  Hard to distinguish!
```

**Issues:**
- Narrow score range (typically 0.6-0.9)
- Magnitude wasted (always 1.0)
- Dimensional collapse
- Poor compositionality

### **Solution: Isotropic Gaussian:**

```python
# Embeddings NOT normalized, following N(0, I)
query_emb = [0.2, -0.5, 1.2, ...]  # ||emb|| ≈ 2.3 (uncertain)
doc1_emb = [0.3, -0.4, 1.0, ...]   # ||emb|| ≈ 2.8 (confident)
doc2_emb = [3.2, -1.5, 0.8, ...]   # ||emb|| ≈ 5.1 (very confident)

# Euclidean distance
euclidean_dist(query, doc1) = 0.5   # Good match!
euclidean_dist(query, doc2) = 4.2   # Bad match
# Difference: 8.4x larger separation!
```

**Benefits:**
- ✅ Wide score range (0 to ∞)
- ✅ Magnitude = confidence signal
- ✅ All dimensions used equally (isotropic)
- ✅ Composition works naturally
- ✅ Probabilistic interpretation (Gaussian likelihood)

---

## 📊 Key Files & Outputs

### **Trained Models:**
```
data/embeddings/
└── gaussian_embeddinggemma_final.pt   # Your trained model (download from Colab)
```

### **Evaluation Results:**
```
data/processed/
├── isotropy_comparison.png            # Distribution visualizations
├── isotropy_comparison.csv            # Statistical metrics
├── rag_performance_comparison.csv     # Retrieval metrics
├── rag_performance.png                # Performance charts
└── retriever_index.pkl                # Saved FAISS index (optional)
```

### **Logs:**
```
checkpoints/                            # Training checkpoints (in Colab)
├── best_model.pt                      # Best validation loss
└── epoch_*.pt                         # Periodic checkpoints
```

---

## 🔧 Hyperparameters (from LeJEPA paper)

### **Training:**
```python
# Model
output_dim = 512                    # Embedding dimension
freeze_early_layers = True          # Freeze first 4 transformer layers

# Optimizer
optimizer = AdamW
lr = 1e-5                          # Low LR for fine-tuning
weight_decay = 0.05                # From LeJEPA paper
batch_size = 16                    # Adjust based on GPU

# Loss weights
lambda_contrastive = 1.0           # Task loss weight
lambda_isotropy = 0.01             # LeJEPA loss weight (0.01-0.1 range)

# LeJEPA SIGReg
univariate_test = EppsPulley(num_points=17)
num_slices = 1024                  # Random projections

# Training
num_epochs = 5                     # Quick training
```

### **Retrieval:**
```python
# Distance metric: Euclidean (L2)
# NO normalization!
# Use FAISS IndexFlatL2 for efficient search
```

---

## 🐛 Troubleshooting

### **Training Issues:**

**Q: Out of memory in Colab?**
```python
# Reduce batch size
batch_size = 8  # Instead of 16

# Or reduce model size
output_dim = 256  # Instead of 512
```

**Q: Training too slow?**
```python
# Use smaller dataset
num_samples = 2000  # Instead of 5000

# Fewer epochs
num_epochs = 3  # Instead of 5
```

**Q: Isotropy not improving?**
```python
# Increase lambda_isotropy
lambda_isotropy = 0.05  # Instead of 0.01

# More slices
num_slices = 2048  # Instead of 1024
```

### **Retrieval Issues:**

**Q: Results worse than original?**
- Check isotropy metrics (run `evaluate_isotropy.ipynb`)
- Verify using Euclidean distance, NOT cosine
- Ensure embeddings are NOT normalized

**Q: FAISS not working?**
```python
# Use numpy fallback (slower but works)
# Automatically handled in GaussianRetriever
```

---

## 📚 Next Steps

### **Immediate:**
1. ✅ Run training notebook in Colab
2. ✅ Download and save model
3. ✅ Verify isotropy
4. ✅ Test retrieval
5. ✅ Evaluate performance

### **Advanced:**
1. **Fine-tune on your domain:**
   - Collect domain-specific query-doc pairs
   - Continue training with your data

2. **Optimize hyperparameters:**
   - Try different `lambda_isotropy` values
   - Experiment with `output_dim`
   - Test different `num_slices`

3. **Scale up:**
   - Use full MS MARCO dataset (500K examples)
   - Train for more epochs (10-20)
   - Use larger model (if you have A100)

4. **Production deployment:**
   - Quantize model (ONNX, TensorRT)
   - Build FAISS GPU index for speed
   - Add caching layer

---

## 📖 References

- **LeJEPA Paper:** [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- **EmbeddingGemma:** [HuggingFace](https://huggingface.co/google/embeddinggemma-300m)
- **MS MARCO:** [Dataset](https://microsoft.github.io/msmarco/)

---

## ✅ Quick Checklist

- [ ] Training notebook runs successfully
- [ ] Model downloaded to `data/embeddings/`
- [ ] Isotropy verified (mean ~0, cov ~I)
- [ ] Retrieval example works
- [ ] RAG evaluation shows improvement
- [ ] Results saved to `data/processed/`

🎉 **You're done! You now have state-of-the-art isotropic Gaussian embeddings for RAG!**
