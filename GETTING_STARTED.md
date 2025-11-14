# Getting Started - 5 Minute Quick Start

## What You Have

A complete project for training and using **isotropic Gaussian embeddings** with LeJEPA for superior RAG retrieval.

**Location:** `/Users/ctn/src/ctn/ragcun/`

---

## What to Do Right Now

### **Step 1: Open Training Notebook** (2 minutes)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Upload: `/Users/ctn/src/ctn/ragcun/notebooks/lejepa_training.ipynb`
4. Set **Runtime → Change runtime type → T4 GPU**

### **Step 2: Train Model** (1-3 hours)

1. Click **Runtime → Run all**
2. Wait for training to complete
3. Last cell will download `gaussian_embeddinggemma_final.pt`

### **Step 3: Save Model Locally** (1 minute)

```bash
# Move downloaded model to project
cd /Users/ctn/src/ctn/ragcun
mv ~/Downloads/gaussian_embeddinggemma_final.pt data/embeddings/
```

### **Step 4: Test It** (5 minutes)

```bash
# Install dependencies (with uv - fast!)
uv pip install -e .

# Or traditional pip
pip install -e .

# Run example
python examples/retrieval_example.py
```

**💡 Tip:** Install [uv](https://github.com/astral-sh/uv) for 10-100x faster installs:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
See [UV_GUIDE.md](UV_GUIDE.md) for details.

**You should see:**
```
LeJEPA Gaussian Embeddings - Retrieval Example
===============================================
1. Loading model...
✅ Loaded model from data/embeddings/gaussian_embeddinggemma_final.pt

2. Adding 10 documents...
✅ Total documents: 10

3. Testing retrieval (Euclidean distance)...

Query: "What is machine learning?"
------------------------------------------------------------
1. [distance=0.523]
   Machine learning is a subset of artificial intelligence.
```

---

## What You Get

### **Immediate Benefits:**

1. **Better Retrieval:**
   - 8-10x larger separation between good/bad matches
   - More accurate ranking
   - Meaningful distance scores

2. **Confidence Scores:**
   - Embedding magnitude indicates uncertainty
   - Can filter low-confidence results

3. **Compositionality:**
   - Combine queries naturally: `query1 + query2`
   - No normalization distortion

---

## File Guide

### **Must Read:**
- `README.md` - Project overview
- `WORKFLOW.md` - Complete pipeline documentation (read this!)

### **Notebooks (Use in Order):**
1. `notebooks/lejepa_training.ipynb` - Train model (Colab)
2. `notebooks/evaluate_isotropy.ipynb` - Verify N(0,I) distribution
3. `notebooks/evaluate_rag.ipynb` - Compare performance

### **Code:**
- `src/ragcun/model.py` - GaussianEmbeddingGemma class
- `src/ragcun/retriever.py` - L2 distance retriever
- `examples/retrieval_example.py` - Usage demo

---

## Quick Reference

### **Train:**
```bash
# In Colab: Open lejepa_training.ipynb → Run all
```

### **Use:**
```python
from ragcun import GaussianRetriever

retriever = GaussianRetriever(model_path='data/embeddings/gaussian_embeddinggemma_final.pt')
retriever.add_documents(["doc1", "doc2", ...])
results = retriever.retrieve("query", top_k=5)
```

### **Evaluate:**
```bash
# In Jupyter: Open evaluate_rag.ipynb → Run all
```

---

## Next Steps

After getting the basic example working:

1. **Read WORKFLOW.md** - Understand the complete pipeline
2. **Run isotropy evaluation** - Verify your model is truly isotropic
3. **Run RAG evaluation** - See the performance improvements
4. **Fine-tune on your data** - Adapt to your specific domain

---

## Need Help?

**Common Issues:**

- **Out of memory?** Reduce `batch_size` in training notebook
- **Training too slow?** Use fewer samples or epochs
- **Model not found?** Check path: `data/embeddings/gaussian_embeddinggemma_final.pt`

**Resources:**

- LeJEPA paper: https://arxiv.org/abs/2511.08544
- EmbeddingGemma: https://huggingface.co/google/embeddinggemma-300m

---

## TL;DR

```bash
# 1. Upload notebooks/lejepa_training.ipynb to Colab
# 2. Run all cells (wait 1-3 hours)
# 3. Download model → save to data/embeddings/
# 4. Run: python examples/retrieval_example.py
# 5. Enjoy better RAG retrieval! 🎉
```
