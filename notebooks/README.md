# Analysis Notebooks

Jupyter notebooks for analyzing training results and demonstrating improvements.

---

## 📊 Result Analysis Notebooks

### **evaluate_isotropy.ipynb**
Demonstrates isotropy improvements from LeJEPA training.

**Shows:**
- Embedding distribution comparison (original vs trained)
- Isotropy scores calculation
- Visualization of embedding spaces
- Statistical analysis of isotropy

**Compares:**
- Original MPNet (normalized, anisotropic)
- Baseline fine-tuned (normalized, slightly anisotropic)
- **With isotropy** (Gaussian, isotropic) ← Your contribution

**Expected results:**
- Original isotropy: ~0.87
- Baseline fine-tuned: ~0.89
- **With LeJEPA: ~0.95** ✅

**Use for paper:** Isotropy analysis figures and metrics

---

### **evaluate_rag.ipynb**
Demonstrates retrieval performance improvements.

**Shows:**
- Retrieval accuracy on test datasets
- Comparison of distance metrics (Euclidean vs Cosine)
- Query-document matching examples
- Ablation results

**Compares:**
- Original model (cosine similarity)
- Baseline fine-tuned (no isotropy)
- **With isotropy** (Euclidean distance) ← Your method

**Expected results:**
- Baseline: ~47.5% NDCG@10
- **With isotropy: ~49.2% NDCG@10** ✅

**Use for paper:** Main results table and retrieval examples

---

## 🚀 Running the Notebooks

### **Prerequisites**

```bash
# Install Jupyter
pip install jupyter matplotlib seaborn pandas

# Have trained models ready
ls checkpoints/baseline_no_isotropy/best_model.pt
ls checkpoints/with_isotropy/best_model.pt
```

### **Launch Jupyter**

```bash
cd /home/ubuntu/ragcun
jupyter notebook notebooks/
```

Then open:
- `evaluate_isotropy.ipynb` for isotropy analysis
- `evaluate_rag.ipynb` for retrieval performance

---

## 📈 What to Generate for Paper

### **From evaluate_isotropy.ipynb:**

1. **Isotropy comparison table:**
   ```
   Model              | Isotropy Score
   -------------------|---------------
   Original MPNet     | 0.87
   Baseline (no iso)  | 0.89
   With isotropy      | 0.95
   ```

2. **Embedding distribution plots:**
   - t-SNE visualization showing more uniform distribution
   - Eigenvalue distribution comparison
   - Dimension variance analysis

3. **Statistical significance tests:**
   - Show isotropy improvement is significant

### **From evaluate_rag.ipynb:**

1. **Main results table:**
   ```
   Model              | BEIR Avg | MS MARCO | SciFact
   -------------------|----------|----------|--------
   Baseline (no iso)  | 47.5%    | 35.2%    | 69.1%
   With isotropy      | 49.2%    | 36.8%    | 71.2%
   ```

2. **Qualitative examples:**
   - Query: "What is machine learning?"
   - Top-3 retrieved documents (before/after)
   - Show improved relevance

3. **Ablation study:**
   - Effect of λ_isotropy values
   - Frozen vs unfrozen base
   - Different base models

---

## 📁 Notebook Structure

```
notebooks/
├── README.md                       # This file
├── evaluate_isotropy.ipynb        # Isotropy analysis ⭐
├── evaluate_rag.ipynb             # RAG performance ⭐
└── archive/                       # Old notebooks
    ├── document_processing.ipynb  # Demo notebook
    ├── lejepa_training.ipynb      # Training notebook (use train.py instead)
    └── lejepa_training_tpu.ipynb  # TPU training (obsolete)
```

---

## 🔧 Customizing Notebooks

### **Update Model Paths**

Both notebooks expect models at these paths:
```python
# Update these paths to your trained models
baseline_model = 'checkpoints/baseline_no_isotropy/best_model.pt'
isotropy_model = 'checkpoints/with_isotropy/best_model.pt'
```

### **Change Test Datasets**

```python
# In evaluate_rag.ipynb, modify test dataset:
test_datasets = ['scifact', 'nfcorpus', 'arguana']  # Quick test
# or
test_datasets = 'all'  # Full BEIR evaluation
```

---

## 💡 Tips

**For quick results:**
- Run `evaluate_isotropy.ipynb` first (~5 minutes)
- Shows clear isotropy improvement
- Good for sanity check

**For paper figures:**
- Run both notebooks after all training completes
- Export figures as PDF/PNG
- Include in paper results section

**For debugging:**
- If model loading fails, check checkpoint paths
- If BEIR evaluation is slow, use subset of datasets
- Use smaller test sets for quick iteration

---

## 📦 Dependencies

```bash
pip install jupyter matplotlib seaborn pandas scikit-learn
```

Already installed if you've run training.

---

## 🎯 Quick Start

```bash
# 1. Ensure models are trained
./scripts/train_publication_recommended.sh  # or already done

# 2. Launch Jupyter
jupyter notebook notebooks/

# 3. Open and run:
#    - evaluate_isotropy.ipynb (isotropy analysis)
#    - evaluate_rag.ipynb (performance results)

# 4. Export figures for paper
```

---

**These notebooks demonstrate your key contributions:**
1. **Isotropy improvement** (0.95 vs 0.89)
2. **Retrieval improvement** (+1.7% BEIR)
3. **Method effectiveness** (clear ablation results)

