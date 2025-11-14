# GPU vs TPU for LeJEPA Training

## Quick Decision Guide

### **Use GPU (Recommended for This Project)** ✅

**Choose GPU if:**
- ✅ You're training on 5K-10K samples (like default setup)
- ✅ Batch size is 16-32
- ✅ You want simplest setup
- ✅ You want best library compatibility

**Notebooks:**
- `notebooks/lejepa_training.ipynb` (GPU version)

---

### **Use TPU (Advanced)**

**Choose TPU if:**
- ✅ Training on 50K+ samples
- ✅ Want batch size 128-256
- ✅ GPUs are busy/unavailable
- ✅ Comfortable with torch_xla

**Notebooks:**
- `notebooks/lejepa_training_tpu.ipynb` (TPU version)

---

## Detailed Comparison

### **1. Performance**

| Metric | GPU (T4) | GPU (V100) | GPU (A100) | TPU (v2-8) |
|--------|----------|------------|------------|------------|
| **Optimal batch size** | 16 | 32 | 64 | 128-256 |
| **Training time (5K samples)** | 2-3 hours | 1-2 hours | 30-60 min | 1-2 hours* |
| **Training time (50K samples)** | 20+ hours | 10-15 hours | 5-8 hours | **3-5 hours** |
| **Peak memory** | 16GB | 32GB | 40GB | 64GB (8x8GB) |
| **Availability (Colab)** | Medium | Low | Very Low | High |

*TPU slower on small batches, faster on large batches

### **2. Ease of Use**

| Aspect | GPU | TPU |
|--------|-----|-----|
| **Setup** | ✅ Simple | ⚠️ Requires torch_xla |
| **Code changes** | ✅ None | ⚠️ Moderate |
| **Debugging** | ✅ Easy | ❌ Harder |
| **Library support** | ✅ 100% | ⚠️ ~90% |
| **Learning curve** | ✅ Low | ⚠️ Medium |

### **3. Compatibility**

| Library/Feature | GPU | TPU |
|----------------|-----|-----|
| PyTorch | ✅ Full | ✅ Via torch_xla |
| FAISS | ✅ Works | ❌ CPU fallback only |
| LeJEPA | ✅ Works | ✅ Works |
| Sentence Transformers | ✅ Works | ✅ Works |
| Debugging tools | ✅ Full | ⚠️ Limited |

### **4. Cost (Google Colab)**

| Tier | GPU | TPU |
|------|-----|-----|
| **Free** | T4 (limited hours) | v2-8 (limited hours) |
| **Colab Pro ($10/mo)** | V100/A100 | v2-8 |
| **Colab Pro+ ($50/mo)** | A100 (more hours) | v3-8 |

---

## Code Differences

### **GPU Version (Simple)**

```python
# Standard PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GaussianEmbeddingGemma().to(device)

# Standard training loop
for batch in dataloader:
    loss = compute_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # ← Standard
```

### **TPU Version (More Complex)**

```python
# Need torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()  # ← Different
model = GaussianEmbeddingGemma().to(device)

# Need ParallelLoader
para_loader = pl.ParallelLoader(dataloader, [device])
loader_tpu = para_loader.per_device_loader(device)

for batch in loader_tpu:  # ← Different
    loss = compute_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    xm.optimizer_step(optimizer)  # ← Different (syncs gradients)
```

---

## When Each Excels

### **GPU Wins:**

1. **Small to medium datasets** (5K-20K samples)
   - T4 is sufficient
   - Simple code
   - Fast iteration

2. **Small batch sizes** (16-32)
   - TPU underutilized
   - GPU more efficient

3. **Prototyping**
   - Easier debugging
   - Better library support
   - Familiar workflow

4. **FAISS indexing**
   - GPU FAISS is very fast
   - TPU doesn't support FAISS

### **TPU Wins:**

1. **Large datasets** (50K+ samples)
   - 2-3x faster than V100
   - Better parallelism

2. **Large batch sizes** (128-256)
   - TPU designed for this
   - More memory efficient

3. **Matrix-heavy operations**
   - Large matrix multiplications
   - Transformer layers

4. **When GPUs unavailable**
   - TPUs often available when GPUs busy

---

## Recommendation for This Project

### **Default Setup (5K samples, batch 16-32):**

```bash
✅ USE GPU (T4 or V100)
```

**Reasons:**
- Simpler code
- Sufficient speed (2-3 hours)
- Better library compatibility
- Easier debugging

### **If Scaling Up (50K+ samples, batch 128+):**

```bash
✅ USE TPU (v2-8)
```

**Reasons:**
- 2-3x faster
- Handles large batches better
- More memory

---

## How to Switch

### **From GPU to TPU:**

1. **Upload TPU notebook:**
   - Use `notebooks/lejepa_training_tpu.ipynb`

2. **Change runtime:**
   - Runtime → Change runtime type → **TPU**

3. **Increase batch size:**
   ```python
   batch_size = 128  # Instead of 16
   ```

4. **Run all cells**

### **From TPU to GPU:**

1. **Upload GPU notebook:**
   - Use `notebooks/lejepa_training.ipynb`

2. **Change runtime:**
   - Runtime → Change runtime type → **T4 GPU**

3. **Use smaller batch size:**
   ```python
   batch_size = 16  # Instead of 128
   ```

4. **Run all cells**

---

## Troubleshooting

### **GPU Issues:**

**Out of memory?**
```python
# Reduce batch size
batch_size = 8  # Instead of 16

# Or reduce output dim
output_dim = 256  # Instead of 512
```

**Slow training?**
```python
# Upgrade to V100 or A100 (Colab Pro)
# Or reduce dataset size for quick experiment
num_samples = 2000
```

### **TPU Issues:**

**Slow training?**
```python
# Increase batch size (TPU loves large batches!)
batch_size = 256  # Instead of 128

# Ensure using ParallelLoader
para_loader = pl.ParallelLoader(dataloader, [device])
```

**Library not working?**
```python
# Some ops not supported on TPU
# Check torch_xla documentation
# Or fall back to GPU
```

---

## Quick Start Commands

### **For GPU (Recommended):**

```bash
# In Colab:
# 1. Upload notebooks/lejepa_training.ipynb
# 2. Runtime → Change runtime type → T4 GPU
# 3. Run all cells
```

### **For TPU (Advanced):**

```bash
# In Colab:
# 1. Upload notebooks/lejepa_training_tpu.ipynb
# 2. Runtime → Change runtime type → TPU
# 3. Run all cells (takes longer to setup torch_xla)
```

---

## Summary Table

| Use Case | Recommended | Batch Size | Training Time |
|----------|------------|------------|---------------|
| **Quick experiment (2K samples)** | T4 GPU | 16 | 30-60 min |
| **Default (5K samples)** | **T4/V100 GPU** | 16-32 | 1-3 hours |
| **Medium (20K samples)** | V100/A100 GPU | 32-64 | 3-6 hours |
| **Large (50K+ samples)** | **TPU v2-8** | 128-256 | 3-5 hours |
| **Production scale (500K samples)** | TPU v3-8 / A100 | 256+ | 10-20 hours |

---

## Bottom Line

**For this project (default 5K samples):**

```
✅ Use GPU (T4/V100) - Simpler and sufficient
❌ Don't use TPU unless you scale up significantly
```

**Only switch to TPU if:**
- Training on 50K+ samples
- Using batch size 128+
- Need faster training at scale
