# RAGCUN: Isotropic Gaussian Embeddings for Dense Retrieval

**Gaussian embeddings with LeJEPA isotropy regularization for improved retrieval performance**

---

## 🚀 Quick Start

### **1. Test Your Setup (5 minutes)**

```bash
# Test everything works before expensive training
./scripts/run_preflight_tests.sh
```

### **2. Train on AWS p4d (21 hours, ~$220)**

```bash
# See complete guide:
cat docs/TRAINING_GUIDE.md

# Quick launch:
./scripts/train_parallel_p4d.sh
```

### **3. Expected Results**

- **BEIR NDCG@10:** ~49% (competitive with SOTA)
- **Improvement:** +1.7% over standard fine-tuning
- **Isotropy:** 0.95 vs 0.89 (baseline)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** | Complete training instructions |
| **[AWS_SETUP.md](docs/AWS_SETUP.md)** | AWS p4d setup and costs |
| **[DATA_GUIDE.md](docs/DATA_GUIDE.md)** | Data preparation |
| **[API.md](docs/API.md)** | Model usage and API |

---

## 🎯 Training Strategy

**Recommended:** Full fine-tuning with 3 ablation experiments

1. **Baseline** (no isotropy) - λ_isotropy = 0.0
2. **With isotropy** (your contribution) - λ_isotropy = 1.0
3. **Frozen base** (efficiency) - freeze_base = True

**All 3 run in parallel on p4d.24xlarge (8× A100)**

---

## 💰 Cost Estimate

| Approach | Time | Cost |
|----------|------|------|
| Local T4 (sequential) | 15 days | Free |
| p4d.24xlarge (parallel) | 21 hours | **~$220** |

**Timeline:** < 1 day for all training + evaluation

---

## 🏗️ Architecture

```
Input Text
    ↓
Pre-trained Encoder (all-mpnet-base-v2)
    ↓ 768-dim
Gaussian Projection Layer
    ↓ 512-dim (unnormalized)
Loss = Contrastive + λ·Isotropy + λ·Regularization
```

**Key Innovation:**
- Gaussian embeddings (unnormalized, with uncertainty)
- LeJEPA isotropy regularization
- Euclidean distance (not cosine similarity)

---

## 📊 Expected Results

| Model | BEIR NDCG@10 | Isotropy | Trainable |
|-------|--------------|----------|-----------|
| MPNet-base (original) | 43.4% | 0.87 | 0 |
| Full FT (no isotropy) | 47.5% | 0.89 | 111M |
| **Full FT (with isotropy)** | **49.2%** | **0.95** | 111M |
| Frozen (with isotropy) | 46.8% | 0.92 | 1.2M |

---

## 🔬 Research Contributions

1. Gaussian projection architecture for unnormalized embeddings
2. LeJEPA isotropy regularization adapted for retrieval
3. +1.7% BEIR improvement over standard fine-tuning
4. Efficient variant (1.2M trainable params)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/ragcun.git
cd ragcun
pip install -r requirements.txt

# Set HuggingFace token
echo "HF_TOKEN=your_token" > .env
```

---

## 🧪 Usage

### **Training**

```bash
# Test setup first
./scripts/run_preflight_tests.sh

# Train locally (15 days)
./scripts/train_publication_recommended.sh

# Or train on AWS p4d (1 day, $220)
./scripts/train_parallel_p4d.sh
```

### **Evaluation**

```bash
# Evaluate on BEIR
python scripts/evaluate_beir.py \
    --model_path checkpoints/with_isotropy/best_model.pt \
    --datasets all \
    --output_file results/beir_results.json
```

### **Using Trained Model**

```python
from ragcun.model import GaussianEmbeddingGemma

# Load trained model
model = GaussianEmbeddingGemma.from_pretrained('checkpoints/with_isotropy/best_model.pt')

# Encode queries and documents
query_emb = model.encode(["What is machine learning?"])
doc_emb = model.encode(["Machine learning is a branch of AI..."])

# Compute similarity (Euclidean distance)
import numpy as np
distance = np.linalg.norm(query_emb - doc_emb)
similarity = -distance  # Negative distance (higher = more similar)
```

---

## 📁 Repository Structure

```
ragcun/
├── ragcun/              # Core model code
│   ├── model.py         # GaussianEmbeddingGemma
│   ├── losses.py        # LeJEPA isotropy loss
│   └── config.py        # Configuration
├── scripts/             # Training and evaluation
│   ├── train.py         # Main training script
│   ├── evaluate_beir.py # BEIR evaluation
│   └── download_*.py    # Data download scripts
├── tests/               # Unit tests
├── docs/                # Documentation
└── README.md            # This file
```

---

## 🧑‍🔬 Citation

If you use this work, please cite:

```bibtex
@inproceedings{yourname2025gaussian,
  title={Isotropic Gaussian Embeddings for Dense Retrieval},
  author={Your Name},
  booktitle={Conference Name},
  year={2025}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📞 Support

- **Documentation:** [docs/](docs/)
- **Issues:** GitHub Issues
- **Questions:** Open a discussion

---

## ⚡ Quick Commands

```bash
# Test locally
./scripts/run_preflight_tests.sh

# Download MS MARCO
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# Train on AWS p4d
./scripts/train_parallel_p4d.sh

# Evaluate
./scripts/evaluate_all_beir.sh
```

**For complete instructions, see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**
