# Documentation Index

## 📚 Core Documentation

| Document | Description |
|----------|-------------|
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | Complete training instructions for all approaches |
| **[AWS_SETUP.md](AWS_SETUP.md)** | AWS p4d setup, costs, and fast training |
| **[DATA_GUIDE.md](DATA_GUIDE.md)** | Data preparation and formats |
| **[API.md](API.md)** | Model usage, API reference, and examples |

---

## 🚀 Quick Start

1. **Test your setup:**
   ```bash
   ./scripts/run_preflight_tests.sh
   ```

2. **Download data:**
   ```bash
   python scripts/download_msmarco.py --output_dir data/processed/msmarco
   ```

3. **Train:**
   - **Local:** `./scripts/train_publication_recommended.sh` (15 days)
   - **AWS p4d:** `./scripts/train_parallel_p4d.sh` (1 day, $220)

4. **Evaluate:**
   ```bash
   ./scripts/evaluate_all_beir.sh
   ```

---

## 📖 Documentation by Topic

### **Getting Started**
- Start here: `../README.md`
- Training guide: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Data preparation: [DATA_GUIDE.md](DATA_GUIDE.md)

### **Training**
- Full training guide: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- AWS fast training: [AWS_SETUP.md](AWS_SETUP.md)
- Scripts documentation: `../scripts/README.md`

### **Usage**
- Model API: [API.md](API.md)
- Examples: See API.md for code examples
- Testing: `../tests/README.md`

### **Advanced**
- AWS setup details: [AWS_SETUP.md](AWS_SETUP.md)
- Custom data: [DATA_GUIDE.md](DATA_GUIDE.md)
- Multi-GPU training: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## 📁 Documentation Structure

```
docs/
├── README.md              # This file
├── TRAINING_GUIDE.md      # Complete training instructions
├── AWS_SETUP.md           # AWS p4d setup and costs
├── DATA_GUIDE.md          # Data preparation
├── API.md                 # Model API and examples
└── archive/               # Old documentation (for reference)
```

---

## ⚡ Most Common Tasks

### **I want to train a model**
→ Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### **I want to train fast on AWS**
→ Read [AWS_SETUP.md](AWS_SETUP.md)

### **I need to prepare data**
→ Read [DATA_GUIDE.md](DATA_GUIDE.md)

### **I want to use a trained model**
→ Read [API.md](API.md)

### **I want to test everything first**
→ Run `./scripts/run_preflight_tests.sh`

---

## 🔄 Document Updates

- **2025-11-15:** Consolidated all documentation
  - Moved obsolete docs to `archive/`
  - Created 4 core documents
  - Streamlined navigation

---

## 📞 Need Help?

- Check the specific guide for your task (see above)
- Run tests: `./scripts/run_preflight_tests.sh`
- Review examples in [API.md](API.md)
- Check archive for historical documentation

---

**Start with:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

