# Scripts Directory

Essential scripts for training and evaluation.

---

## 🧪 Testing Scripts

### **run_preflight_tests.sh** ⭐
Pre-flight testing before expensive training runs.

```bash
./scripts/run_preflight_tests.sh
```

**Tests:**
- Dependencies installed
- Model loads correctly
- All 3 training configs work
- Training loop starts
- Checkpoints save properly

**Time:** 5 minutes  
**Run before:** Any training, especially AWS

---

### **test_training_setup.sh**
Comprehensive setup verification (for AWS instances).

```bash
./scripts/test_training_setup.sh
```

**Tests:**
- GPU availability (expects 8 for p4d)
- Large batch sizes (64 on A100)
- S3 access
- Complete training pipeline

**Time:** 5 minutes  
**Run on:** p4d instance after setup

---

## 📥 Data Scripts

### **download_msmarco.py**
Download MS MARCO passage ranking dataset.

```bash
# Full dataset (500K pairs)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# Small subset for testing
python scripts/download_msmarco.py \
    --output_dir data/processed/msmarco_test \
    --max_train_samples 10000
```

**Output:**
- `train.json` (~2GB, 502K examples)
- `dev.json` (~50MB, 6.9K examples)

---

### **download_wiki.py**
Download Wikipedia passages (optional, for unsupervised pre-training).

```bash
python scripts/download_wiki.py \
    --num_passages 100000 \
    --output data/raw/wiki_100k.txt
```

---

### **prepare_data.py**
Prepare custom training data.

```bash
python scripts/prepare_data.py \
    --documents data/raw/docs.txt \
    --generate_pairs \
    --num_pairs 100000 \
    --output_dir data/processed/custom
```

**Use:** Creating custom datasets beyond MS MARCO

---

## 🏋️ Training Scripts

### **train.py** ⭐
Main training script - all experiments use this.

```bash
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/model
```

**Key Arguments:**
- `--freeze_base`: Freeze base encoder (True/False)
- `--lambda_isotropy`: Isotropy loss weight (0.0 = off, 1.0 = default)
- `--lambda_reg`: Regularization weight
- `--mixed_precision`: Enable FP16 training
- `--base_learning_rate`: LR for base encoder
- `--projection_learning_rate`: LR for projection layer

---

### **train_publication_recommended.sh**
Sequential training of all 3 experiments (local use).

```bash
./scripts/train_publication_recommended.sh
```

**Trains:**
1. Baseline (no isotropy)
2. With isotropy (your method)
3. Frozen base (efficiency)

**Time:** 15 days on single T4  
**Use:** Local GPU training

---

### **train_parallel_p4d.sh**
Parallel training of all 3 experiments (AWS use).

```bash
./scripts/train_parallel_p4d.sh
```

**Trains:** All 3 experiments in parallel on different GPUs  
**Time:** 18 hours on p4d.24xlarge  
**Cost:** ~$189  
**Use:** AWS p4d fast training

---

## 📊 Evaluation Scripts

### **evaluate_beir.py** ⭐
Evaluate model on BEIR benchmark.

```bash
# Single dataset
python scripts/evaluate_beir.py \
    --model_path checkpoints/model.pt \
    --datasets scifact \
    --output_file results/beir_scifact.json

# Multiple datasets
python scripts/evaluate_beir.py \
    --model_path checkpoints/model.pt \
    --datasets scifact nfcorpus arguana \
    --output_file results/beir_results.json

# All 18 BEIR datasets
python scripts/evaluate_beir.py \
    --model_path checkpoints/model.pt \
    --datasets all \
    --output_file results/beir_full.json
```

**Time:** ~3-4 hours for all 18 datasets on GPU

---

### **evaluate_all_beir.sh**
Evaluate all trained models on BEIR.

```bash
./scripts/evaluate_all_beir.sh
```

**Evaluates:**
- Original MPNet baseline
- Baseline (no isotropy)
- With isotropy (your method)
- Frozen base (efficiency)

**Time:** ~12-16 hours total  
**Use:** After all training completes

---

## 🔧 Setup Scripts

### **setup_p4d_instance.sh**
Setup AWS p4d instance after launch.

```bash
./scripts/setup_p4d_instance.sh
```

**Does:**
- Installs dependencies
- Verifies 8 GPUs
- Sets up S3 backups
- Prepares directories
- Configures environment

**Time:** 10-15 minutes  
**Use:** Once after SSH into p4d instance

---

## 🗂️ Script Organization

### **Essential Scripts (11 total)**

**Testing:**
- `run_preflight_tests.sh` - Pre-flight testing
- `test_training_setup.sh` - AWS setup verification

**Data:**
- `download_msmarco.py` - MS MARCO download
- `download_wiki.py` - Wikipedia download (optional)
- `prepare_data.py` - Custom data preparation

**Training:**
- `train.py` - Main training script
- `train_publication_recommended.sh` - Local sequential training
- `train_parallel_p4d.sh` - AWS parallel training

**Evaluation:**
- `evaluate_beir.py` - BEIR evaluation
- `evaluate_all_beir.sh` - Evaluate all models

**Setup:**
- `setup_p4d_instance.sh` - AWS instance setup

---

## 📋 Typical Workflow

### **Local Training**

```bash
# 1. Test setup
./scripts/run_preflight_tests.sh

# 2. Download data
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 3. Train all models
./scripts/train_publication_recommended.sh

# 4. Evaluate
./scripts/evaluate_all_beir.sh
```

### **AWS p4d Training**

```bash
# On local machine:
# 1. Test setup first
./scripts/run_preflight_tests.sh

# 2. Launch p4d instance (AWS Console)

# On p4d instance:
# 3. Setup
./scripts/setup_p4d_instance.sh

# 4. Test
./scripts/test_training_setup.sh

# 5. Download data
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 6. Train (in tmux!)
tmux new -s training
./scripts/train_parallel_p4d.sh

# 7. Evaluate
./scripts/evaluate_all_beir.sh

# 8. Download results and terminate
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
aws ec2 terminate-instances --instance-ids <id>
```

---

## 🔍 Quick Reference

| Task | Script |
|------|--------|
| Test before training | `run_preflight_tests.sh` |
| Download MS MARCO | `download_msmarco.py` |
| Train locally | `train_publication_recommended.sh` |
| Train on AWS | `train_parallel_p4d.sh` |
| Evaluate single model | `evaluate_beir.py` |
| Evaluate all models | `evaluate_all_beir.sh` |
| Setup AWS instance | `setup_p4d_instance.sh` |
| Test AWS setup | `test_training_setup.sh` |

---

## 📦 Archive

Old/obsolete scripts moved to `archive/` for reference:
- Numbered workflow scripts (0_, 1_, 2_, etc.)
- Pipeline scripts (98_, 99_)
- Legacy training scripts
- Old evaluation scripts

These are kept for historical reference but not needed for current workflows.

---

## 💡 Tips

**Before any training:**
```bash
./scripts/run_preflight_tests.sh  # Always run this first!
```

**For fast AWS training:**
```bash
./scripts/train_parallel_p4d.sh  # 18 hours on p4d
```

**For local training:**
```bash
./scripts/train_publication_recommended.sh  # 15 days on T4
```

**For evaluation:**
```bash
./scripts/evaluate_all_beir.sh  # Evaluate everything
```

---

For more details, see:
- Training guide: `../docs/TRAINING_GUIDE.md`
- AWS setup: `../docs/AWS_SETUP.md`
- API reference: `../docs/API.md`
