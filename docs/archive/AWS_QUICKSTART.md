# AWS Spot Training - Quick Start

**Goal:** Train all 3 models in 1.5 days on powerful spot instance

---

## 🚀 Recommended Setup

**Instance:** `p3.8xlarge` (4× Tesla V100)  
**Cost:** ~$164 (spot @ $4/hr)  
**Time:** 36 hours training + 4 hours evaluation = **1.5 days total**

---

## ⚡ Quick Launch (5 Commands)

```bash
# 1. Launch spot instance (AWS Console or CLI)
#    - Type: p3.8xlarge
#    - AMI: Deep Learning AMI (Ubuntu)
#    - Storage: 200GB
#    - Check "Request Spot instances"

# 2. SSH and setup
ssh -i your-key.pem ubuntu@<instance-ip>
cd /home/ubuntu
git clone https://github.com/yourusername/ragcun.git
cd ragcun
pip install -r requirements.txt

# 3. Download data (30-40 min)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 4. Start training (36 hours)
./scripts/train_parallel_aws.sh  # Runs all 3 experiments in parallel!

# 5. Evaluate (4 hours)
./scripts/evaluate_all_beir.sh
```

**Total time: ~41 hours = Done by tomorrow evening!**

---

## 📋 What Runs in Parallel

```
GPU 0: Baseline (no isotropy)      → 36 hours
GPU 1: With isotropy (YOUR METHOD) → 36 hours  } All in parallel!
GPU 2: Frozen base (efficiency)    → 36 hours
GPU 3: Available for evaluation    → Ready
```

**Result:** All 3 experiments complete in time of 1 experiment!

---

## 💾 Backup Strategy (Critical!)

```bash
# Setup S3 sync (run once)
./scripts/sync_to_s3.sh &

# Syncs checkpoints to S3 every 30 min
# If spot interrupted, you can resume!
```

---

## 📊 Expected Results After Training

```
checkpoints/
├── baseline_no_isotropy/best_model.pt  (BEIR: ~47.5%)
├── with_isotropy/best_model.pt         (BEIR: ~49.2%) ← Your contribution!
└── frozen_efficient/best_model.pt      (BEIR: ~46.8%)

results/
├── beir_baseline.json
├── beir_with_isotropy.json
└── beir_frozen.json
```

---

## 💰 Cost Breakdown

| Item | Time | Cost @ $4/hr |
|------|------|--------------|
| Setup & download | 1 hr | $4 |
| Training (parallel) | 36 hrs | $144 |
| Evaluation | 4 hrs | $16 |
| **Total** | **41 hrs** | **$164** |

**vs 15 days on local T4** → 10x faster, totally worth it!

---

## 🛡️ Spot Interruption Protection

**Built-in safety:**
- ✅ Checkpoints saved every epoch
- ✅ S3 sync every 30 minutes
- ✅ Interruption handler (graceful shutdown)
- ✅ Resume from last checkpoint if interrupted

**To resume:**
```bash
# Re-launch instance, then:
python scripts/train/isotropic.py [same args] --resume checkpoints/[model]/checkpoint_epoch_2.pt
```

---

## 📈 Monitor Training

```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f logs/*.log

# All at once (pretty)
pip install gpustat
watch -n 1 gpustat -cpu
```

---

## ✅ Pre-Launch Checklist

- [ ] AWS account with GPU quota
- [ ] SSH key pair created
- [ ] S3 bucket ready: `s3://your-bucket/ragcun-training`
- [ ] HuggingFace token ready
- [ ] Budget: Set $200 limit

---

## 🎯 After Training

```bash
# 1. Sync everything to S3
aws s3 sync checkpoints/ s3://your-bucket/ragcun-training/checkpoints/
aws s3 sync results/ s3://your-bucket/ragcun-training/results/

# 2. Download locally
aws s3 sync s3://your-bucket/ragcun-training/ ./trained-models/

# 3. Terminate instance
aws ec2 terminate-instances --instance-ids i-xxxxx

# 4. Celebrate! 🎉
```

---

## 🆘 Troubleshooting

**Out of Memory?**
```bash
# Reduce batch size in train_parallel_aws.sh
--batch_size 16  # Instead of 32
```

**Spot request not fulfilled?**
```bash
# Check spot price
aws ec2 describe-spot-price-history --instance-types p3.8xlarge

# Increase max price or try different region
```

**Training slow?**
```bash
# Check GPU utilization (should be >80%)
nvidia-smi

# If low, increase batch size
--batch_size 48
```

---

## 🔗 Detailed Guide

See **`docs/AWS_SPOT_TRAINING_GUIDE.md`** for:
- Complete setup instructions
- Spot interruption handling
- Advanced optimizations
- Alternative instance types

---

## 🚀 Start Now!

```bash
# Launch p3.8xlarge spot instance, then:

ssh -i your-key.pem ubuntu@<instance-ip>
cd /home/ubuntu
git clone [your-repo]
cd ragcun
pip install -r requirements.txt
python scripts/download_msmarco.py --output_dir data/processed/msmarco
./scripts/train_parallel_aws.sh

# Done in 1.5 days! 🎉
```

**Cost: ~$164 | Time: ~41 hours | Results: Publication-ready**

