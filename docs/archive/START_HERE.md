# START HERE: Training Path to Publication

**Last Updated:** November 15, 2025  
**Goal:** Get publication-ready results in < 1 day using AWS p4d.24xlarge

---

## 🎯 The Plan

1. **Test locally** (2-3 hours) - Make sure everything works
2. **Launch p4d** ($220, 21 hours) - Train all models in parallel
3. **Publish** - Write paper with results

**Total cost:** ~$220  
**Total time:** < 2 days wall-clock

---

## 📋 Step-by-Step Checklist

### **Phase 1: Local Testing (DO THIS NOW)** ✅

**Location:** Your current machine (`/home/ubuntu/ragcun`)  
**Time:** 5 minutes (without data download)  
**Cost:** $0

```bash
cd /home/ubuntu/ragcun

# Run complete pre-flight tests
./scripts/run_preflight_tests.sh
```

**What this tests:**
- ✅ All scripts exist and work
- ✅ Dependencies installed correctly
- ✅ Model loads (frozen and unfrozen)
- ✅ All 3 training configs work
- ✅ Checkpointing works

**Expected output:**
```
✅✅✅ ALL TESTS PASSED ✅✅✅
Your code is ready for p4d.24xlarge training!
```

**If tests fail:**
- Fix issues (check error messages)
- Re-run `./scripts/run_preflight_tests.sh`
- Don't proceed until all tests pass

---

### **Phase 2: Download Full MS MARCO (Optional Now, Required Later)**

**Location:** Your current machine OR p4d instance  
**Time:** 1-2 hours (depending on connection)  
**Cost:** $0

```bash
# Download full MS MARCO (500K examples)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# This creates:
#   data/processed/msmarco/train.json (~2GB)
#   data/processed/msmarco/dev.json (~50MB)
```

**Options:**
- **Option A:** Download now on your machine (slower connection)
- **Option B:** Download on p4d instance (faster, but uses p4d time)

**Recommendation:** If you have time, download now to save p4d costs.

---

### **Phase 3: Launch p4d.24xlarge**

**Location:** AWS Console or CLI  
**Time:** 5 minutes to launch  
**Cost:** ~$10.50/hour (spot pricing)

#### **Option A: AWS Console (Easier)**

1. Go to EC2 → Launch Instance
2. **Name:** `ragcun-training`
3. **AMI:** Deep Learning AMI (Ubuntu 20.04)
4. **Instance type:** `p4d.24xlarge`
5. **Check:** ✅ Request Spot instances
6. **Max price:** $15.00/hour
7. **Storage:** 300 GB gp3
8. **Key pair:** Select your SSH key
9. **Security group:** Allow SSH from your IP
10. Launch!

#### **Option B: AWS CLI (Faster)**

See detailed instructions in `docs/AWS_SPOT_TRAINING_GUIDE.md`

---

### **Phase 4: Setup p4d Instance**

**Location:** SSH into p4d instance  
**Time:** 15 minutes  
**Cost:** ~$3

```bash
# 1. SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# 2. Clone repo
cd /home/ubuntu
git clone https://github.com/yourusername/ragcun.git
cd ragcun

# 3. Copy your .env file (HuggingFace token)
# From your local machine:
scp -i your-key.pem .env ubuntu@<instance-ip>:/home/ubuntu/ragcun/

# 4. Run setup script
./scripts/setup_p4d_instance.sh

# Expected: "✅ Setup Complete!" with 8 GPUs detected
```

---

### **Phase 5: Final Pre-Flight on p4d**

**Location:** p4d instance  
**Time:** 5 minutes  
**Cost:** ~$1

```bash
# Run final checks on p4d hardware
./scripts/test_training_setup.sh

# Expected: "✅ All tests passed! Ready for training."
```

**This verifies:**
- 8× A100 GPUs working
- Large batch sizes fit in memory
- S3 backups configured
- Everything ready for 18-hour run

---

### **Phase 6: Download MS MARCO (if not done locally)**

**Location:** p4d instance  
**Time:** 30-40 minutes  
**Cost:** ~$5

```bash
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# Fast download on AWS network
```

---

### **Phase 7: Start Training! 🚀**

**Location:** p4d instance  
**Time:** 18 hours  
**Cost:** ~$189

```bash
# Start tmux session (so you can disconnect)
tmux new -s training

# Start parallel training (all 3 experiments at once!)
./scripts/train_parallel_p4d.sh

# Detach from tmux: Ctrl+B, then D
# Reattach anytime: tmux attach -t training
```

**What happens:**
- GPU 0: Baseline (no isotropy) - 18 hours
- GPU 1: With isotropy (YOUR METHOD) - 18 hours
- GPU 2: Frozen base (efficiency) - 18 hours
- All run in parallel!

**Monitor progress:**
```bash
# Watch GPUs
watch -n 1 nvidia-smi

# Check logs
tail -f logs/*.log

# Reattach to training session
tmux attach -t training
```

---

### **Phase 8: Evaluate Models**

**Location:** p4d instance  
**Time:** 2 hours  
**Cost:** ~$21

```bash
# After training completes (18 hours later)
./scripts/evaluate_all_beir.sh

# Evaluates all 3 models on BEIR (18 datasets)
```

**Results saved to:**
- `results/beir_baseline.json`
- `results/beir_with_isotropy.json`
- `results/beir_frozen.json`

---

### **Phase 9: Download Results & Terminate**

**Location:** p4d instance, then local  
**Time:** 10 minutes  
**Cost:** ~$2

```bash
# On p4d: Sync to S3
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
aws s3 sync results/ $S3_BACKUP_BUCKET/results/
aws s3 sync logs/ $S3_BACKUP_BUCKET/logs/

# On local machine: Download from S3
aws s3 sync s3://your-bucket/ragcun-training/ ./trained-models/

# Terminate p4d instance
aws ec2 terminate-instances --instance-ids <instance-id>
```

---

### **Phase 10: Celebrate! 🎉**

You now have:
- ✅ 3 trained models
- ✅ Full BEIR evaluation results
- ✅ Publication-ready data
- ✅ ~49% NDCG@10 on BEIR

**Write your paper!**

---

## 📊 Cost Breakdown

| Phase | Time | Cost @ $10.50/hr |
|-------|------|------------------|
| Testing (local) | 2 hrs | $0 |
| Setup p4d | 0.5 hrs | $5 |
| Download MS MARCO | 0.5 hrs | $5 |
| Training | 18 hrs | $189 |
| Evaluation | 2 hrs | $21 |
| Download & cleanup | 0.5 hrs | $5 |
| **Total** | **21 hours** | **$225** |

**vs 15 days on local T4** = 360 hours saved!

---

## ⚠️ Critical Reminders

### **Before Launching p4d:**
- [ ] ✅ `./scripts/run_preflight_tests.sh` passes ALL tests
- [ ] Have HuggingFace token ready
- [ ] AWS budget alert set ($250)
- [ ] S3 bucket created
- [ ] SSH key working

### **During p4d Training:**
- Use `tmux` so you can disconnect safely
- Monitor first 30 minutes closely
- S3 syncs automatically every 30 min
- Spot interruption risk < 5% (rare)

### **If Interrupted:**
- Don't panic! Checkpoints saved every epoch
- Re-launch p4d, resume from checkpoint:
  ```bash
  python scripts/train.py [same args] --resume checkpoints/[model]/checkpoint_epoch_2.pt
  ```

---

## 🆘 Troubleshooting

### **Pre-flight tests fail:**
```bash
# Common fixes:
pip install -r requirements.txt  # Missing dependencies
python -m pip install --upgrade pip  # Old pip version
```

### **p4d won't launch:**
- Check GPU quota: `aws service-quotas get-service-quota --service-code ec2 --quota-code L-417A185B`
- Try different region: us-east-1, us-west-2
- Increase max spot price: $15-20/hour

### **Training crashes:**
- Reduce batch size: `--batch_size 32` instead of 64
- Check logs: `tail -100 logs/*.log`
- Test locally first: `./scripts/run_preflight_tests.sh`

### **Out of money:**
- Set budget alerts in AWS Console
- Monitor costs: AWS Cost Explorer
- Expected: $220-230 total

---

## 📚 Documentation

- **This file** - Quick start guide
- **`PRE_FLIGHT_TEST_PLAN.md`** - Detailed testing strategy
- **`P4D_TRAINING_PLAN.md`** - Complete p4d plan with costs
- **`docs/RECOMMENDED_TRAINING_PATH.md`** - Training strategy explained
- **`docs/AWS_SPOT_TRAINING_GUIDE.md`** - AWS technical details

---

## 🚀 Quick Commands Reference

```bash
# 1. Test locally (NOW)
./scripts/run_preflight_tests.sh

# 2. Download MS MARCO (optional now)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 3. Launch p4d (AWS Console or CLI)

# 4. Setup p4d
ssh -i key.pem ubuntu@<ip>
cd /home/ubuntu && git clone [repo] && cd ragcun
./scripts/setup_p4d_instance.sh

# 5. Test on p4d
./scripts/test_training_setup.sh

# 6. Train (in tmux!)
tmux new -s training
./scripts/train_parallel_p4d.sh

# 7. Evaluate
./scripts/evaluate_all_beir.sh

# 8. Download and terminate
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
aws ec2 terminate-instances --instance-ids <id>
```

---

## ✅ Current Status

**You are here:** Phase 1 (Local Testing)

**Next step:** Run `./scripts/run_preflight_tests.sh`

**After that:** Launch p4d.24xlarge

**ETA to results:** ~24 hours from now

---

**Ready? Start with Phase 1:**

```bash
cd /home/ubuntu/ragcun
./scripts/run_preflight_tests.sh
```

**Good luck! 🚀**

