# AWS Setup Guide for Fast Training

**Goal:** Train all 3 models in parallel on powerful AWS hardware  
**Recommended:** p4d.24xlarge (8× A100 GPUs)  
**Time:** 21 hours total  
**Cost:** ~$220 (spot pricing)

---

## 💰 Cost & Time Estimates

| Instance | GPUs | Training Time | Total Cost | Best For |
|----------|------|---------------|------------|----------|
| p3.8xlarge | 4× V100 | 36 hours | $164 | Good value |
| **p4d.24xlarge** | **8× A100** | **18 hours** | **$220** | **Fastest** ⭐ |

**vs Local T4:** 360 hours (15 days) saved!

---

## 📋 Prerequisites

### **1. AWS Account Setup**

- [ ] Active AWS account
- [ ] GPU quota verified (p4d instances)
  ```bash
  aws service-quotas get-service-quota \
      --service-code ec2 \
      --quota-code L-417A185B
  ```
  If quota = 0, request increase (takes 24-48 hours)

- [ ] Budget alert set ($250)
- [ ] S3 bucket created for backups
  ```bash
  aws s3 mb s3://ragcun-training-$(date +%Y%m%d)
  ```

### **2. Local Prerequisites**

- [ ] AWS CLI installed and configured
- [ ] SSH key pair created
  ```bash
  aws ec2 create-key-pair \
      --key-name ragcun-training \
      --query 'KeyMaterial' \
      --output text > ~/.ssh/ragcun-training.pem
  chmod 400 ~/.ssh/ragcun-training.pem
  ```

- [ ] HuggingFace token ready (in `.env` file)

---

## 🚀 Step-by-Step Launch

### **Step 1: Launch Spot Instance**

#### **Option A: AWS Console (Easier)**

1. Go to **EC2 → Launch Instance**
2. **Name:** `ragcun-training`
3. **AMI:** Deep Learning AMI (Ubuntu 20.04)  
   Search: `Deep Learning AMI GPU PyTorch`
4. **Instance type:** `p4d.24xlarge`
5. **✅ Check:** Request Spot instances
6. **Max price:** $15.00/hour (spot usually $10-11/hour)
7. **Storage:** 300 GB gp3
8. **Key pair:** Select your SSH key
9. **Security group:** Allow SSH (port 22) from your IP
10. **Launch!**

Wait ~5 minutes for fulfillment, note the Public IP.

#### **Option B: AWS CLI (Faster)**

Save configuration:
```json
// spot-config.json
{
  "ImageId": "ami-0c2b0d3fb02824d92",
  "InstanceType": "p4d.24xlarge",
  "KeyName": "ragcun-training",
  "SecurityGroupIds": ["sg-xxxxx"],
  "BlockDeviceMappings": [{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 300,
      "VolumeType": "gp3"
    }
  }]
}
```

Launch:
```bash
aws ec2 request-spot-instances \
    --spot-price "15.00" \
    --instance-count 1 \
    --type "persistent" \
    --launch-specification file://spot-config.json
```

---

### **Step 2: Connect and Setup**

```bash
# SSH into instance
ssh -i ~/.ssh/ragcun-training.pem ubuntu@<instance-ip>

# Clone repo
cd /home/ubuntu
git clone https://github.com/yourusername/ragcun.git
cd ragcun

# Copy .env file with HuggingFace token
# From local machine:
scp -i ~/.ssh/ragcun-training.pem .env ubuntu@<instance-ip>:/home/ubuntu/ragcun/

# Run automated setup
./scripts/setup_p4d_instance.sh
```

**This setup script:**
- Installs dependencies
- Verifies 8 GPUs
- Sets up S3 backup
- Prepares directories

**Expected output:**
```
✅ Setup Complete!
  - Instance: p4d.24xlarge
  - GPUs: 8 × A100 (40GB each)
  - S3 Backup: s3://ragcun-training-xxxxx
```

---

### **Step 3: Pre-Flight Test on p4d**

```bash
# Final verification on p4d hardware
./scripts/test_training_setup.sh
```

**This tests:**
- All 8 GPUs accessible
- Large batch sizes fit in memory (64 on A100)
- S3 backups configured
- Everything ready for 18-hour run

**Expected:** `✅ All tests passed! Ready for training.`

---

### **Step 4: Download MS MARCO**

```bash
# Fast download on AWS network (~30-40 min)
python scripts/download_msmarco.py --output_dir data/processed/msmarco
```

---

### **Step 5: Start Training (Use tmux!)**

```bash
# Create tmux session (so you can disconnect safely)
tmux new -s training

# Start parallel training - all 3 experiments at once!
./scripts/train_parallel_p4d.sh

# Detach from tmux: Ctrl+B, then D
# Reattach anytime: tmux attach -t training
```

**What runs:**
- **GPU 0:** Baseline (no isotropy) - 18 hours
- **GPU 1:** With isotropy (YOUR METHOD) - 18 hours
- **GPU 2:** Frozen base (efficiency) - 18 hours
- **GPUs 3-7:** Available for other tasks

**All complete in ~18 hours!**

---

### **Step 6: Monitor Training**

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f logs/*.log

# View all logs in tmux
tmux attach -t monitoring
```

**Monitor first 30 minutes closely:**
- GPU utilization should be >80%
- No out-of-memory errors
- Loss decreasing
- Checkpoints saving

---

### **Step 7: Evaluate Models**

```bash
# After training completes (~18 hours later)
./scripts/evaluate_all_beir.sh
```

**Time:** ~2 hours to evaluate all 3 models  
**Output:** `results/beir_*.json`

---

### **Step 8: Download Results & Terminate**

```bash
# Sync to S3 (automatic every 30 min, but final sync)
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
aws s3 sync results/ $S3_BACKUP_BUCKET/results/
aws s3 sync logs/ $S3_BACKUP_BUCKET/logs/

# Get instance ID
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)

# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Download locally (from your machine)
aws s3 sync s3://ragcun-training-xxxxx/ ./trained-models/
```

---

## 💾 Backup Strategy

### **Automatic S3 Sync**

Runs in background during training:
- Syncs checkpoints + logs every 30 minutes
- Continues even if spot interrupted
- Recoverable if training stops

### **Manual Backup**

```bash
# Anytime during training
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
```

---

## 🚨 Spot Interruption Handling

**Probability:** <5% for p4d (rare)

**If interrupted:**
1. Checkpoints saved every epoch (max 6 hours lost)
2. S3 has last sync (max 30 min lost)
3. Re-launch instance, resume from checkpoint:

```bash
python scripts/train.py [same args] \
    --resume checkpoints/with_isotropy/checkpoint_epoch_2.pt
```

---

## 📊 Cost Breakdown

| Phase | Time | Cost @ $10.50/hr |
|-------|------|------------------|
| Setup + download | 1 hour | $10 |
| Training (parallel) | 18 hours | $189 |
| Evaluation | 2 hours | $21 |
| **Total** | **21 hours** | **~$220** |

**Additional costs:**
- EBS storage (300GB): ~$1/day
- S3 storage (6GB): ~$0.14/month
- Data transfer: ~$1

**Total estimate:** $220-230

---

## 🎯 Optimization Tips

### **Larger Batch Sizes on A100**

```bash
# A100 has 40GB vs V100's 16GB
--batch_size 64  # 2x larger than V100
```

### **Monitor Costs**

```bash
# Check current spot price
aws ec2 describe-spot-price-history \
    --instance-types p4d.24xlarge \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%S)
```

### **Set Budget Alert**

AWS Console → Billing → Budgets → Create budget
- Amount: $250
- Alert at: 80% ($200)

---

## 🐛 Troubleshooting

### **Spot Request Not Fulfilled**

```bash
# Check availability in different regions
aws ec2 describe-spot-price-history \
    --instance-types p4d.24xlarge \
    --region us-east-1

# Try different region or increase max price
```

### **Out of Memory**

```bash
# Reduce batch size
--batch_size 32  # Instead of 64
```

### **Training Slow**

```bash
# Check GPU utilization (should be >80%)
nvidia-smi

# If low, increase batch size
--batch_size 96  # If memory allows
```

### **Can't Connect via SSH**

- Check security group allows port 22 from your IP
- Verify key pair permissions: `chmod 400 ~/.ssh/ragcun-training.pem`
- Check instance is running: `aws ec2 describe-instances`

---

## 📈 Expected Timeline

```
00:00 - Launch spot instance (5 min)
00:05 - SSH in, run setup script (10 min)
00:15 - Download MS MARCO (30 min)
00:45 - Start parallel training
        ├─ GPU 0: Baseline
        ├─ GPU 1: With isotropy
        └─ GPU 2: Frozen base
───────────────────────────────────
18:45 - All training complete
18:45 - Start BEIR evaluation (2 hours)
20:45 - All evaluation complete
20:45 - Sync to S3, download results
21:00 - Terminate instance
───────────────────────────────────
Total: 21 hours (single calendar day!)
Cost: ~$220
```

---

## ✅ Launch Checklist

Before launching p4d:

- [ ] Pre-flight tests passed locally
- [ ] AWS GPU quota verified
- [ ] S3 bucket created
- [ ] SSH key working
- [ ] Budget alert set
- [ ] HuggingFace token ready
- [ ] Time available to monitor first hour

---

## 🚀 Quick Launch Commands

```bash
# 1. Launch instance (AWS Console recommended for first time)

# 2. SSH and setup
ssh -i ~/.ssh/ragcun-training.pem ubuntu@<ip>
cd /home/ubuntu && git clone [repo] && cd ragcun
./scripts/setup_p4d_instance.sh

# 3. Test
./scripts/test_training_setup.sh

# 4. Download data
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 5. Train (in tmux!)
tmux new -s training
./scripts/train_parallel_p4d.sh

# 6. Evaluate
./scripts/evaluate_all_beir.sh

# 7. Download and terminate
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
aws ec2 terminate-instances --instance-ids <id>
```

---

**You'll have publication-ready results by tomorrow!** 🚀

