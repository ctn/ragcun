# AWS Spot Instance Training Guide

**Goal:** Fast parallel training using powerful AWS spot instances with snapshot protection

---

## 🎯 Recommended Instance Configuration

### **Best Value: `p3.8xlarge`** ⭐

**Specs:**
- **4× Tesla V100 GPUs** (16GB each)
- 32 vCPUs
- 244 GB RAM
- 2 TB NVMe SSD

**Cost:**
- On-demand: $12.24/hour
- Spot: **$3.67-4.90/hour** (70% cheaper!)

**Training time:**
- All 3 experiments in parallel: **1.5 days**
- Total cost: **~$150-180**

**Why this one:**
- ✅ 4 GPUs = run all 3 experiments + 1 spare
- ✅ V100s are 3x faster than T4
- ✅ Great spot availability
- ✅ Best cost/performance ratio

---

### **Premium Option: `p4d.24xlarge`** 🔥

**Specs:**
- **8× Tesla A100 GPUs** (40GB each)
- 96 vCPUs
- 1152 GB RAM
- 8 TB NVMe SSD

**Cost:**
- On-demand: $32.77/hour
- Spot: **$9.83-13.11/hour** (70% cheaper!)

**Training time:**
- All 3 experiments in parallel: **<1 day**
- Total cost: **~$250-300**

**Why this one:**
- ✅ 8× A100s = 8x faster than T4, 2x faster than V100
- ✅ Massive RAM for large batch sizes
- ✅ Can run experiments + evaluation simultaneously
- ✅ Future-proof for larger models

---

### **Budget Option: `p3.2xlarge`**

**Specs:**
- **1× Tesla V100** (16GB)
- 8 vCPUs
- 61 GB RAM

**Cost:**
- Spot: **$0.92-1.23/hour**

**Training time:**
- Sequential: **~4.5 days**
- Total cost: **~$100-130**

**Why this one:**
- ✅ Cheapest multi-day option
- ✅ Still 3x faster than T4
- ❌ No parallelization (sequential training)

---

## 📋 Complete Setup Guide

### **Step 1: Launch Spot Instance**

#### Option A: AWS Console (Easy)

1. Go to EC2 → Launch Instance
2. **AMI:** Deep Learning AMI (Ubuntu 20.04) - `ami-0c2b0d3fb02824d92`
3. **Instance type:** `p3.8xlarge`
4. **Request Spot instances:** ✅ Check this box
5. **Storage:** 200 GB gp3 (or use instance store)
6. **Security group:** Allow SSH (port 22) from your IP
7. **Key pair:** Create/select your SSH key
8. Launch!

#### Option B: AWS CLI (Fast)

```bash
# Save this as launch_spot_instance.sh

#!/bin/bash

# Configuration
INSTANCE_TYPE="p3.8xlarge"
AMI_ID="ami-0c2b0d3fb02824d92"  # Deep Learning AMI
KEY_NAME="your-key-pair"
SECURITY_GROUP="your-sg-id"
MAX_PRICE="5.00"  # Maximum spot price ($/hour)

# Launch spot instance
aws ec2 request-spot-instances \
    --spot-price "$MAX_PRICE" \
    --instance-count 1 \
    --type "persistent" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {
                \"VolumeSize\": 200,
                \"VolumeType\": \"gp3\",
                \"DeleteOnTermination\": false
            }
        }],
        \"IamInstanceProfile\": {
            \"Name\": \"EC2-S3-Access\"
        }
    }"

echo "Spot instance requested!"
echo "Check status: aws ec2 describe-spot-instance-requests"
```

---

### **Step 2: Connect and Setup**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Verify GPUs
nvidia-smi
# Should show 4x V100 GPUs

# Clone your repo
cd /home/ubuntu
git clone https://github.com/yourusername/ragcun.git
cd ragcun

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cat > .env << EOF
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key  # Optional: for monitoring
EOF

# Verify installation
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
# Should print: GPUs available: 4
```

---

### **Step 3: Parallel Training Strategy** 🚀

With 4 GPUs on `p3.8xlarge`, run all 3 experiments in parallel:

```bash
# Create parallel training script
cat > scripts/train_parallel_aws.sh << 'EOF'
#!/bin/bash
set -e

echo "============================================"
echo "Parallel Training on AWS (4× V100)"
echo "============================================"
echo ""
echo "Starting all 3 experiments in parallel..."
echo "GPU 0: Baseline (no isotropy)"
echo "GPU 1: With isotropy (YOUR METHOD)"
echo "GPU 2: Frozen base (efficiency)"
echo "GPU 3: Reserved for evaluation/monitoring"
echo ""

# Create log directory
mkdir -p logs checkpoints

# Common arguments
COMMON_ARGS="--train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 3 \
    --batch_size 32 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --mixed_precision \
    --save_interval 1 \
    --log_interval 50"

# Experiment 1: Baseline (GPU 0)
echo "Starting Experiment 1 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python scripts/train/isotropic.py \
    $COMMON_ARGS \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --output_dir checkpoints/baseline_no_isotropy \
    2>&1 | tee logs/baseline_no_isotropy.log &
PID1=$!

# Experiment 2: With isotropy (GPU 1)
echo "Starting Experiment 2 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python scripts/train/isotropic.py \
    $COMMON_ARGS \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/with_isotropy \
    2>&1 | tee logs/with_isotropy.log &
PID2=$!

# Experiment 3: Frozen base (GPU 2)
echo "Starting Experiment 3 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python scripts/train/isotropic.py \
    $COMMON_ARGS \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/frozen_efficient \
    2>&1 | tee logs/frozen_efficient.log &
PID3=$!

echo ""
echo "All experiments started!"
echo "PIDs: $PID1, $PID2, $PID3"
echo ""
echo "Monitor progress:"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Training logs: tail -f logs/*.log"
echo ""

# Wait for all to complete
wait $PID1
echo "✅ Experiment 1 complete"

wait $PID2
echo "✅ Experiment 2 complete"

wait $PID3
echo "✅ Experiment 3 complete"

echo ""
echo "============================================"
echo "✅ All Training Complete!"
echo "============================================"
echo "Elapsed: $(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m"
EOF

chmod +x scripts/train_parallel_aws.sh
```

---

### **Step 4: Snapshot Strategy** 💾

**Critical: Spot instances can be interrupted!**

#### Auto-Checkpoint During Training

Already handled in your training script:
```bash
--save_interval 1  # Saves checkpoint every epoch
```

#### Periodic EBS Snapshots

```bash
# Create snapshot script
cat > scripts/create_snapshot.sh << 'EOF'
#!/bin/bash
# Create EBS snapshot every 2 hours while training

VOLUME_ID=$(aws ec2 describe-volumes \
    --filters "Name=attachment.instance-id,Values=$(ec2-metadata --instance-id | cut -d' ' -f2)" \
    --query "Volumes[0].VolumeId" --output text)

while true; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "Creating snapshot at $TIMESTAMP..."
    
    aws ec2 create-snapshot \
        --volume-id $VOLUME_ID \
        --description "ragcun-training-$TIMESTAMP" \
        --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=ragcun-training-$TIMESTAMP}]"
    
    # Sleep 2 hours
    sleep 7200
done
EOF

chmod +x scripts/create_snapshot.sh

# Run in background
nohup ./scripts/create_snapshot.sh > logs/snapshots.log 2>&1 &
```

#### Sync to S3 (Recommended)

```bash
# Sync checkpoints to S3 continuously
cat > scripts/sync_to_s3.sh << 'EOF'
#!/bin/bash
# Sync checkpoints to S3 every 30 minutes

S3_BUCKET="s3://your-bucket/ragcun-training"

while true; do
    echo "Syncing to S3 at $(date)..."
    
    # Sync checkpoints
    aws s3 sync checkpoints/ $S3_BUCKET/checkpoints/ \
        --exclude "*.tmp" --exclude "*.swp"
    
    # Sync logs
    aws s3 sync logs/ $S3_BUCKET/logs/
    
    # Sleep 30 minutes
    sleep 1800
done
EOF

chmod +x scripts/sync_to_s3.sh

# Run in background
nohup ./scripts/sync_to_s3.sh > logs/s3_sync.log 2>&1 &
```

---

### **Step 5: Download Data and Start Training**

```bash
# 1. Download MS MARCO (30-40 min with fast instance)
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 2. Start background snapshot sync
nohup ./scripts/sync_to_s3.sh > logs/s3_sync.log 2>&1 &

# 3. Start parallel training
./scripts/train_parallel_aws.sh

# Training will take ~36 hours on p3.8xlarge (4× V100)
```

---

### **Step 6: Monitor Training**

#### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Or use gpustat (prettier)
pip install gpustat
watch -n 1 gpustat -cpu
```

#### Training Progress
```bash
# Monitor all experiments
tmux new-session -d -s monitoring '
    tmux split-window -v;
    tmux split-window -v;
    tmux select-pane -t 0;
    tmux send-keys "tail -f logs/baseline_no_isotropy.log" C-m;
    tmux select-pane -t 1;
    tmux send-keys "tail -f logs/with_isotropy.log" C-m;
    tmux select-pane -t 2;
    tmux send-keys "tail -f logs/frozen_efficient.log" C-m;
'

# Attach to view
tmux attach -t monitoring
```

#### WandB Monitoring (Optional but Recommended)
```bash
# Add to your training script
pip install wandb

# In scripts/train/isotropic.py, add:
import wandb
wandb.init(project="ragcun-training", name=f"{experiment_name}")
wandb.config.update(args)
# Log metrics during training
wandb.log({"loss": loss, "epoch": epoch})
```

---

### **Step 7: Handle Spot Interruption**

Spot instances give 2-minute warning before termination.

```bash
# Create interruption handler
cat > scripts/spot_interruption_handler.sh << 'EOF'
#!/bin/bash
# Monitor spot interruption notice

while true; do
    # Check for interruption notice
    if curl -s http://169.254.169.254/latest/meta-data/spot/instance-action | grep -q terminate; then
        echo "⚠️  SPOT INTERRUPTION DETECTED! Saving state..."
        
        # Kill training processes gracefully
        pkill -SIGTERM -f "train.py"
        
        # Wait for checkpoints to save
        sleep 60
        
        # Final sync to S3
        aws s3 sync checkpoints/ s3://your-bucket/ragcun-training/checkpoints/
        aws s3 sync logs/ s3://your-bucket/ragcun-training/logs/
        
        echo "✅ State saved. Instance will terminate soon."
        break
    fi
    
    sleep 5
done
EOF

chmod +x scripts/spot_interruption_handler.sh

# Run in background
nohup ./scripts/spot_interruption_handler.sh > logs/interruption.log 2>&1 &
```

#### Resume After Interruption

```bash
# Check which experiments completed
ls checkpoints/*/best_model.pt

# Resume incomplete experiments
CUDA_VISIBLE_DEVICES=0 python scripts/train/isotropic.py \
    [same args as before] \
    --resume checkpoints/baseline_no_isotropy/checkpoint_epoch_2.pt \
    --output_dir checkpoints/baseline_no_isotropy
```

---

## 📊 Timeline & Cost Breakdown

### **p3.8xlarge (4× V100) - RECOMMENDED**

| Phase | Time | Cost (Spot @ $4/hr) |
|-------|------|---------------------|
| Setup & download | 1 hour | $4 |
| Parallel training | 36 hours | $144 |
| Evaluation | 4 hours | $16 |
| **Total** | **41 hours** | **$164** |

**Key advantage:** All 3 experiments done in 1.5 days!

---

### **p4d.24xlarge (8× A100) - PREMIUM**

| Phase | Time | Cost (Spot @ $10/hr) |
|-------|------|----------------------|
| Setup & download | 0.5 hour | $5 |
| Parallel training | 18 hours | $180 |
| Evaluation | 2 hours | $20 |
| **Total** | **20.5 hours** | **$205** |

**Key advantage:** Done in under 1 day! Can run evaluation in parallel too.

---

### **Comparison with Local T4**

| Setup | Time | Cost |
|-------|------|------|
| Local T4 (sequential) | 15 days | Free (electricity) |
| p3.8xlarge (parallel) | 1.5 days | $164 |
| p4d.24xlarge (parallel) | <1 day | $205 |

**Verdict:** If you value time, AWS spot is absolutely worth it!

---

## 🎯 Optimizations for Fast Training

### 1. **Larger Batch Sizes**
```bash
# With 4× V100 (16GB each), increase batch size:
--batch_size 32  # Instead of 16
# Or even 64 with gradient checkpointing
```

### 2. **Optimized Data Loading**
```bash
# Preload to instance store (NVMe SSD)
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme1n1 /mnt/nvme
cp -r data/ /mnt/nvme/data/

# Update training to use /mnt/nvme/data
```

### 3. **Compiled Model (PyTorch 2.0+)**
```bash
# Add to train.py
model = torch.compile(model)  # 20-30% speedup
```

### 4. **Mixed Precision**
```bash
# Already in your script
--mixed_precision  # 2x speedup, no accuracy loss
```

---

## ✅ Pre-Launch Checklist

Before launching spot instance:

- [ ] AWS account with GPU quota (request if needed)
- [ ] S3 bucket created for backups
- [ ] IAM role for EC2 with S3 access
- [ ] SSH key pair created
- [ ] Security group configured (port 22 from your IP)
- [ ] HuggingFace token ready
- [ ] Budget alert set ($200-300)

---

## 🚀 Quick Launch Commands

```bash
# 1. Launch instance (AWS CLI)
aws ec2 request-spot-instances \
    --spot-price "5.00" \
    --instance-count 1 \
    --type "persistent" \
    --launch-specification file://spot-config.json

# 2. SSH in
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Setup
cd /home/ubuntu
git clone https://github.com/yourusername/ragcun.git
cd ragcun
pip install -r requirements.txt

# 4. Download data
python scripts/download_msmarco.py --output_dir data/processed/msmarco

# 5. Start S3 sync
nohup ./scripts/sync_to_s3.sh > logs/s3_sync.log 2>&1 &

# 6. Start interruption handler
nohup ./scripts/spot_interruption_handler.sh > logs/interruption.log 2>&1 &

# 7. Train!
./scripts/train_parallel_aws.sh

# 8. Evaluate
./scripts/evaluate_all_beir.sh
```

**Done in 1.5-2 days!** 🎉

---

## 💡 Pro Tips

### 1. **Use Spot Fleet (Even Better)**
Request multiple instance types for better availability:
```bash
aws ec2 create-spot-fleet-request --spot-fleet-request-config file://fleet-config.json
```

### 2. **Persistent Spot Requests**
If interrupted, AWS automatically launches replacement:
```bash
--type "persistent"  # Already in our config
```

### 3. **Use Screen/Tmux**
Detach from SSH without stopping training:
```bash
tmux new -s training
./scripts/train_parallel_aws.sh
# Ctrl+B, then D to detach
# Later: tmux attach -t training
```

### 4. **Monitor Costs**
```bash
# Check current spot price
aws ec2 describe-spot-price-history \
    --instance-types p3.8xlarge \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --product-descriptions "Linux/UNIX"
```

### 5. **Snapshot the Entire Instance (Not Just EBS)**
```bash
# After training completes, create AMI
aws ec2 create-image \
    --instance-id i-1234567890abcdef0 \
    --name "ragcun-trained-$(date +%Y%m%d)" \
    --description "RAGCUN with trained models"
```

Then you can launch from this AMI anytime with all models ready!

---

## 🔄 After Training: Terminate & Save

```bash
# 1. Final sync to S3
aws s3 sync checkpoints/ s3://your-bucket/ragcun-training/checkpoints/
aws s3 sync results/ s3://your-bucket/ragcun-training/results/
aws s3 sync logs/ s3://your-bucket/ragcun-training/logs/

# 2. Create AMI snapshot (optional)
aws ec2 create-image \
    --instance-id $(ec2-metadata --instance-id | cut -d' ' -f2) \
    --name "ragcun-complete-$(date +%Y%m%d)"

# 3. Download results locally
aws s3 sync s3://your-bucket/ragcun-training/ ./ragcun-training-results/

# 4. Terminate instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

---

## 📦 Deliverables After Training

You'll have in S3:
```
s3://your-bucket/ragcun-training/
├── checkpoints/
│   ├── baseline_no_isotropy/best_model.pt
│   ├── with_isotropy/best_model.pt
│   └── frozen_efficient/best_model.pt
├── results/
│   ├── beir_baseline.json
│   ├── beir_with_isotropy.json
│   └── beir_frozen.json
└── logs/
    └── [all training logs]
```

**Ready for paper!** 📊

---

## Summary

**Recommended:** `p3.8xlarge` with 4× V100 GPUs
- **Cost:** ~$164 (spot)
- **Time:** 1.5 days
- **Strategy:** All 3 experiments in parallel
- **Backup:** S3 sync every 30 min + spot interruption handler

**Commands:**
```bash
# Launch → Setup → Train → Evaluate → Snapshot → Terminate
# Total: <2 days wall-clock time
```

**You'll have publication-ready results by Monday!** 🚀

