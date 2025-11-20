# p4d.24xlarge Training Plan (8× A100 GPUs)

**Instance:** AWS p4d.24xlarge  
**Goal:** Complete all training + evaluation in <24 hours

---

## 💰 Cost Estimate

### **Spot Pricing:**
- **Hourly rate:** $9.83-13.11/hr (region dependent)
- **Average:** ~$10.50/hr
- **vs On-demand:** $32.77/hr (68% savings!)

### **Timeline Breakdown:**

| Phase | Time on A100 | Cost @ $10.50/hr |
|-------|--------------|------------------|
| Setup + data download | 0.5 hours | $5 |
| Parallel training (3 experiments) | 18 hours | $189 |
| BEIR evaluation (all models) | 2 hours | $21 |
| **Total** | **20.5 hours** | **$215** |

**Completion:** < 1 day (single session!)

---

## ⚡ Speed Comparison

| Instance | GPUs | Training Time | Total Time | Cost (Spot) |
|----------|------|---------------|------------|-------------|
| Local T4 | 1× T4 | 15 days | 15 days | Free |
| p3.8xlarge | 4× V100 | 36 hours | 1.5 days | $164 |
| **p4d.24xlarge** | **8× A100** | **18 hours** | **<1 day** | **$215** |

**Key advantages:**
- ✅ **Done in single work day** (launch morning, done by next morning)
- ✅ **A100s = 2x faster than V100s** (higher throughput, larger batch sizes)
- ✅ **8 GPUs** = run all 3 experiments + extras simultaneously
- ✅ **Huge memory** (40GB per GPU, 1152GB RAM)

---

## 🚀 Optimized Configuration

### **Increased Batch Sizes for A100:**

```bash
# A100 has 40GB vs V100's 16GB
# Can use MUCH larger batches

COMMON_ARGS="--train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 3 \
    --batch_size 64 \           # 2x larger than V100!
    --gradient_accumulation_steps 1 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --mixed_precision \
    --compile \                 # PyTorch 2.0 compile for extra speed
    --save_interval 1 \
    --log_interval 50"
```

**Expected speedup:**
- Larger batch: ~1.5x faster
- A100 architecture: ~2x faster than V100
- Total: ~3x faster than V100 = **~18 hours** vs 36 hours

---

## 📋 Pre-Training Checklist

### **1. AWS Account Setup**

- [ ] GPU quota check:
  ```bash
  aws service-quotas get-service-quota \
      --service-code ec2 \
      --quota-code L-417A185B  # p4d instances
  
  # If quota = 0, request increase (takes 24-48 hours)
  ```

- [ ] Budget alert:
  ```bash
  # Set $250 budget alert in AWS Console
  # Billing > Budgets > Create budget
  ```

- [ ] S3 bucket for backups:
  ```bash
  aws s3 mb s3://ragcun-training-$(date +%Y%m%d)
  ```

- [ ] IAM role with S3 access:
  ```bash
  # Create EC2 role with AmazonS3FullAccess policy
  ```

### **2. Local Environment**

- [ ] SSH key pair created:
  ```bash
  aws ec2 create-key-pair \
      --key-name ragcun-training \
      --query 'KeyMaterial' \
      --output text > ~/.ssh/ragcun-training.pem
  
  chmod 400 ~/.ssh/ragcun-training.pem
  ```

- [ ] Security group configured:
  ```bash
  # Allow SSH from your IP
  aws ec2 create-security-group \
      --group-name ragcun-sg \
      --description "RAGCUN training security group"
  
  aws ec2 authorize-security-group-ingress \
      --group-name ragcun-sg \
      --protocol tcp \
      --port 22 \
      --cidr $(curl -s ifconfig.me)/32
  ```

- [ ] HuggingFace token ready:
  ```bash
  echo "HF_TOKEN=hf_your_token_here" > .env.production
  ```

### **3. Code Preparation**

- [ ] All scripts executable:
  ```bash
  cd /home/ubuntu/ragcun
  chmod +x scripts/*.sh
  ```

- [ ] Dependencies list ready:
  ```bash
  # Ensure requirements.txt has all deps
  cat requirements.txt
  ```

- [ ] Test scripts locally (if possible):
  ```bash
  # Dry run to check imports
  python scripts/train/isotropic.py --help
  python scripts/eval/beir.py --help
  ```

---

## 🔧 Launch Configuration

### **Spot Instance Request (AWS CLI):**

Save as `launch_p4d_spot.sh`:

```bash
#!/bin/bash
# Launch p4d.24xlarge spot instance

set -e

# Configuration
INSTANCE_TYPE="p4d.24xlarge"
AMI_ID="ami-0c2b0d3fb02824d92"  # Deep Learning AMI (Ubuntu 20.04)
KEY_NAME="ragcun-training"
SECURITY_GROUP_NAME="ragcun-sg"
SUBNET_ID="subnet-xxxxx"  # Your subnet with p4d availability
MAX_SPOT_PRICE="15.00"  # Max willing to pay (spot usually $10-11)
S3_BUCKET="ragcun-training-$(date +%Y%m%d)"

echo "============================================"
echo "Launching p4d.24xlarge Spot Instance"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Instance type: $INSTANCE_TYPE"
echo "  Max price: \$$MAX_SPOT_PRICE/hour"
echo "  AMI: $AMI_ID"
echo ""

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups \
    --group-names $SECURITY_GROUP_NAME \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo "Security group: $SG_ID"
echo ""

# Create S3 bucket for backups
echo "Creating S3 bucket: $S3_BUCKET"
aws s3 mb s3://$S3_BUCKET 2>/dev/null || echo "Bucket exists"

# Create spot instance request
echo "Requesting spot instance..."
SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --spot-price "$MAX_SPOT_PRICE" \
    --instance-count 1 \
    --type "persistent" \
    --instance-interruption-behavior "stop" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$SG_ID\"],
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {
                \"VolumeSize\": 300,
                \"VolumeType\": \"gp3\",
                \"Iops\": 3000,
                \"Throughput\": 125,
                \"DeleteOnTermination\": false
            }
        }],
        \"UserData\": \"$(base64 -w 0 <<EOF
#!/bin/bash
# Auto-setup script
echo "S3_BUCKET=$S3_BUCKET" >> /home/ubuntu/.bashrc
echo "export S3_BACKUP_BUCKET=s3://$S3_BUCKET" >> /home/ubuntu/.bashrc
EOF
)\"
    }" \
    --output json)

REQUEST_ID=$(echo $SPOT_REQUEST | jq -r '.SpotInstanceRequests[0].SpotInstanceRequestId')

echo ""
echo "✅ Spot request created: $REQUEST_ID"
echo ""
echo "Waiting for fulfillment..."

# Wait for fulfillment
aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $REQUEST_ID

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids $REQUEST_ID \
    --query 'SpotInstanceRequests[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo ""
echo "Waiting for instance to be running..."

# Wait for running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "============================================"
echo "✅ Instance Ready!"
echo "============================================"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "S3 Bucket: s3://$S3_BUCKET"
echo ""
echo "Connect with:"
echo "  ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo ""
echo "Monitor spot price:"
echo "  aws ec2 describe-spot-price-history \\"
echo "    --instance-types $INSTANCE_TYPE \\"
echo "    --start-time \$(date -u +%Y-%m-%dT%H:%M:%S) \\"
echo "    --product-descriptions \"Linux/UNIX\""
echo ""

# Save instance info
cat > instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
S3 Bucket: s3://$S3_BUCKET
Launched: $(date)
EOF

echo "Instance info saved to: instance_info.txt"
```

Make executable:
```bash
chmod +x launch_p4d_spot.sh
```

---

## 🎯 Setup Script for Instance

Save as `scripts/setup_p4d_instance.sh`:

```bash
#!/bin/bash
# Run this immediately after SSH into p4d instance

set -e

echo "============================================"
echo "Setting up p4d.24xlarge for RAGCUN Training"
echo "============================================"
echo ""

# Update system
echo "1. Updating system..."
sudo apt-get update -qq

# Install essential tools
echo "2. Installing tools..."
sudo apt-get install -y -qq tmux htop ncdu jq

# Verify GPUs
echo ""
echo "3. Verifying GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "✅ Detected $NUM_GPUS GPUs"

# Clone repo
echo ""
echo "4. Cloning repository..."
cd /home/ubuntu
if [ ! -d "ragcun" ]; then
    git clone https://github.com/yourusername/ragcun.git
    cd ragcun
else
    cd ragcun
    git pull
fi

# Install Python dependencies
echo ""
echo "5. Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Additional optimizations for A100
pip install triton -q  # For better performance

# Setup environment
echo ""
echo "6. Setting up environment..."
if [ -f .env.production ]; then
    cp .env.production .env
    echo "✅ Environment variables loaded"
else
    echo "⚠️  .env.production not found. Create it with HF_TOKEN"
fi

# Verify installation
echo ""
echo "7. Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPUs detected: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Create directories
mkdir -p data/processed data/raw checkpoints results logs

# Setup S3 sync
echo ""
echo "8. Setting up S3 backup..."
S3_BUCKET=$(echo $S3_BACKUP_BUCKET | sed 's|s3://||')
if [ -z "$S3_BUCKET" ]; then
    S3_BUCKET="ragcun-training-$(date +%Y%m%d)"
    echo "export S3_BACKUP_BUCKET=s3://$S3_BUCKET" >> ~/.bashrc
fi

echo "S3 bucket: s3://$S3_BUCKET"

# Test S3 access
aws s3 ls s3://$S3_BUCKET 2>/dev/null || aws s3 mb s3://$S3_BUCKET

echo ""
echo "============================================"
echo "✅ Setup Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  - GPUs: $NUM_GPUS × A100 (40GB each)"
echo "  - PyTorch: Installed with CUDA"
echo "  - Repository: /home/ubuntu/ragcun"
echo "  - S3 Backup: s3://$S3_BUCKET"
echo ""
echo "Next steps:"
echo ""
echo "1. Download MS MARCO (30-40 min):"
echo "   python scripts/download_msmarco.py --output_dir data/processed/msmarco"
echo ""
echo "2. Start parallel training (18 hours):"
echo "   ./scripts/train_parallel_p4d.sh"
echo ""
echo "3. Evaluate all models (2 hours):"
echo "   ./scripts/evaluate_all_beir.sh"
echo ""
```

---

## 🚀 Optimized Training Script for p4d

Save as `scripts/train_parallel_p4d.sh`:

```bash
#!/bin/bash
# Optimized parallel training for p4d.24xlarge (8× A100)
# Uses larger batch sizes and all optimizations

set -e

echo "============================================"
echo "Parallel Training on p4d.24xlarge (8× A100)"
echo "============================================"
echo ""

# Verify 8 GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $NUM_GPUS -lt 8 ]; then
    echo "⚠️  Warning: Expected 8 GPUs, found $NUM_GPUS"
fi

echo "GPUs available: $NUM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check data
if [ ! -f "data/processed/msmarco/train.json" ]; then
    echo "❌ MS MARCO not found."
    echo "Download first: python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    exit 1
fi

mkdir -p logs checkpoints

# Start S3 sync in background
if [ -n "$S3_BACKUP_BUCKET" ]; then
    echo "Starting S3 backup sync..."
    nohup bash -c 'while true; do
        aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/ --quiet
        aws s3 sync logs/ $S3_BACKUP_BUCKET/logs/ --quiet
        sleep 1800  # Every 30 min
    done' > logs/s3_sync.log 2>&1 &
    echo "✅ S3 sync started (every 30 min)"
fi

# Record start
START_TIME=$(date +%s)
echo ""
echo "Started at: $(date)"
echo ""
echo "Starting all 3 experiments in parallel..."
echo "  GPU 0: Baseline (no isotropy)"
echo "  GPU 1: With isotropy (YOUR METHOD)"
echo "  GPU 2: Frozen base (efficiency)"
echo "  GPUs 3-7: Available"
echo ""

# Common arguments - OPTIMIZED FOR A100
COMMON_ARGS="--train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 3 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --mixed_precision \
    --save_interval 1 \
    --log_interval 50"

# Experiment 1: Baseline (GPU 0)
echo "Starting Experiment 1..."
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

sleep 5

# Experiment 2: With isotropy (GPU 1)
echo "Starting Experiment 2..."
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

sleep 5

# Experiment 3: Frozen base (GPU 2)
echo "Starting Experiment 3..."
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
echo "✅ All experiments running!"
echo "PIDs: $PID1, $PID2, $PID3"
echo ""
echo "Expected completion: ~18 hours"
echo ""
echo "Monitor:"
echo "  - nvidia-smi:  watch -n 1 nvidia-smi"
echo "  - Logs:        tail -f logs/*.log"
echo ""

# Wait for completion
wait $PID1 $PID2 $PID3

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "✅ All Training Complete!"
echo "============================================"
echo ""
echo "Wall-clock time: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m"
echo "Finished: $(date)"
echo ""
echo "Models ready for evaluation!"
echo ""
```

---

## ⏱️ Detailed Timeline

### **Complete p4d.24xlarge Session:**

```
00:00 - Launch spot instance (5 min)
00:05 - SSH in, run setup script (10 min)
00:15 - Download MS MARCO (30 min)
00:45 - Start parallel training
       ├─ GPU 0: Baseline
       ├─ GPU 1: With isotropy
       └─ GPU 2: Frozen base
       (All run for ~18 hours)
───────────────────────────────────
18:45 - All training complete
18:45 - Start BEIR evaluation (2 hours)
20:45 - All evaluation complete
20:45 - Sync to S3, download results
21:00 - Terminate instance
───────────────────────────────────
Total: 21 hours (single calendar day!)
Cost: ~$220 @ $10.50/hour
```

---

## 💾 Backup Strategy

### **Automatic S3 Sync:**
- Runs in background during training
- Syncs checkpoints + logs every 30 min
- Continues even if spot interrupted

### **Manual Backup:**
```bash
# Final sync before termination
aws s3 sync checkpoints/ $S3_BACKUP_BUCKET/checkpoints/
aws s3 sync results/ $S3_BACKUP_BUCKET/results/
aws s3 sync logs/ $S3_BACKUP_BUCKET/logs/
```

---

## 🎯 Success Metrics

After training, you should have:

```
checkpoints/
├── baseline_no_isotropy/
│   ├── best_model.pt (~450MB)
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── checkpoint_epoch_3.pt
├── with_isotropy/
│   └── [same structure]
└── frozen_efficient/
    └── [same structure]

results/
├── beir_baseline.json
├── beir_with_isotropy.json
└── beir_frozen.json

logs/
└── [all training logs]

Total size: ~5-6GB
```

---

## 🚨 Risk Mitigation

### **Spot Interruption:**
- Probability: <5% for p4d in most regions
- Protection: Checkpoints + S3 sync
- Recovery: Resume from last checkpoint (< 6 hours lost max)

### **Training Failure:**
- Monitor first 30 min closely
- Check GPU utilization (should be >80%)
- Verify batch size fits in memory
- Check logs for errors

### **Budget Overrun:**
- Set AWS budget alert at $250
- Training should complete in 18-20 hours
- If >24 hours, investigate (something wrong)

---

## ✅ Final Pre-Flight Checklist

Before `./launch_p4d_spot.sh`:

- [ ] AWS GPU quota verified (p4d instances)
- [ ] Budget alert set ($250)
- [ ] S3 bucket name chosen
- [ ] SSH key pair created and tested
- [ ] Security group allows SSH from your IP
- [ ] HuggingFace token in `.env.production`
- [ ] Local repo up-to-date with all scripts
- [ ] `requirements.txt` complete
- [ ] Time available to monitor first hour

---

## 🚀 Launch Commands

```bash
# On your local machine:

# 1. Launch instance
./launch_p4d_spot.sh

# Wait ~5 min for fulfillment, note the IP address

# 2. SSH in
ssh -i ~/.ssh/ragcun-training.pem ubuntu@<instance-ip>

# 3. Setup
cd /home/ubuntu
# Upload .env.production if needed
# scp -i ~/.ssh/ragcun-training.pem .env.production ubuntu@<ip>:/home/ubuntu/ragcun/

./ragcun/scripts/setup_p4d_instance.sh

# 4. Download data
python ragcun/scripts/download_msmarco.py --output_dir ragcun/data/processed/msmarco

# 5. Start training (use tmux!)
cd ragcun
tmux new -s training
./scripts/train_parallel_p4d.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training

# 6. Come back in 18-20 hours, evaluate
./scripts/evaluate_all_beir.sh

# 7. Download and terminate
aws s3 sync $S3_BACKUP_BUCKET ./local-results/
aws ec2 terminate-instances --instance-ids <instance-id>
```

---

## 📊 Expected Cost Breakdown

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| Instance time | 21 | $10.50 | $220.50 |
| EBS storage (300GB @ $0.08/GB) | - | - | $24/month (prorated) |
| S3 storage (6GB) | - | - | $0.14/month |
| Data transfer | - | - | ~$1 |
| **Total** | **21 hours** | - | **~$225** |

**Final estimate: $220-230 for complete training + evaluation**

---

**You're ready! This will get you publication-ready results in a single day.** 🚀

