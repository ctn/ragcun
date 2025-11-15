#!/bin/bash
# Setup script for p4d.24xlarge instance
# Run immediately after SSH into the instance

set -e

echo "============================================"
echo "Setting up p4d.24xlarge for RAGCUN Training"
echo "============================================"
echo ""

# Update system
echo "1. Updating system..."
sudo apt-get update -qq

# Install essential tools
echo "2. Installing essential tools..."
sudo apt-get install -y -qq tmux htop ncdu jq

# Verify GPUs
echo ""
echo "3. Verifying GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo ""
echo "✅ Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -ne 8 ]; then
    echo "⚠️  Warning: Expected 8 GPUs for p4d.24xlarge, found $NUM_GPUS"
fi

# Clone repo
echo ""
echo "4. Setting up repository..."
cd /home/ubuntu

if [ ! -d "ragcun" ]; then
    echo "Cloning repository..."
    git clone https://github.com/yourusername/ragcun.git
    cd ragcun
else
    echo "Repository exists, updating..."
    cd ragcun
    git pull
fi

# Install Python dependencies
echo ""
echo "5. Installing Python dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt -q

# Install optimizations for A100
echo ""
echo "6. Installing A100 optimizations..."
pip install triton -q  # Better performance on A100

# Setup environment
echo ""
echo "7. Setting up environment..."
if [ -f .env.production ]; then
    cp .env.production .env
    echo "✅ Environment variables loaded from .env.production"
elif [ -f .env ]; then
    echo "✅ Using existing .env"
else
    echo "⚠️  No .env file found!"
    echo "Create .env with:"
    echo "  HF_TOKEN=your_huggingface_token"
    echo "  WANDB_API_KEY=your_wandb_key (optional)"
fi

# Verify installation
echo ""
echo "8. Verifying installation..."
python -c "
import torch
import transformers
import sentence_transformers
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ CUDA version: {torch.version.cuda}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Sentence-transformers: {sentence_transformers.__version__}')
print(f'')
print(f'GPUs detected: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'         Memory: {props.total_memory / 1024**3:.1f} GB')
"

# Create directories
mkdir -p data/processed data/raw checkpoints results logs paper

# Setup S3 sync
echo ""
echo "9. Setting up S3 backup..."
S3_BUCKET_NAME=${S3_BACKUP_BUCKET:-"ragcun-training-$(date +%Y%m%d)"}

if [[ ! $S3_BUCKET_NAME =~ ^s3:// ]]; then
    S3_BUCKET_NAME="s3://$S3_BUCKET_NAME"
fi

echo "S3 bucket: $S3_BUCKET_NAME"
echo "export S3_BACKUP_BUCKET=$S3_BUCKET_NAME" >> ~/.bashrc

# Test S3 access
BUCKET_ONLY=$(echo $S3_BUCKET_NAME | sed 's|s3://||')
aws s3 ls s3://$BUCKET_ONLY 2>/dev/null || aws s3 mb s3://$BUCKET_ONLY

echo "✅ S3 bucket ready"

# Get instance metadata
echo ""
echo "10. Instance information..."
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)
INSTANCE_TYPE=$(ec2-metadata --instance-type | cut -d' ' -f2)
AZ=$(ec2-metadata --availability-zone | cut -d' ' -f2)

echo "Instance ID: $INSTANCE_ID"
echo "Instance type: $INSTANCE_TYPE"
echo "Availability zone: $AZ"

# Save instance info
cat > instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Instance type: $INSTANCE_TYPE
Availability zone: $AZ
GPUs: $NUM_GPUS
S3 Bucket: $S3_BUCKET_NAME
Setup completed: $(date)
EOF

echo ""
echo "============================================"
echo "✅ Setup Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  - Instance: $INSTANCE_TYPE"
echo "  - GPUs: $NUM_GPUS × A100 (40GB each)"
echo "  - PyTorch: Installed with CUDA"
echo "  - Repository: /home/ubuntu/ragcun"
echo "  - S3 Backup: $S3_BUCKET_NAME"
echo ""
echo "Estimated costs (spot pricing):"
echo "  - Training (~18 hrs): ~\$189"
echo "  - Evaluation (~2 hrs): ~\$21"
echo "  - Total: ~\$220"
echo ""
echo "Next steps:"
echo ""
echo "1. Download MS MARCO (30-40 min on p4d):"
echo "   python scripts/download_msmarco.py --output_dir data/processed/msmarco"
echo ""
echo "2. Test training setup (5 min):"
echo "   ./scripts/test_training_setup.sh"
echo ""
echo "3. Start parallel training (18 hours):"
echo "   tmux new -s training"
echo "   ./scripts/train_parallel_p4d.sh"
echo "   # Ctrl+B, D to detach"
echo ""
echo "4. Evaluate all models (2 hours):"
echo "   ./scripts/evaluate_all_beir.sh"
echo ""
echo "5. Download results and terminate:"
echo "   aws s3 sync checkpoints/ $S3_BUCKET_NAME/checkpoints/"
echo "   aws s3 sync results/ $S3_BUCKET_NAME/results/"
echo "   aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""

