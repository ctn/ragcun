# Complete AWS GPU Setup for Cursor Development

I'll walk you through setting up a remote AWS GPU instance and connecting Cursor to it.

## Phase 1: Launch AWS EC2 GPU Instance

### Step 1: Choose Instance Type

```bash
# Common GPU instances:
# g4dn.xlarge   - $0.526/hr  - 1x T4 GPU (16GB)   - Good for training
# g4dn.2xlarge  - $0.752/hr  - 1x T4 GPU (16GB)   - More CPU/RAM
# p3.2xlarge    - $3.06/hr   - 1x V100 (16GB)     - Faster training
# g5.xlarge     - $1.006/hr  - 1x A10G (24GB)     - Best price/performance

# Recommendation for LeJEPA: g4dn.xlarge (cheapest, sufficient for 5K samples)
```

### Step 2: Launch Instance (AWS Console)

1. **Go to EC2 Dashboard** → Click "Launch Instance"

2. **Name**: `lejepa-gpu-dev`

3. **AMI (Operating System)**:
   - Search: "Deep Learning AMI GPU PyTorch"
   - Select: **Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)**
   - ✅ Pre-installed: CUDA, PyTorch, drivers

4. **Instance Type**:
   - Select: `g4dn.xlarge` (1x T4, 4 vCPU, 16GB RAM)

5. **Key Pair**:
   - Click "Create new key pair"
   - Name: `lejepa-gpu-key`
   - Type: RSA
   - Format: `.pem`
   - **Download** `lejepa-gpu-key.pem` → Save to `~/.ssh/`

6. **Network Settings**:
   - ✅ Allow SSH traffic from: My IP
   - ✅ Allow HTTPS traffic
   - (Can add Jupyter port 8888 later if needed)

7. **Storage**:
   - 100 GB gp3 (default 30GB is too small)

8. **Launch Instance** 🚀

### Step 3: Configure SSH Key

```bash
# Move key to SSH directory
mv ~/Downloads/lejepa-gpu-key.pem ~/.ssh/

# Set correct permissions (IMPORTANT!)
chmod 600 ~/.ssh/lejepa-gpu-key.pem

# Get instance public IP from AWS Console
# Look for "Public IPv4 address" (e.g., 54.123.45.67)
```

## Phase 2: Connect and Setup

### Step 4: Test SSH Connection

```bash
# Replace with your instance IP
export GPU_IP="54.123.45.67"  # Your actual IP from AWS Console

# Connect (Ubuntu is the default user for Ubuntu AMIs)
ssh -i ~/.ssh/lejepa-gpu-key.pem ubuntu@$GPU_IP

# You should see:
# Welcome to Ubuntu 20.04...
# ubuntu@ip-xxx-xxx-xxx-xxx:~$
```

### Step 5: Initial Setup on GPU Instance

```bash
# Now you're on the remote GPU machine

# 1. Verify GPU
nvidia-smi
# Should show: Tesla T4 GPU

# 2. Update system
sudo apt-get update
sudo apt-get upgrade -y

# 3. Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 4. Clone your project
git clone --recurse-submodules git@github.com:ctn/ragcun.git
cd ragcun

# If you get permission denied (public key):
# You need to add SSH key to GitHub from this machine
ssh-keygen -t ed25519 -C "aws-gpu-instance"
cat ~/.ssh/id_ed25519.pub
# Copy output and add to GitHub: Settings → SSH Keys → New SSH key

# Then retry clone
```

### Step 6: Install Dependencies

```bash
# Still on remote GPU machine
cd ~/ragcun

# Install project dependencies
uv pip install -e ".[all]"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Should show: PyTorch: 2.x.x, CUDA: True

python -c "from ragcun import GaussianRetriever; print('✅ ragcun works!')"
python -c "import lejepa; print('✅ LeJEPA works!')"
```

## Phase 3: Connect Cursor to AWS

### Step 7: Install Remote-SSH in Cursor

```bash
# On your local Mac
# 1. Open Cursor
# 2. Cmd+Shift+X (Extensions)
# 3. Search: "Remote - SSH"
# 4. Install: "Remote - SSH" by Microsoft
```

### Step 8: Configure SSH in Cursor

```bash
# On your local Mac, edit SSH config
code ~/.ssh/config  # or: cursor ~/.ssh/config

# Add this configuration:
```

```
Host lejepa-gpu
    HostName 54.123.45.67          # Your GPU instance IP
    User ubuntu
    IdentityFile ~/.ssh/lejepa-gpu-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### Step 9: Connect Cursor to Remote

```bash
# In Cursor:
# 1. Cmd+Shift+P (Command Palette)
# 2. Type: "Remote-SSH: Connect to Host"
# 3. Select: "lejepa-gpu"
# 4. New Cursor window opens - wait for connection
# 5. Click "Open Folder" → /home/ubuntu/ragcun
```

**You're now developing on AWS GPU with Cursor! 🎉**

## Phase 4: Develop and Train

### Step 10: Train Model on AWS GPU

In Cursor (connected to remote):

```python
# Open terminal in Cursor (it's on the remote machine!)
# Cmd+` or Terminal → New Terminal

# Navigate to notebooks
cd ~/ragcun/notebooks

# Convert notebook to script (easier for remote)
jupyter nbconvert --to script lejepa_training.ipynb

# Or run notebook directly
jupyter notebook --no-browser --port=8888
```

**To access Jupyter from local Mac:**

```bash
# On local Mac, in new terminal
ssh -N -L 8888:localhost:8888 -i ~/.ssh/lejepa-gpu-key.pem ubuntu@$GPU_IP

# Then open in browser: http://localhost:8888
```

### Step 11: Run Training

```python
# In Cursor terminal (on remote GPU)
cd ~/ragcun

# Create a training script
cat > train_remote.py << 'EOF'
import torch
from src.ragcun.model import GaussianEmbeddingGemma
from datasets import load_dataset
from lejepa.losses import sigreg
from tqdm.auto import tqdm

print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load model
model = GaussianEmbeddingGemma(output_dim=512, freeze_early_layers=True)
model = model.cuda()

# Load dataset
print("📚 Loading dataset...")
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:5000]")

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("🚀 Starting training...")
for epoch in range(3):
    for batch in tqdm(dataset.batch(32)):
        # Your training code here
        pass

# Save model
torch.save(model.state_dict(), 'data/embeddings/gaussian_embeddinggemma_final.pt')
print("✅ Training complete!")
EOF

# Run training
python train_remote.py
```

### Step 12: Download Trained Model

```bash
# On local Mac (new terminal)
# Download trained model from AWS to local
scp -i ~/.ssh/lejepa-gpu-key.pem \
    ubuntu@$GPU_IP:~/ragcun/data/embeddings/gaussian_embeddinggemma_final.pt \
    /Users/ctn/src/ctn/ragcun/data/embeddings/

# Now you can use it locally!
```

## Phase 5: Cost Management

### Important: Stop Instance When Not Using!

```bash
# AWS Console → EC2 → Instances
# Select instance → Instance State → Stop

# Or via AWS CLI:
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Costs:
# g4dn.xlarge: $0.526/hour when running
#              $0.10/month for 100GB storage when stopped
```

### Auto-shutdown Script (Save Money!)

```bash
# On remote GPU instance, create auto-shutdown
cat > ~/auto_shutdown.sh << 'EOF'
#!/bin/bash
# Shutdown if idle for 1 hour
IDLE_TIME=$(who -s | awk '{print $1}' | wc -l)
if [ $IDLE_TIME -eq 0 ]; then
    sudo shutdown -h now
fi
EOF

chmod +x ~/auto_shutdown.sh

# Add to crontab (runs every hour)
(crontab -l 2>/dev/null; echo "0 * * * * /home/ubuntu/auto_shutdown.sh") | crontab -
```

## Quick Reference Card

```bash
# Connect via SSH
ssh lejepa-gpu

# Connect via Cursor
# Cmd+Shift+P → "Remote-SSH: Connect to Host" → lejepa-gpu

# Start instance (AWS Console or CLI)
aws ec2 start-instances --instance-ids i-xxx

# Stop instance (IMPORTANT!)
aws ec2 stop-instances --instance-ids i-xxx

# Download files from remote
scp -i ~/.ssh/lejepa-gpu-key.pem ubuntu@$GPU_IP:/path/to/file /local/path

# Upload files to remote
scp -i ~/.ssh/lejepa-gpu-key.pem /local/file ubuntu@$GPU_IP:/remote/path
```

## Troubleshooting

### Connection refused
```bash
# Check instance is running in AWS Console
# Verify security group allows SSH (port 22) from your IP
```

### Permission denied (publickey)
```bash
# Check key permissions
chmod 600 ~/.ssh/lejepa-gpu-key.pem

# Verify you're using correct user (ubuntu for Ubuntu AMI)
```

### CUDA out of memory
```bash
# Reduce batch_size in training script
# Or upgrade to larger instance (g4dn.2xlarge)
```

### Can't access Jupyter
```bash
# Forward port correctly
ssh -N -L 8888:localhost:8888 lejepa-gpu

# Check Jupyter is running on remote
jupyter notebook list
```

## Next Steps

1. **Launch instance** → Get IP address
2. **Configure SSH** → Add to `~/.ssh/config`
3. **Connect Cursor** → Remote-SSH extension
4. **Setup project** → Clone ragcun, install dependencies
5. **Train model** → Run training script
6. **Download model** → SCP to local
7. **STOP INSTANCE** → Save money! 💰

## Alternative: Google Colab vs AWS

| Feature | Google Colab | AWS GPU |
|---------|-------------|---------|
| **Cost** | Free (limited) | $0.526/hr (g4dn.xlarge) |
| **GPU** | T4 (free tier) | T4, V100, A10G (choice) |
| **Time Limit** | 12 hours | Unlimited |
| **Cursor Integration** | No (browser only) | Yes (Remote-SSH) |
| **Storage** | Temporary | Persistent (EBS) |
| **Setup Complexity** | Easy | Moderate |

**Recommendation:**
- **Use Colab** for quick experiments (already set up in `notebooks/lejepa_training.ipynb`)
- **Use AWS** for longer training runs and Cursor development workflow
