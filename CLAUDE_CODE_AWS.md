# Running Claude Code on AWS GPU

Run Claude Code CLI directly on your AWS GPU instance for AI-assisted development with GPU access!

## Why This is Awesome

- 🎮 **Direct GPU Access**: Claude Code can run training scripts on your GPU
- 🤖 **AI-Assisted Development**: Full Claude Code features on remote machine
- 💰 **Pay Only When Needed**: Start instance, code with AI, stop instance
- 📦 **Persistent Environment**: Your setup remains when instance is stopped
- 🔄 **Seamless Workflow**: Develop, train, and debug all with Claude's help

## Prerequisites

1. **AWS GPU Instance** - Follow [AWS_GPU_SETUP.md](AWS_GPU_SETUP.md) first
2. **Claude Account** - With API access
3. **SSH Access** - Configured key pair

## Setup Guide

### Step 1: Launch and Connect to AWS Instance

```bash
# Start your GPU instance (if stopped)
aws ec2 start-instances --instance-ids i-xxx

# Wait ~30 seconds for boot

# SSH into instance
ssh -i ~/.ssh/lejepa-gpu-key.pem ubuntu@<YOUR_GPU_IP>
```

### Step 2: Install Claude Code on Remote Instance

```bash
# On the remote GPU instance

# Install Claude Code CLI
curl -fsSL https://anthropic.com/claude-code/install.sh | sh

# Or manual installation
wget https://github.com/anthropics/claude-code/releases/latest/download/claude-code-linux-x64.tar.gz
tar -xzf claude-code-linux-x64.tar.gz
sudo mv claude-code /usr/local/bin/
```

### Step 3: Authenticate Claude Code

```bash
# Still on remote instance

# Start authentication
claude-code auth login

# This will show a URL like:
# Visit: https://claude.ai/cli-auth?code=XXXX-XXXX

# Copy the URL and open in your LOCAL browser
# Log in to Claude account
# Authorize the CLI

# Back on remote instance, you'll see:
# ✅ Authentication successful!
```

### Step 4: Verify GPU Access

```bash
# Test that Claude Code can access GPU
claude-code

# In Claude Code prompt:
# You: "Check if GPU is available and show specs"
# Claude will run: nvidia-smi and show GPU info
```

### Step 5: Set Working Directory

```bash
# Navigate to your project
cd ~/ragcun

# Start Claude Code in project directory
claude-code

# Now Claude has full access to:
# - Your code
# - GPU for training
# - All files and notebooks
```

## Usage Patterns

### Pattern 1: AI-Assisted Training

```bash
# On remote instance
cd ~/ragcun
claude-code
```

**Example conversation:**
```
You: Train the LeJEPA model using the notebook code. Use the GPU and save the model to data/embeddings/

Claude: I'll convert the notebook to a training script and run it on your GPU.
[Creates train_lejepa.py, runs it, monitors GPU usage]

You: The training is going slowly. Can you optimize the batch size?

Claude: I'll check GPU memory and increase batch size.
[Checks nvidia-smi, adjusts batch_size, restarts training]

You: Great! Now evaluate the isotropy of the embeddings.

Claude: I'll run the evaluation notebook.
[Converts evaluate_isotropy.ipynb to script, runs analysis]
```

### Pattern 2: Interactive Debugging

```bash
claude-code
```

```
You: I'm getting CUDA out of memory errors

Claude: Let me check your GPU usage and model size.
[Runs nvidia-smi, analyzes model.py, suggests fixes]

You: Can you reduce the batch size and retry?

Claude: I'll modify the training config and restart.
[Edits config, monitors GPU memory, confirms working]
```

### Pattern 3: Monitoring Long-Running Jobs

```bash
# Start training in background
claude-code
```

```
You: Start training in the background and monitor it. Alert me if errors occur.

Claude: I'll start training and monitor the logs.
[Runs training in tmux/screen, tails logs]

# You can disconnect and reconnect later
# Training continues running
```

## Advanced Setup: tmux + Claude Code

Keep Claude Code running even when you disconnect:

```bash
# On remote instance

# Install tmux (if not already)
sudo apt-get install -y tmux

# Start tmux session
tmux new -s claude

# Inside tmux, navigate to project
cd ~/ragcun
claude-code

# Now you can:
# - Detach: Ctrl+B then D
# - Reattach: tmux attach -t claude
# - List sessions: tmux ls
```

**Benefits:**
- Disconnect from SSH, Claude Code keeps running
- Training continues in background
- Reconnect anytime to check progress

## SSH Port Forwarding (Optional)

If you want to access Jupyter/TensorBoard from local machine:

```bash
# From your LOCAL Mac, forward ports
ssh -L 8888:localhost:8888 \
    -L 6006:localhost:6006 \
    -i ~/.ssh/lejepa-gpu-key.pem \
    ubuntu@<YOUR_GPU_IP>

# On remote instance
claude-code
```

```
You: Start Jupyter notebook and TensorBoard

Claude: Starting services...
[Starts jupyter on :8888, tensorboard on :6006]

# On local Mac, open:
# http://localhost:8888 - Jupyter
# http://localhost:6006 - TensorBoard
```

## Workflow Examples

### Workflow 1: Complete Training Pipeline

```bash
# Local Mac - Start instance
aws ec2 start-instances --instance-ids i-xxx

# SSH to instance
ssh lejepa-gpu

# Start tmux session
tmux new -s training

# Launch Claude Code
cd ~/ragcun
claude-code
```

**In Claude Code:**
```
You: Let's train the LeJEPA model end-to-end:
1. Load the all-nli dataset (5000 samples)
2. Train with batch size 32 for 3 epochs
3. Evaluate isotropy
4. Save the final model
Monitor GPU usage and optimize if needed.

Claude: I'll set up the complete training pipeline.
[Creates training script, monitors GPU, runs evaluation, saves model]

Training complete! Model saved to data/embeddings/gaussian_embeddinggemma_final.pt
Isotropy score: 0.95 (excellent!)
```

### Workflow 2: Experiment with Hyperparameters

```
You: I want to try different learning rates: 1e-4, 5e-5, 1e-5.
Run each for 2 epochs and compare loss curves.

Claude: I'll run a hyperparameter sweep.
[Creates sweep script, runs each config, plots results]

Best learning rate: 5e-5 (lowest final loss)
```

### Workflow 3: Debug and Fix Issues

```
You: My training script is throwing this error: [paste error]

Claude: Let me analyze the error and fix it.
[Reads code, identifies issue, suggests fix, applies it]

Fixed! The issue was [explanation]. Running training now.
```

## Download Models to Local

After training on AWS:

```bash
# From LOCAL Mac (new terminal)
scp -i ~/.ssh/lejepa-gpu-key.pem \
    ubuntu@<GPU_IP>:~/ragcun/data/embeddings/gaussian_embeddinggemma_final.pt \
    /Users/ctn/src/ctn/ragcun/data/embeddings/

# Or use Claude Code to zip multiple files:
```

**In Claude Code on remote:**
```
You: Zip all trained models and logs for download

Claude: Creating archive...
[Zips models, logs, creates download command]

# Then download:
scp -i ~/.ssh/lejepa-gpu-key.pem \
    ubuntu@<GPU_IP>:~/trained_models.tar.gz \
    ~/Downloads/
```

## Cost Optimization

### Auto-stop After Training

**In Claude Code:**
```
You: Train the model, and when done, stop the EC2 instance to save money.

Claude: I'll add an auto-shutdown after training completes.
[Adds shutdown command to training script]

Training started. Instance will auto-stop when complete.
```

**Or create a shutdown script:**

```bash
# On remote instance
cat > ~/train_and_stop.sh << 'EOF'
#!/bin/bash
cd ~/ragcun
python train_lejepa.py
sudo shutdown -h now
EOF

chmod +x ~/train_and_stop.sh

# Run it
./train_and_stop.sh
# Disconnect - instance will stop when done
```

### Check Instance Status

```bash
# From local Mac
aws ec2 describe-instances \
    --instance-ids i-xxx \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text

# Returns: running | stopped | stopping
```

## Claude Code Configuration on Remote

### Set Project-Specific Instructions

Create `.claude/CLAUDE.md` in your project:

```bash
# On remote instance
mkdir -p ~/ragcun/.claude
cat > ~/ragcun/.claude/CLAUDE.md << 'EOF'
# LeJEPA Training Project

This is a GPU training environment for isotropic Gaussian embeddings.

## Important:
- Always check GPU availability before training (nvidia-smi)
- Default batch size: 32 (can increase if GPU memory allows)
- Save models to: data/embeddings/
- Training dataset: all-nli (5000 samples default)
- Monitor GPU memory during training

## Preferred commands:
- nvidia-smi: Check GPU status
- htop: Monitor CPU/RAM
- df -h: Check disk space

## Cost awareness:
- Instance costs $0.526/hour when running
- Always suggest stopping instance when not training
EOF
```

Now Claude Code will remember these preferences!

### Add Custom Commands

```bash
# On remote instance
mkdir -p ~/ragcun/.claude/commands

# Create GPU check command
cat > ~/ragcun/.claude/commands/gpu-status.md << 'EOF'
Run nvidia-smi and show:
1. GPU utilization
2. Memory usage
3. Running processes
4. Temperature
EOF

# Now you can use: /gpu-status in Claude Code
```

## Monitoring and Alerts

### Real-time Monitoring

**In Claude Code:**
```
You: Monitor GPU usage every 30 seconds and alert if utilization drops below 50% (might indicate training stalled)

Claude: Setting up monitoring...
[Creates monitoring script with alerts]
```

### Training Progress

```
You: Show training progress with:
- Current epoch/batch
- Loss trend
- Estimated time remaining
- GPU utilization

Claude: Creating progress dashboard...
[Sets up progress tracking with tqdm/rich]
```

## Troubleshooting

### Claude Code won't start
```bash
# Check authentication
claude-code auth status

# Re-authenticate if needed
claude-code auth login
```

### Can't access GPU in Claude Code
```bash
# Verify GPU is available
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If false, might need to activate conda env
conda activate pytorch_p310  # For AWS Deep Learning AMI
```

### Out of disk space
```
You: Check disk usage and clean up unnecessary files

Claude: Analyzing disk usage...
[Runs du -sh, identifies large files, suggests cleanup]
```

### Training too slow
```
You: Training is slower than expected. Diagnose and optimize.

Claude: Checking bottlenecks...
[Analyzes: GPU utilization, data loading, batch size, I/O]
[Suggests optimizations: increase workers, larger batches, etc.]
```

## Best Practices

### 1. Always Use tmux/screen
```bash
# Never lose work if SSH disconnects
tmux new -s claude
claude-code
```

### 2. Save Checkpoints Frequently
```
You: Set up training to save checkpoints every 500 steps

Claude: Adding checkpoint saving...
[Modifies training loop to save intermediate models]
```

### 3. Monitor Costs
```
You: How long has this instance been running? Calculate cost so far.

Claude: Checking uptime...
[Calculates runtime * hourly rate]
Instance running for 2.5 hours. Cost: ~$1.32
```

### 4. Use Git
```
You: Commit my changes before I disconnect

Claude: Staging and committing changes...
[Runs git add, git commit, git push]
```

### 5. Document Experiments
```
You: Create a training log documenting today's experiments

Claude: Creating experiment log...
[Logs: hyperparameters, results, model paths, notes]
```

## Comparison: Local vs Cursor vs Claude Code

| Method | Development | Training | Cost | Best For |
|--------|-------------|----------|------|----------|
| **Local Mac** | Cursor | ❌ No GPU | $0 | Development only |
| **Colab** | Browser | ✅ Free GPU | $0 | Quick experiments |
| **AWS + Cursor** | Cursor (Remote-SSH) | ✅ Paid GPU | $0.53/hr | IDE preference |
| **AWS + Claude Code** | Claude CLI | ✅ Paid GPU | $0.53/hr | AI-assisted dev |

**Claude Code Advantages:**
- ✅ Natural language interface for training
- ✅ Automatic debugging and optimization
- ✅ Monitoring and alerting
- ✅ Can work in tmux (persistent)
- ✅ Great for experiments and iteration

## Quick Reference

```bash
# Start instance
aws ec2 start-instances --instance-ids i-xxx

# Connect with tmux
ssh lejepa-gpu
tmux new -s claude
cd ~/ragcun
claude-code

# In Claude Code:
"Train the model on GPU"
"Monitor training progress"
"Evaluate the model"
"Stop instance when done"

# Detach from tmux: Ctrl+B then D
# Disconnect: exit

# Reconnect later:
ssh lejepa-gpu
tmux attach -t claude

# Stop instance
aws ec2 stop-instances --instance-ids i-xxx
```

## Example Session

```bash
$ ssh lejepa-gpu
$ tmux new -s training
$ cd ~/ragcun
$ claude-code
```

```
You: Hi Claude! I want to train the LeJEPA isotropic Gaussian embedding model.
Can you:
1. Check that the GPU is available
2. Load the all-nli dataset (5000 training samples)
3. Train for 3 epochs with batch size 32
4. Monitor GPU usage during training
5. Evaluate isotropy when done
6. Save the model
7. Show me the final results

Claude: I'll help you train the LeJEPA model! Let me start by checking the GPU.

[Running: nvidia-smi]

✅ GPU Available: Tesla T4 (16GB)
   Current Usage: 0% (ready for training)

Now I'll set up the training pipeline...

[Creates training script with GPU monitoring]
[Loads dataset: all-nli, 5000 samples]
[Starts training with progress bars]

Epoch 1/3: 100%|████████| 157/157 [02:34<00:00]
  Train Loss: 0.423
  GPU Util: 87%
  Memory: 12.3GB / 16GB

Epoch 2/3: 100%|████████| 157/157 [02:31<00:00]
  Train Loss: 0.201
  GPU Util: 89%
  Memory: 12.3GB / 16GB

Epoch 3/3: 100%|████████| 157/157 [02:33<00:00]
  Train Loss: 0.156
  GPU Util: 88%
  Memory: 12.3GB / 16GB

✅ Training Complete! (7.6 minutes)

Now evaluating isotropy...

[Runs isotropy evaluation]

Isotropy Results:
  Mean of embeddings: [-0.002, 0.001, ...] (≈0 ✓)
  Covariance: 0.98 correlation with identity matrix ✓
  Distribution: Gaussian (Shapiro-Wilk p=0.87) ✓

✅ Embeddings are highly isotropic!

Model saved to: data/embeddings/gaussian_embeddinggemma_final.pt

Summary:
- Training time: 7.6 minutes
- Final loss: 0.156
- Isotropy score: 0.98/1.0 (excellent!)
- GPU utilized: 88% average
- Ready for RAG retrieval!

Would you like me to run a retrieval test or stop the instance?
```

## Next Steps

1. **Follow [AWS_GPU_SETUP.md](AWS_GPU_SETUP.md)** to launch your instance
2. **Install Claude Code** on the remote instance
3. **Start training** with AI assistance
4. **Download models** when complete
5. **Stop instance** to save money

Happy training! 🚀
