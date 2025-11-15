# Pre-Flight Test Plan

**Critical: Test everything on cheap/local hardware BEFORE expensive p4d run**

---

## 🎯 Why Test First?

**Without testing:**
- Launch $220 p4d instance
- Discover script error 2 hours in
- Lost: $21 + wasted time 😰

**With testing:**
- Test on local T4 or cheap t2.micro
- Fix all issues
- Confident $220 run succeeds ✅

**Time investment:** 2-3 hours testing  
**Savings:** Potentially $220 + 20 hours

---

## 📋 Three-Stage Testing Strategy

### **Stage 1: Local Testing (No GPU needed) - 30 min**
Test all non-GPU code on your current machine

### **Stage 2: GPU Testing (Your T4 or cheap g4dn) - 1 hour**
Test model loading and training loop with small data

### **Stage 3: Mini Training Run (Your T4 or g4dn.xlarge) - 30 min**
Run 1 epoch with tiny dataset to verify end-to-end

---

## Stage 1: Local Testing (No GPU) ✅

**What to test:**
- All scripts are executable
- Imports work
- Data download works
- Data format is correct

**Run on your current machine:**

```bash
cd /home/ubuntu/ragcun

# Test 1: Check all scripts exist and are executable
echo "Test 1: Script availability..."
for script in \
    scripts/download_msmarco.py \
    scripts/train.py \
    scripts/evaluate_beir.py \
    scripts/train_parallel_p4d.sh \
    scripts/setup_p4d_instance.sh \
    scripts/test_training_setup.sh; do
    
    if [ -f "$script" ]; then
        echo "  ✅ $script exists"
    else
        echo "  ❌ $script MISSING!"
    fi
done

# Test 2: Python imports (no GPU)
echo ""
echo "Test 2: Python imports..."
python << 'EOF'
import sys
try:
    import torch
    import transformers
    import sentence_transformers
    import datasets
    import numpy as np
    import json
    from pathlib import Path
    print("✅ All imports work")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Install: pip install -r requirements.txt")
    sys.exit(1)
EOF

# Test 3: Download small MS MARCO subset (10K examples)
echo ""
echo "Test 3: Data download (10K subset)..."
python scripts/download_msmarco.py \
    --output_dir data/processed/msmarco_test \
    --max_train_samples 10000

# Verify data format
python << 'EOF'
import json

with open('data/processed/msmarco_test/train.json') as f:
    data = json.load(f)

assert isinstance(data, list), "Data must be list"
assert len(data) > 0, "Data is empty"
assert 'query' in data[0], "Missing 'query' key"
assert 'positive' in data[0], "Missing 'positive' key"
assert 'negative' in data[0], "Missing 'negative' key"

print(f"✅ Data format correct ({len(data):,} examples)")
print(f"   Example: {data[0]['query'][:50]}...")
EOF

# Test 4: Verify model.py exists and is importable
echo ""
echo "Test 4: Custom model import..."
python << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from ragcun.model import GaussianEmbeddingGemma
    print("✅ GaussianEmbeddingGemma imports successfully")
except ImportError as e:
    print(f"❌ Model import failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "============================================"
echo "✅ Stage 1 Complete: All non-GPU tests pass"
echo "============================================"
```

**Expected time:** 30 minutes (mostly data download)  
**Cost:** $0

---

## Stage 2: GPU Testing (Your T4) ⚡

**What to test:**
- GPU is accessible
- Model loads on GPU
- Forward pass works
- Training loop starts
- Checkpointing works

**Run on your T4:**

```bash
cd /home/ubuntu/ragcun

# Test 1: GPU availability
echo "Test 1: GPU check..."
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Test 2: Load base model on GPU
echo ""
echo "Test 2: Base model loading..."
python << 'EOF'
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = model.cuda()

# Test encoding
embeddings = model.encode(["Test"], convert_to_tensor=True)
print(f"✅ Base model works on GPU")
print(f"   Output shape: {embeddings.shape}")
print(f"   On GPU: {embeddings.is_cuda}")
EOF

# Test 3: Custom model on GPU
echo ""
echo "Test 3: Custom model (GaussianEmbeddingGemma)..."
python << 'EOF'
from ragcun.model import GaussianEmbeddingGemma
import torch

# Test frozen base
model = GaussianEmbeddingGemma(
    output_dim=512,
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=True
)
model = model.cuda()

# Test encoding
embeddings = model.encode(["Test sentence", "Another test"])
print(f"✅ Custom model works")
print(f"   Output shape: {embeddings.shape}")
print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test unfrozen base
model2 = GaussianEmbeddingGemma(
    output_dim=512,
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=False
)
trainable = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"✅ Unfrozen model: {trainable:,} trainable params")
EOF

# Test 4: Training script dry-run (2 steps only)
echo ""
echo "Test 4: Training script dry-run..."
python scripts/train.py \
    --train_data data/processed/msmarco_test/train.json \
    --val_data data/processed/msmarco_test/train.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 1 \
    --batch_size 4 \
    --output_dim 512 \
    --warmup_steps 2 \
    --output_dir /tmp/test_run \
    --max_steps 2 \
    --log_interval 1

# Check output
if [ -d "/tmp/test_run" ]; then
    echo "✅ Training script creates output directory"
    ls -la /tmp/test_run
    rm -rf /tmp/test_run
else
    echo "❌ Training script failed to create output"
    exit 1
fi

echo ""
echo "============================================"
echo "✅ Stage 2 Complete: All GPU tests pass"
echo "============================================"
```

**Expected time:** 15-20 minutes  
**Cost:** $0 (your existing T4)

---

## Stage 3: Mini Training Run (Your T4 or g4dn.xlarge) 🏃

**What to test:**
- Complete training epoch works
- Checkpoints save correctly
- Validation runs
- All 3 experiment configurations work
- Scripts can run in parallel

**Option A: Your T4 (Free)**

```bash
cd /home/ubuntu/ragcun

echo "============================================"
echo "Mini Training Run (1 epoch, 1000 examples)"
echo "============================================"
echo ""

# Create tiny subset
python << 'EOF'
import json

# Load full 10K dataset
with open('data/processed/msmarco_test/train.json') as f:
    data = json.load(f)

# Take only 1000 examples
tiny_data = data[:1000]

# Save as tiny dataset
with open('data/processed/msmarco_tiny/train.json', 'w') as f:
    json.dump(tiny_data, f)

# Use same for val
with open('data/processed/msmarco_tiny/val.json', 'w') as f:
    json.dump(tiny_data[:100], f)

print(f"✅ Created tiny dataset: {len(tiny_data)} train, 100 val")
EOF

# Test Experiment 1: Baseline (no isotropy)
echo ""
echo "Test Experiment 1: Baseline..."
python scripts/train.py \
    --train_data data/processed/msmarco_tiny/train.json \
    --val_data data/processed/msmarco_tiny/val.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --epochs 1 \
    --batch_size 8 \
    --output_dim 512 \
    --warmup_steps 10 \
    --mixed_precision \
    --output_dir /tmp/test_baseline \
    --save_interval 1

# Verify checkpoint exists
if [ -f "/tmp/test_baseline/checkpoint_epoch_1.pt" ]; then
    echo "✅ Experiment 1: Checkpoint saved"
else
    echo "❌ Experiment 1: Checkpoint missing!"
    exit 1
fi

# Test Experiment 2: With isotropy
echo ""
echo "Test Experiment 2: With isotropy..."
python scripts/train.py \
    --train_data data/processed/msmarco_tiny/train.json \
    --val_data data/processed/msmarco_tiny/val.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 1 \
    --batch_size 8 \
    --output_dim 512 \
    --warmup_steps 10 \
    --mixed_precision \
    --output_dir /tmp/test_with_isotropy \
    --save_interval 1

# Verify checkpoint
if [ -f "/tmp/test_with_isotropy/checkpoint_epoch_1.pt" ]; then
    echo "✅ Experiment 2: Checkpoint saved"
else
    echo "❌ Experiment 2: Checkpoint missing!"
    exit 1
fi

# Test Experiment 3: Frozen base
echo ""
echo "Test Experiment 3: Frozen base..."
python scripts/train.py \
    --train_data data/processed/msmarco_tiny/train.json \
    --val_data data/processed/msmarco_tiny/val.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 1 \
    --batch_size 8 \
    --output_dim 512 \
    --warmup_steps 10 \
    --mixed_precision \
    --output_dir /tmp/test_frozen \
    --save_interval 1

# Verify checkpoint
if [ -f "/tmp/test_frozen/checkpoint_epoch_1.pt" ]; then
    echo "✅ Experiment 3: Checkpoint saved"
else
    echo "❌ Experiment 3: Checkpoint missing!"
    exit 1
fi

# Test model loading
echo ""
echo "Test: Loading saved checkpoint..."
python << 'EOF'
from ragcun.model import GaussianEmbeddingGemma
import torch

# Load checkpoint
checkpoint_path = '/tmp/test_with_isotropy/checkpoint_epoch_1.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"✅ Checkpoint loads successfully")
print(f"   Keys: {list(checkpoint.keys())}")

# Try loading into model
model = GaussianEmbeddingGemma.from_pretrained(checkpoint_path)
print(f"✅ Model loads from checkpoint")

# Test encoding
embeddings = model.encode(["Test"])
print(f"✅ Loaded model works: {embeddings.shape}")
EOF

# Cleanup
rm -rf /tmp/test_baseline /tmp/test_with_isotropy /tmp/test_frozen

echo ""
echo "============================================"
echo "✅ Stage 3 Complete: Full pipeline works!"
echo "============================================"
```

**Expected time:** 20-30 minutes  
**Cost:** $0 (your T4)

---

## Option B: Cheap AWS Test (g4dn.xlarge)

If you want to test on AWS infrastructure first:

```bash
# Launch cheap GPU instance
# g4dn.xlarge: 1× T4, $0.15/hour spot

aws ec2 request-spot-instances \
    --spot-price "0.20" \
    --instance-count 1 \
    --launch-specification file://g4dn-test-config.json

# Run Stage 2 and 3 tests
# Total cost: ~$0.50 (3 hours max)
```

---

## ✅ Pre-Flight Checklist

Before launching expensive p4d.24xlarge:

### **Code Ready:**
- [ ] All scripts exist and are executable
- [ ] All imports work
- [ ] Model loads correctly (frozen and unfrozen)
- [ ] Training loop starts and runs
- [ ] Checkpoints save and load
- [ ] All 3 experiment configs tested

### **Data Ready:**
- [ ] MS MARCO downloads without errors
- [ ] Data format is correct (query/positive/negative)
- [ ] Small subset (10K) tested successfully

### **Infrastructure Ready:**
- [ ] AWS GPU quota confirmed (p4d)
- [ ] S3 bucket created and accessible
- [ ] SSH keys working
- [ ] Budget alerts set

### **Scripts Tested:**
- [ ] `scripts/download_msmarco.py` ✅
- [ ] `scripts/train.py` with all 3 configs ✅
- [ ] `scripts/setup_p4d_instance.sh` reviewed
- [ ] `scripts/train_parallel_p4d.sh` reviewed
- [ ] `scripts/test_training_setup.sh` works

---

## 🚀 Complete Test Sequence

**Run this complete sequence before p4d:**

```bash
cd /home/ubuntu/ragcun

# Stage 1: Local tests (30 min)
echo "Stage 1: Local tests..."
# [Run all Stage 1 commands above]

# Stage 2: GPU tests (20 min)
echo "Stage 2: GPU tests..."
# [Run all Stage 2 commands above]

# Stage 3: Mini training (30 min)
echo "Stage 3: Mini training..."
# [Run all Stage 3 commands above]

echo ""
echo "============================================"
echo "✅ ALL PRE-FLIGHT TESTS PASSED"
echo "============================================"
echo ""
echo "You are ready to launch p4d.24xlarge!"
echo ""
echo "Next steps:"
echo "1. Review P4D_TRAINING_PLAN.md"
echo "2. Launch p4d instance"
echo "3. Run setup_p4d_instance.sh"
echo "4. Run test_training_setup.sh on p4d"
echo "5. Start full training"
echo ""
```

---

## 💰 Cost Comparison

| Approach | Time | Cost | Risk |
|----------|------|------|------|
| **No testing → Launch p4d** | 0 hours prep | Potential $220 lost | HIGH ❌ |
| **Test locally → Launch p4d** | 2 hours prep | $0 + $220 p4d | LOW ✅ |
| **Test on g4dn → Launch p4d** | 3 hours prep | $0.50 + $220 p4d | VERY LOW ✅✅ |

**Recommendation:** Spend 2-3 hours testing, save potentially $220 and 20 hours.

---

## 🎯 What Each Stage Catches

### **Stage 1 catches:**
- Missing dependencies
- Import errors
- Script permission issues
- Data download problems
- File path issues

### **Stage 2 catches:**
- GPU availability issues
- CUDA version mismatches
- Model loading errors
- Memory errors (batch size too large)
- Forward pass failures

### **Stage 3 catches:**
- Training loop bugs
- Checkpoint saving issues
- Validation errors
- Multi-experiment conflicts
- End-to-end pipeline issues

---

## 📝 Quick Test Script

Save as `scripts/run_all_tests.sh`:

```bash
#!/bin/bash
# Run all pre-flight tests

set -e

echo "============================================"
echo "Complete Pre-Flight Test Suite"
echo "============================================"
echo ""

# Stage 1
echo "Stage 1: Local tests..."
bash -c '[Stage 1 commands]'

# Stage 2
echo "Stage 2: GPU tests..."
bash -c '[Stage 2 commands]'

# Stage 3
echo "Stage 3: Mini training..."
bash -c '[Stage 3 commands]'

echo ""
echo "✅ ALL TESTS PASSED - READY FOR P4D"
```

---

## 🚦 Go/No-Go Decision

After testing:

### **GO (Launch p4d) if:**
- ✅ All 3 stages pass
- ✅ All 3 experiment configs work
- ✅ Checkpoints save/load correctly
- ✅ No memory errors
- ✅ S3 access works

### **NO-GO (Fix issues first) if:**
- ❌ Any test fails
- ❌ Import errors
- ❌ Model loading issues
- ❌ Training crashes
- ❌ Checkpoints don't save

---

**Bottom line: Invest 2-3 hours in testing to avoid wasting $220 and 20 hours!** 

Start with Stage 1 now?

