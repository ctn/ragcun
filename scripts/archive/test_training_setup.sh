#!/bin/bash
# Pre-flight test: Verify everything works before expensive training run
# Run this BEFORE starting the 18-hour training

set -e

echo "============================================"
echo "Pre-Flight Testing for p4d Training"
echo "============================================"
echo ""
echo "This will run quick tests to ensure:"
echo "  - All dependencies installed"
echo "  - GPUs accessible"
echo "  - Data loading works"
echo "  - Model initialization works"
echo "  - Training loop starts without errors"
echo ""
echo "Expected time: ~5 minutes"
echo ""

# Track failures
FAILURES=0

# Test 1: Python imports
echo "Test 1: Python imports..."
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
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then ((FAILURES++)); fi

# Test 2: CUDA availability
echo ""
echo "Test 2: CUDA availability..."
python << 'EOF'
import torch
import sys

if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"✅ CUDA available: {num_gpus} GPUs")

for i in range(num_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"   GPU {i}: {name} ({mem:.1f} GB)")

# Test GPU computation
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.mm(x, y)
print(f"✅ GPU computation works")
EOF
if [ $? -ne 0 ]; then ((FAILURES++)); fi

# Test 3: Data availability
echo ""
echo "Test 3: Data availability..."
if [ -f "data/processed/msmarco/train.json" ]; then
    echo "✅ MS MARCO train data found"
    
    # Check size
    SIZE=$(du -h data/processed/msmarco/train.json | cut -f1)
    echo "   Size: $SIZE"
    
    # Check format
    python << 'EOF'
import json
import sys

try:
    with open('data/processed/msmarco/train.json') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print("❌ Data is not a list")
        sys.exit(1)
    
    if len(data) == 0:
        print("❌ Data is empty")
        sys.exit(1)
    
    # Check first example
    example = data[0]
    required_keys = ['query', 'positive', 'negative']
    for key in required_keys:
        if key not in example:
            print(f"❌ Missing key: {key}")
            sys.exit(1)
    
    print(f"✅ Data format correct ({len(data):,} examples)")
    print(f"   Example query: {example['query'][:50]}...")
    
except Exception as e:
    print(f"❌ Data check failed: {e}")
    sys.exit(1)
EOF
    if [ $? -ne 0 ]; then ((FAILURES++)); fi
else
    echo "❌ MS MARCO data not found!"
    echo "   Run: python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    ((FAILURES++))
fi

# Test 4: Model loading
echo ""
echo "Test 4: Model loading..."
python << 'EOF'
import sys
import torch
from sentence_transformers import SentenceTransformer

try:
    print("Loading base model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = model.cuda()
    
    # Test encoding
    embeddings = model.encode(["Test sentence"], convert_to_tensor=True)
    print(f"✅ Base model loaded and works")
    print(f"   Output dim: {embeddings.shape[-1]}")
    print(f"   Model on GPU: {next(model.parameters()).is_cuda}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then ((FAILURES++)); fi

# Test 5: Custom model initialization
echo ""
echo "Test 5: Custom model (IsotropicGaussianEncoder)..."
python << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from ragcun.model import IsotropicGaussianEncoder
    import torch
    
    print("Initializing IsotropicGaussianEncoder...")
    model = IsotropicGaussianEncoder(
        output_dim=512,
        base_model='sentence-transformers/all-mpnet-base-v2',
        freeze_base=True
    )
    model = model.cuda()
    
    # Test forward pass
    test_input = ["This is a test sentence", "Another test sentence"]
    embeddings = model.encode(test_input)
    
    print(f"✅ Custom model works")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Output dim: {embeddings.shape[-1]}")
    print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
except Exception as e:
    print(f"❌ Custom model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then ((FAILURES++)); fi

# Test 6: Training script dry-run
echo ""
echo "Test 6: Training script dry-run (1 step)..."
if [ -f "data/processed/msmarco/train.json" ]; then
    timeout 120 python scripts/train.py \
        --train_data data/processed/msmarco/train.json \
        --val_data data/processed/msmarco/dev.json \
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
        --output_dir /tmp/test_training \
        --max_steps 2 \
        > /tmp/training_test.log 2>&1 || true
    
    if grep -q "Step 1/" /tmp/training_test.log || grep -q "Step 2/" /tmp/training_test.log; then
        echo "✅ Training loop starts successfully"
        tail -5 /tmp/training_test.log
    else
        echo "❌ Training loop failed to start"
        echo "Last 20 lines of log:"
        tail -20 /tmp/training_test.log
        ((FAILURES++))
    fi
    
    # Cleanup
    rm -rf /tmp/test_training
else
    echo "⚠️  Skipping (no data)"
fi

# Test 7: GPU memory test
echo ""
echo "Test 7: GPU memory test (batch size 64)..."
python << 'EOF'
import sys
import torch
from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model = model.cuda()
    model.eval()
    
    # Test batch size 64 (what we'll use on A100)
    batch = ["Test sentence " + str(i) for i in range(64)]
    
    with torch.no_grad():
        embeddings = model.encode(batch, batch_size=64, convert_to_tensor=True)
    
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"✅ Batch size 64 works")
    print(f"   Memory allocated: {mem_allocated:.2f} GB")
    print(f"   Memory reserved: {mem_reserved:.2f} GB")
    
    if mem_reserved > 35:  # A100 has 40GB
        print(f"⚠️  High memory usage, might need smaller batch")
    
except Exception as e:
    print(f"❌ Memory test failed: {e}")
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then ((FAILURES++)); fi

# Test 8: Parallel GPU test
echo ""
echo "Test 8: Multi-GPU test..."
python << 'EOF'
import torch
import sys

try:
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 3:
        print(f"⚠️  Only {num_gpus} GPUs available (need 3 for parallel training)")
    else:
        # Test computation on each GPU
        for i in range(min(3, num_gpus)):
            with torch.cuda.device(i):
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.mm(x, y)
                print(f"✅ GPU {i} works")
        
        print(f"✅ All GPUs accessible for parallel training")
    
except Exception as e:
    print(f"❌ Multi-GPU test failed: {e}")
    sys.exit(1)
EOF
if [ $? -ne 0 ]; then ((FAILURES++)); fi

# Test 9: S3 access
echo ""
echo "Test 9: S3 access..."
if [ -n "$S3_BACKUP_BUCKET" ]; then
    BUCKET=$(echo $S3_BACKUP_BUCKET | sed 's|s3://||')
    
    # Test write
    echo "test" > /tmp/test_s3.txt
    aws s3 cp /tmp/test_s3.txt s3://$BUCKET/test_s3.txt --quiet 2>&1
    
    if [ $? -eq 0 ]; then
        # Test read
        aws s3 cp s3://$BUCKET/test_s3.txt /tmp/test_s3_download.txt --quiet 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ S3 read/write works"
            echo "   Bucket: s3://$BUCKET"
            # Cleanup
            aws s3 rm s3://$BUCKET/test_s3.txt --quiet 2>&1
            rm /tmp/test_s3.txt /tmp/test_s3_download.txt
        else
            echo "❌ S3 read failed"
            ((FAILURES++))
        fi
    else
        echo "❌ S3 write failed"
        ((FAILURES++))
    fi
else
    echo "⚠️  S3_BACKUP_BUCKET not set"
    echo "   Backups will not be automatic"
fi

# Test 10: Disk space
echo ""
echo "Test 10: Disk space..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
AVAILABLE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

echo "Available space: $AVAILABLE"

if [ $AVAILABLE_GB -lt 50 ]; then
    echo "⚠️  Low disk space (need ~50GB for training)"
    ((FAILURES++))
else
    echo "✅ Sufficient disk space"
fi

# Summary
echo ""
echo "============================================"
echo "Pre-Flight Test Summary"
echo "============================================"
echo ""

if [ $FAILURES -eq 0 ]; then
    echo "✅ All tests passed! Ready for training."
    echo ""
    echo "Estimated costs:"
    echo "  - Training (18 hrs): ~\$189"
    echo "  - Evaluation (2 hrs): ~\$21"
    echo "  - Total: ~\$210-220"
    echo ""
    echo "Start training with:"
    echo "  tmux new -s training"
    echo "  ./scripts/train_parallel_p4d.sh"
    echo ""
    exit 0
else
    echo "❌ $FAILURES test(s) failed!"
    echo ""
    echo "Fix issues before starting expensive training."
    echo "Review logs above for details."
    echo ""
    exit 1
fi

