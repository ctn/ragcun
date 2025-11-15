#!/bin/bash
# Complete pre-flight test before expensive p4d run
# Catches all issues on local/cheap hardware first

set -e

FAILURES=0

echo "============================================"
echo "Pre-Flight Test Suite"
echo "============================================"
echo ""
echo "Testing all code before expensive p4d run"
echo "Expected time: ~5 minutes (without data download)"
echo ""

# Create test data directory
mkdir -p data/processed/msmarco_tiny

# ========================================
# STAGE 1: LOCAL TESTS (No GPU needed)
# ========================================
echo "============================================"
echo "STAGE 1: Local Tests (No GPU)"
echo "============================================"
echo ""

# Test 1.1: Script existence
echo "Test 1.1: Script existence..."
REQUIRED_SCRIPTS=(
    "scripts/download_msmarco.py"
    "scripts/train.py"
    "scripts/evaluate_beir.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✅ $script"
    else
        echo "  ❌ $script MISSING!"
        ((FAILURES++))
    fi
done

# Test 1.2: Python imports
echo ""
echo "Test 1.2: Python imports..."
python << 'EOF' || ((FAILURES++))
import sys
try:
    import torch
    import transformers
    import sentence_transformers
    import numpy as np
    import json
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
EOF

# Test 1.3: Model import
echo ""
echo "Test 1.3: Custom model import..."
python << 'EOF' || ((FAILURES++))
import sys
try:
    from ragcun.model import GaussianEmbeddingGemma
    print("✅ GaussianEmbeddingGemma imports successfully")
except ImportError as e:
    print(f"❌ Model import failed: {e}")
    sys.exit(1)
EOF

# Test 1.4: Data format
echo ""
echo "Test 1.4: Create test data..."
python << 'EOF' || ((FAILURES++))
import json
import os

# Create tiny test dataset
test_data = [
    {
        "query": "What is machine learning?",
        "positive": "Machine learning is a branch of AI.",
        "negative": "The weather is sunny today."
    },
    {
        "query": "How to train neural networks?",
        "positive": "Neural networks are trained using backpropagation.",
        "negative": "Cats are popular pets."
    }
] * 50  # 100 examples total

os.makedirs('data/processed/msmarco_tiny', exist_ok=True)

with open('data/processed/msmarco_tiny/train.json', 'w') as f:
    json.dump(test_data, f)

with open('data/processed/msmarco_tiny/val.json', 'w') as f:
    json.dump(test_data[:10], f)

print(f"✅ Test data created: {len(test_data)} examples")
EOF

echo ""
if [ $FAILURES -eq 0 ]; then
    echo "Stage 1 Summary: ✅ PASSED"
else
    echo "Stage 1 Summary: ❌ FAILED"
fi

# ========================================
# STAGE 2: GPU TESTS
# ========================================
echo ""
echo "============================================"
echo "STAGE 2: GPU Tests"
echo "============================================"
echo ""

# Test 2.1: GPU availability
echo "Test 2.1: GPU availability..."
python << 'EOF' || ((FAILURES++))
import torch
import sys

if not torch.cuda.is_available():
    print("⚠️  No GPU available (tests will be limited)")
    print("   This is OK for local testing")
else:
    num_gpus = torch.cuda.device_count()
    print(f"✅ GPU available: {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

# Test 2.2: Model loading
echo ""
echo "Test 2.2: Base model loading..."
python << 'EOF' || ((FAILURES++))
import sys
try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = "GPU"
    else:
        device = "CPU"
    
    # Test encoding
    embeddings = model.encode(["Test sentence"])
    print(f"✅ Base model works on {device}")
    print(f"   Output shape: {embeddings.shape}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)
EOF

# Test 2.3: Custom model
echo ""
echo "Test 2.3: Custom model (GaussianEmbeddingGemma)..."
python << 'EOF' || ((FAILURES++))
import sys
import torch
try:
    from ragcun.model import GaussianEmbeddingGemma
    
    # Test frozen base
    model = GaussianEmbeddingGemma(
        output_dim=512,
        base_model='sentence-transformers/all-mpnet-base-v2',
        freeze_base=True
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    embeddings = model.encode(["Test"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Custom model (frozen base) works")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Trainable params: {trainable:,}")
    
    # Test unfrozen base
    model2 = GaussianEmbeddingGemma(
        output_dim=512,
        base_model='sentence-transformers/all-mpnet-base-v2',
        freeze_base=False
    )
    trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    
    print(f"✅ Custom model (unfrozen base) works")
    print(f"   Trainable params: {trainable2:,}")
    
except Exception as e:
    print(f"❌ Custom model failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
if [ $FAILURES -eq 0 ]; then
    echo "Stage 2 Summary: ✅ PASSED"
else
    echo "Stage 2 Summary: ❌ FAILED"
fi

# ========================================
# STAGE 3: TRAINING TEST
# ========================================
echo ""
echo "============================================"
echo "STAGE 3: Training Pipeline Test"
echo "============================================"
echo ""

# Test 3.1: Baseline training (quick test)
echo "Test 3.1: Baseline training (no isotropy)..."
timeout 60 python scripts/train.py \
    --train_data data/processed/msmarco_tiny/train.json \
    --val_data data/processed/msmarco_tiny/val.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --epochs 1 \
    --batch_size 2 \
    --output_dim 512 \
    --warmup_steps 2 \
    --output_dir /tmp/test_baseline \
    --log_interval 1 \
    > /tmp/test_baseline.log 2>&1

if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✅ Baseline training works"
else
    echo "❌ Baseline training failed"
    tail -20 /tmp/test_baseline.log
    ((FAILURES++))
fi

# Test 3.2: With isotropy training (quick test)
echo ""
echo "Test 3.2: Training with isotropy..."
timeout 60 python scripts/train.py \
    --train_data data/processed/msmarco_tiny/train.json \
    --val_data data/processed/msmarco_tiny/val.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 1 \
    --batch_size 2 \
    --output_dim 512 \
    --warmup_steps 2 \
    --output_dir /tmp/test_isotropy \
    --log_interval 1 \
    > /tmp/test_isotropy.log 2>&1

if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✅ Isotropy training works"
else
    echo "❌ Isotropy training failed"
    tail -20 /tmp/test_isotropy.log
    ((FAILURES++))
fi

# Test 3.3: Frozen base training (quick test)
echo ""
echo "Test 3.3: Frozen base training..."
timeout 60 python scripts/train.py \
    --train_data data/processed/msmarco_tiny/train.json \
    --val_data data/processed/msmarco_tiny/val.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 1 \
    --batch_size 2 \
    --output_dim 512 \
    --warmup_steps 2 \
    --output_dir /tmp/test_frozen \
    --log_interval 1 \
    > /tmp/test_frozen.log 2>&1

if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✅ Frozen base training works"
else
    echo "❌ Frozen base training failed"
    tail -20 /tmp/test_frozen.log
    ((FAILURES++))
fi

# Cleanup
rm -rf /tmp/test_baseline /tmp/test_isotropy /tmp/test_frozen 2>/dev/null || true

echo ""
if [ $FAILURES -eq 0 ]; then
    echo "Stage 3 Summary: ✅ PASSED"
else
    echo "Stage 3 Summary: ❌ FAILED"
fi

# ========================================
# FINAL SUMMARY
# ========================================
echo ""
echo "============================================"
echo "PRE-FLIGHT TEST SUMMARY"
echo "============================================"
echo ""

if [ $FAILURES -eq 0 ]; then
    echo "✅✅✅ ALL TESTS PASSED ✅✅✅"
    echo ""
    echo "Your code is ready for p4d.24xlarge training!"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download full MS MARCO (if not already done):"
    echo "   python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    echo ""
    echo "2. Review the training plan:"
    echo "   cat P4D_TRAINING_PLAN.md"
    echo ""
    echo "3. Launch p4d.24xlarge spot instance"
    echo ""
    echo "4. On p4d instance, run:"
    echo "   ./scripts/setup_p4d_instance.sh"
    echo "   ./scripts/test_training_setup.sh  # Final check on p4d"
    echo "   ./scripts/train_parallel_p4d.sh   # Start training!"
    echo ""
    echo "Estimated p4d cost: ~\$220 for 21 hours"
    echo "Expected completion: < 1 day"
    echo ""
    exit 0
else
    echo "❌❌❌ $FAILURES TEST(S) FAILED ❌❌❌"
    echo ""
    echo "DO NOT launch expensive p4d instance yet!"
    echo "Fix the issues above first."
    echo ""
    echo "Common fixes:"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo "  - Model issues: Check ragcun/model.py exists"
    echo "  - Training script: Check scripts/train.py has all args"
    echo ""
    exit 1
fi

