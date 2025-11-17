#!/bin/bash
# Pilot Training: Meaningful subset to validate approach
# 
# This trains on enough data to see real results, but fast enough
# to validate before committing to full 15-day or $220 AWS training
#
# Expected time: 1-2 days on local T4
# Expected results: Clear enough to validate isotropy helps

set -e

echo "============================================"
echo "Pilot Training Run"
echo "============================================"
echo ""
echo "Strategy: Train on MS MARCO subset to validate approach"
echo "  - Data: 50K training examples (vs 500K full)"
echo "  - Epochs: 2 (vs 3 full)"
echo "  - Experiments: 2 key ones (baseline + isotropy)"
echo ""
echo "Expected time: ~1-2 days on T4"
echo "Goal: Validate isotropy helps before full training"
echo ""

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for MS MARCO data
if [ ! -f "data/processed/msmarco/train.json" ]; then
    echo "❌ MS MARCO data not found"
    echo ""
    echo "Download first:"
    echo "  python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    exit 1
fi

# Create pilot subset
echo "Step 1: Creating pilot dataset (50K examples)..."
python << 'EOF'
import json
from pathlib import Path

# Load full MS MARCO
with open('data/processed/msmarco/train.json') as f:
    full_data = json.load(f)

# Take first 50K
pilot_data = full_data[:50000]

# Save pilot dataset
Path('data/processed/msmarco_pilot').mkdir(parents=True, exist_ok=True)
with open('data/processed/msmarco_pilot/train.json', 'w') as f:
    json.dump(pilot_data, f)

# Use dev as-is
with open('data/processed/msmarco/dev.json') as f:
    dev_data = json.load(f)

with open('data/processed/msmarco_pilot/dev.json', 'w') as f:
    json.dump(dev_data[:1000], f)  # Smaller dev for speed

print(f"✅ Pilot dataset created: {len(pilot_data):,} train, 1K dev")
EOF

# Common arguments
COMMON_ARGS="--train_data data/processed/msmarco_pilot/train.json \
    --val_data data/processed/msmarco_pilot/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 2 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --mixed_precision \
    --save_interval 1 \
    --log_interval 100"

# Experiment 1: Baseline (no isotropy)
echo ""
echo "============================================"
echo "Experiment 1: Baseline (no isotropy)"
echo "Started: $(date)"
echo "============================================"

python scripts/train.py \
    $COMMON_ARGS \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --output_dir checkpoints/pilot_baseline \
    2>&1 | tee logs/pilot_baseline.log

echo "✅ Experiment 1 complete: $(date)"

# Experiment 2: With isotropy (YOUR METHOD)
echo ""
echo "============================================"
echo "Experiment 2: With Isotropy (YOUR METHOD)"
echo "Started: $(date)"
echo "============================================"

python scripts/train.py \
    $COMMON_ARGS \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/pilot_isotropy \
    2>&1 | tee logs/pilot_isotropy.log

echo "✅ Experiment 2 complete: $(date)"

# Quick evaluation on subset of BEIR
echo ""
echo "============================================"
echo "Quick BEIR Evaluation"
echo "============================================"
echo ""
echo "Evaluating on representative BEIR datasets..."
echo ""

# Baseline
echo "Evaluating baseline..."
python scripts/evaluate_beir.py \
    --model_path checkpoints/pilot_baseline/best_model.pt \
    --datasets scifact nfcorpus arguana \
    --output_file results/pilot_baseline_beir.json \
    2>&1 | tee logs/pilot_eval_baseline.log

# With isotropy
echo ""
echo "Evaluating with isotropy..."
python scripts/evaluate_beir.py \
    --model_path checkpoints/pilot_isotropy/best_model.pt \
    --datasets scifact nfcorpus arguana \
    --output_file results/pilot_isotropy_beir.json \
    2>&1 | tee logs/pilot_eval_isotropy.log

# Summary
echo ""
echo "============================================"
echo "✅ Pilot Training Complete!"
echo "============================================"
echo ""
echo "Results saved:"
echo "  - checkpoints/pilot_baseline/"
echo "  - checkpoints/pilot_isotropy/"
echo "  - results/pilot_*_beir.json"
echo ""

# Quick comparison
python << 'EOF'
import json
from pathlib import Path

def get_avg_score(file_path):
    if not Path(file_path).exists():
        return None
    with open(file_path) as f:
        data = json.load(f)
    if 'average' in data:
        return data['average'].get('ndcg@10', 0) * 100
    return None

baseline = get_avg_score('results/pilot_baseline_beir.json')
isotropy = get_avg_score('results/pilot_isotropy_beir.json')

print("Quick Results (on 3 BEIR datasets):")
print("=" * 50)
if baseline:
    print(f"Baseline (no isotropy): {baseline:.1f}% NDCG@10")
if isotropy:
    print(f"With isotropy:          {isotropy:.1f}% NDCG@10")
if baseline and isotropy:
    improvement = isotropy - baseline
    print(f"Improvement:            {improvement:+.1f}%")
    print("")
    if improvement > 0.5:
        print("✅ Isotropy is helping! Ready for full training.")
    elif improvement > 0:
        print("⚠️  Small improvement. Consider tuning hyperparameters.")
    else:
        print("❌ No improvement. Check configuration before full training.")
else:
    print("⚠️  Could not load results. Check logs above.")
print("=" * 50)
EOF

echo ""
echo "Next steps:"
echo ""
echo "If isotropy helped (>0.5% improvement):"
echo "  → Proceed to full training!"
echo "  → Local: ./scripts/train_publication_recommended.sh"
echo "  → AWS:   Launch p4d and run ./scripts/train_parallel_p4d.sh"
echo ""
echo "If results unclear:"
echo "  → Check logs in logs/pilot_*.log"
echo "  → Try different hyperparameters"
echo "  → Evaluate on more BEIR datasets"
echo ""

