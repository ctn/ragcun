#!/bin/bash
# Parallel training on AWS multi-GPU instance
# Optimized for p3.8xlarge (4× V100) or p4d.24xlarge (8× A100)

set -e

echo "============================================"
echo "Parallel Training on AWS Multi-GPU"
echo "============================================"
echo ""

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected: $NUM_GPUS GPUs"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

echo ""
echo "Starting all 3 experiments in parallel..."
echo ""

# Confirmation
read -p "This will use 3 GPUs for ~36 hours. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check data
if [ ! -f "data/processed/msmarco/train.json" ]; then
    echo ""
    echo "❌ MS MARCO not found."
    echo "Download first: python scripts/download_msmarco.py --output_dir data/processed/msmarco"
    exit 1
fi

# Create directories
mkdir -p logs checkpoints

# Record start time
START_TIME=$(date +%s)
echo "Started at: $(date)"
echo ""

# Common arguments - OPTIMIZED for V100/A100
COMMON_ARGS="--train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --epochs 3 \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --mixed_precision \
    --save_interval 1 \
    --log_interval 50"

# Experiment 1: Baseline (no isotropy) on GPU 0
echo "============================================"
echo "Experiment 1: Baseline (no isotropy)"
echo "GPU: 0"
echo "============================================"
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    $COMMON_ARGS \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 0.0 \
    --lambda_reg 0.0 \
    --output_dir checkpoints/baseline_no_isotropy \
    2>&1 | tee logs/baseline_no_isotropy.log &
PID1=$!
echo "Started with PID: $PID1"
echo ""
sleep 5

# Experiment 2: With isotropy (YOUR METHOD) on GPU 1
echo "============================================"
echo "Experiment 2: With Isotropy (YOUR METHOD)"
echo "GPU: 1"
echo "============================================"
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    $COMMON_ARGS \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/with_isotropy \
    2>&1 | tee logs/with_isotropy.log &
PID2=$!
echo "Started with PID: $PID2"
echo ""
sleep 5

# Experiment 3: Frozen base (efficiency) on GPU 2
echo "============================================"
echo "Experiment 3: Frozen Base (Efficiency)"
echo "GPU: 2"
echo "============================================"
CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
    $COMMON_ARGS \
    --freeze_base True \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --output_dir checkpoints/frozen_efficient \
    2>&1 | tee logs/frozen_efficient.log &
PID3=$!
echo "Started with PID: $PID3"
echo ""

# Summary
echo "============================================"
echo "All Experiments Running!"
echo "============================================"
echo ""
echo "Process IDs:"
echo "  Experiment 1 (Baseline):      PID $PID1 on GPU 0"
echo "  Experiment 2 (With Isotropy): PID $PID2 on GPU 1"
echo "  Experiment 3 (Frozen):        PID $PID3 on GPU 2"
echo ""
echo "Expected duration: ~36 hours on V100, ~18 hours on A100"
echo ""
echo "Monitor progress:"
echo "  - GPU usage:      watch -n 1 nvidia-smi"
echo "  - Training logs:  tail -f logs/*.log"
echo "  - All logs:       tmux attach -t monitoring"
echo ""
echo "To stop all:"
echo "  kill $PID1 $PID2 $PID3"
echo ""

# Create monitoring tmux session
tmux new-session -d -s monitoring 'bash -c "
    tmux split-window -v;
    tmux split-window -v;
    tmux select-pane -t 0;
    tmux send-keys \"tail -f logs/baseline_no_isotropy.log\" C-m;
    tmux select-pane -t 1;
    tmux send-keys \"tail -f logs/with_isotropy.log\" C-m;
    tmux select-pane -t 2;
    tmux send-keys \"tail -f logs/frozen_efficient.log\" C-m;
"' 2>/dev/null || true

echo "Monitoring session created. Attach with: tmux attach -t monitoring"
echo ""

# Wait for all processes
echo "Waiting for training to complete..."
echo ""

wait $PID1
ELAPSED1=$(($(date +%s) - START_TIME))
echo ""
echo "✅ Experiment 1 (Baseline) complete!"
echo "   Elapsed: $(($ELAPSED1 / 3600))h $(($ELAPSED1 % 3600 / 60))m"
echo "   Model: checkpoints/baseline_no_isotropy/best_model.pt"
echo ""

wait $PID2
ELAPSED2=$(($(date +%s) - START_TIME))
echo ""
echo "✅ Experiment 2 (With Isotropy) complete!"
echo "   Elapsed: $(($ELAPSED2 / 3600))h $(($ELAPSED2 % 3600 / 60))m"
echo "   Model: checkpoints/with_isotropy/best_model.pt"
echo ""

wait $PID3
ELAPSED3=$(($(date +%s) - START_TIME))
echo ""
echo "✅ Experiment 3 (Frozen Base) complete!"
echo "   Elapsed: $(($ELAPSED3 / 3600))h $(($ELAPSED3 % 3600 / 60))m"
echo "   Model: checkpoints/frozen_efficient/best_model.pt"
echo ""

# Final summary
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "============================================"
echo "✅ All Training Complete!"
echo "============================================"
echo ""
echo "Total wall-clock time: $(($TOTAL_ELAPSED / 3600))h $(($TOTAL_ELAPSED % 3600 / 60))m"
echo "Finished at: $(date)"
echo ""
echo "Models saved:"
echo "  1. checkpoints/baseline_no_isotropy/best_model.pt"
echo "  2. checkpoints/with_isotropy/best_model.pt"
echo "  3. checkpoints/frozen_efficient/best_model.pt"
echo ""
echo "Training logs: logs/"
echo ""
echo "Next steps:"
echo ""
echo "1. Evaluate all models on BEIR:"
echo "   ./scripts/evaluate_all_beir.sh"
echo ""
echo "2. Sync to S3 (if configured):"
echo "   aws s3 sync checkpoints/ s3://your-bucket/ragcun-training/checkpoints/"
echo "   aws s3 sync results/ s3://your-bucket/ragcun-training/results/"
echo ""
echo "3. Generate comparison table:"
echo "   python scripts/generate_comparison_table.py"
echo ""
echo "🎉 You now have publication-ready models!"
echo ""

