#!/bin/bash
# Quick script to check training status

echo "============================================="
echo "Training Status Check"
echo "============================================="
echo ""

# Check if training process is running
echo "1. Process Status:"
echo "-------------------"
if ps aux | grep -E "train_xy_masked|python.*train_xy_masked" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    ps aux | grep -E "train_xy_masked|python.*train_xy_masked" | grep -v grep | head -1 | awk '{print "   PID: " $2 " | CPU: " $3 "% | Mem: " $4 "%"}'
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "2. GPU Usage:"
echo "-------------------"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s: %s%% | Mem: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4, $5}'

echo ""
echo "3. Latest Training Progress:"
echo "-------------------"
if [ -f "logs/jepa_xy_masked_training.log" ]; then
    # Extract epoch and batch info
    tail -5 logs/jepa_xy_masked_training.log | grep -E "Epoch|loss|batch" | tail -3
    echo ""
    echo "   Full log: tail -f logs/jepa_xy_masked_training.log"
else
    echo "   Log file not found"
fi

echo ""
echo "4. Checkpoint Status:"
echo "-------------------"
if [ -d "checkpoints/jepa_xy_masked" ]; then
    if [ -f "checkpoints/jepa_xy_masked/best_model.pt" ]; then
        echo "   ✅ Best model exists"
        ls -lh checkpoints/jepa_xy_masked/best_model.pt | awk '{print "   Size: " $5 " | Modified: " $6 " " $7 " " $8}'
    else
        echo "   ⏳ Best model not yet saved (training in progress)"
    fi
    
    # Count checkpoints
    checkpoint_count=$(ls -1 checkpoints/jepa_xy_masked/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
    if [ $checkpoint_count -gt 0 ]; then
        echo "   Checkpoints saved: $checkpoint_count"
        ls -1t checkpoints/jepa_xy_masked/checkpoint_epoch_*.pt 2>/dev/null | head -1 | xargs ls -lh | awk '{print "   Latest: " $9 " (" $5 ")"}'
    fi
else
    echo "   ⏳ Checkpoint directory not created yet"
fi

echo ""
echo "============================================="
echo ""
echo "Useful commands:"
echo "  • Watch logs live:     tail -f logs/jepa_xy_masked_training.log"
echo "  • Check GPU:           nvidia-smi"
echo "  • Check process:       ps aux | grep train_xy_masked"
echo "  • View full log:       less logs/jepa_xy_masked_training.log"
echo ""

