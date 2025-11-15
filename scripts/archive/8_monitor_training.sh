#!/bin/bash
# Monitor training progress in real-time

echo "============================================"
echo "Training Monitor"
echo "============================================"
echo ""

# Check if training is running
if ! pgrep -f "scripts/train.py" > /dev/null; then
    echo "⚠️  No training process detected"
    echo ""
    echo "Start training with:"
    echo "  ./scripts/train_quick.sh"
    echo "  ./scripts/train_full.sh"
    exit 0
fi

echo "Training process detected!"
echo ""
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""
echo "============================================"
echo ""

# Monitor GPU and training log
watch -n 2 '
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, '\''{printf "GPU Util: %s%% | Mem: %sMB/%sMB (%s%%) | Temp: %s°C\n", $1, $3, $4, $2, $5}'\'' || echo "No GPU detected"
echo ""
echo "=== Recent Training Logs ==="
tail -n 15 training.log 2>/dev/null || echo "No training.log found"
echo ""
echo "=== Running Processes ==="
ps aux | grep -E "train.py|python" | grep -v grep | awk '\''{printf "PID: %s | CPU: %s%% | Mem: %s%%\n", $2, $3, $4}'\''
'
