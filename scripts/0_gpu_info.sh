#!/bin/bash
# Display GPU information and recommendations

echo "============================================"
echo "GPU Information"
echo "============================================"
echo ""

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    echo "No GPU detected or NVIDIA drivers not installed"
    exit 1
fi

# GPU details
echo "GPU Details:"
echo "----------------------------------------"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader | \
    awk -F', ' '{printf "  Name: %s\n  Memory: %s\n  Compute Capability: %s\n  Driver Version: %s\n", $1, $2, $3, $4}'

echo ""

# Memory info
echo "Memory Usage:"
echo "----------------------------------------"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader | \
    awk -F', ' '{
        used=$1; free=$2; total=$3;
        gsub(/ MiB/, "", used); gsub(/ MiB/, "", free); gsub(/ MiB/, "", total);
        pct=int(used/total*100);
        printf "  Used: %s MiB (%.0f%%)\n  Free: %s MiB\n  Total: %s MiB\n", used, pct, free, total
    }'

echo ""

# GPU utilization
echo "Current Utilization:"
echo "----------------------------------------"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU: %s\n  Memory: %s\n  Temperature: %s\n", $1, $2, $3}'

echo ""

# Recommendations
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

echo "Recommended Batch Sizes:"
echo "----------------------------------------"

if [ "$FREE_MEM" -gt 12000 ]; then
    echo "  ✅ Batch size 16-32: Plenty of memory available"
    echo "  ✅ Batch size 8: Safe (recommended)"
    echo "  ⚠️  Batch size 4: Conservative (underutilizing GPU)"
elif [ "$FREE_MEM" -gt 8000 ]; then
    echo "  ✅ Batch size 8-16: Good amount of memory"
    echo "  ✅ Batch size 8: Safe (recommended)"
    echo "  ⚠️  Batch size 4: Conservative"
elif [ "$FREE_MEM" -gt 4000 ]; then
    echo "  ✅ Batch size 8: Should work"
    echo "  ✅ Batch size 4: Safe (recommended)"
    echo "  ⚠️  Batch size 16: May cause OOM"
else
    echo "  ⚠️  Low memory! Consider:"
    echo "      - Batch size 4 or lower"
    echo "      - Close other GPU processes"
    echo "      - Use smaller model"
fi

echo ""
echo "Current Training Processes:"
echo "----------------------------------------"
if pgrep -f "train.py" > /dev/null; then
    ps aux | grep "train.py" | grep -v grep | awk '{printf "  PID: %s | CPU: %s%% | Mem: %s%%\n", $2, $3, $4}'
else
    echo "  No training processes running"
fi

echo ""
echo "Full GPU Status:"
echo "----------------------------------------"
nvidia-smi
