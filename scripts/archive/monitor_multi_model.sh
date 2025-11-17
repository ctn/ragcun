#!/bin/bash
# Monitor multi-model smoke test progress

LOG_FILE="/home/ubuntu/ragcun/logs/multi_model_smoke_test.log"
WATCH_INTERVAL=60  # Check every 60 seconds

echo "🔍 Monitoring Multi-Model Smoke Test"
echo "======================================"
echo ""

while true; do
    clear
    echo "🔍 Multi-Model Smoke Test Monitor"
    echo "======================================"
    echo "Time: $(date +'%H:%M:%S')"
    echo ""
    
    # Check if process is still running
    if ! pgrep -f "smoke_test_multi_model.sh" > /dev/null; then
        echo "❌ Test process not found!"
        echo ""
        echo "Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        break
    fi
    
    echo "✅ Test is RUNNING"
    echo ""
    
    # Current model being trained
    CURRENT_MODEL=$(tail -100 "$LOG_FILE" | grep "Testing:" | tail -1)
    if [ -n "$CURRENT_MODEL" ]; then
        echo "📊 Current Model:"
        echo "   $CURRENT_MODEL"
        echo ""
    fi
    
    # Check for errors
    ERROR_COUNT=$(grep -i "error\|failed\|exception" "$LOG_FILE" 2>/dev/null | grep -v "FutureWarning" | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  Detected $ERROR_COUNT potential errors"
        echo ""
        echo "Recent errors:"
        grep -i "error\|failed\|exception" "$LOG_FILE" | grep -v "FutureWarning" | tail -5
        echo ""
    else
        echo "✅ No errors detected"
        echo ""
    fi
    
    # GPU status
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "   GPU %s: %s util, %s / %s, %s\n", $1, $2, $3, $4, $5}'
    echo ""
    
    # Training progress
    echo "📈 Recent Progress (last 5 lines):"
    tail -5 "$LOG_FILE" | sed 's/^/   /'
    echo ""
    
    # Checkpoints created
    CHECKPOINT_COUNT=$(find /home/ubuntu/ragcun/checkpoints/smoke_multi -name "best_model.pt" 2>/dev/null | wc -l)
    echo "💾 Checkpoints completed: $CHECKPOINT_COUNT / 10"
    echo ""
    
    echo "======================================"
    echo "Monitoring... (Ctrl+C to stop monitor)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    sleep "$WATCH_INTERVAL"
done

