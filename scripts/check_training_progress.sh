#!/bin/bash
# Check actual training progress with detailed information

echo "📊 Actual Training Progress"
echo "=" * 80
echo ""

LOG_FILE="logs/jepa_xy_masked_training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Get latest batch number
LATEST_BATCH=$(grep -oE '[0-9]+/7420' "$LOG_FILE" | tail -1 | cut -d'/' -f1)
TOTAL_BATCHES=7420

if [ -z "$LATEST_BATCH" ]; then
    echo "⚠️  No batch progress found in log"
else
    PROGRESS_PCT=$(echo "scale=2; $LATEST_BATCH * 100 / $TOTAL_BATCHES" | bc)
    REMAINING_BATCHES=$((TOTAL_BATCHES - LATEST_BATCH))
    
    echo "Epoch Progress:"
    echo "  Current: Epoch 1/3"
    echo "  Batch: $LATEST_BATCH / $TOTAL_BATCHES ($PROGRESS_PCT%)"
    echo "  Remaining: $REMAINING_BATCHES batches"
    echo ""
    
    # Estimate time remaining
    if [ "$LATEST_BATCH" -gt 10 ]; then
        # Get average time per batch from recent logs
        AVG_TIME=$(grep -E '[0-9]+/7420.*\[.*it/s' "$LOG_FILE" | tail -5 | grep -oE '[0-9]+\.[0-9]+it/s' | sed 's/it\/s//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "1.7"}')
        EST_HOURS=$(echo "scale=2; $REMAINING_BATCHES / ($AVG_TIME * 3600)" | bc)
        EST_MIN=$(echo "scale=0; $REMAINING_BATCHES / $AVG_TIME / 60" | bc)
        echo "  Estimated time remaining: ~${EST_MIN} minutes (~${EST_HOURS} hours)"
    fi
    echo ""
fi

# Check for epoch summaries
EPOCH_SUMMARIES=$(grep "Epoch.*Train Loss:" "$LOG_FILE" | tail -3)
if [ -n "$EPOCH_SUMMARIES" ]; then
    echo "Epoch Summaries:"
    echo "$EPOCH_SUMMARIES" | while read line; do
        echo "  $line"
    done
    echo ""
else
    echo "⏳ No epoch summaries yet (still in first epoch)"
    echo ""
fi

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s%% | Mem: %s/%s MB | Temp: %s°C\n", NR-1, $1, $2, $3, $4}'
echo ""

# Check process status
if ps aux | grep -E "train_xy_masked|python.*train_xy_masked" | grep -v grep > /dev/null; then
    echo "✅ Training process is RUNNING"
    ps aux | grep -E "train_xy_masked|python.*train_xy_masked" | grep -v grep | head -1 | \
        awk '{printf "  PID: %s | CPU: %s%% | Mem: %s%%\n", $2, $3, $4}'
else
    echo "❌ Training process is NOT running"
fi
echo ""

# Check checkpoints
if [ -d "checkpoints/jepa_xy_masked" ]; then
    CHECKPOINTS=$(ls -1 checkpoints/jepa_xy_masked/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
    if [ "$CHECKPOINTS" -gt 0 ]; then
        echo "✅ Checkpoints saved: $CHECKPOINTS"
        ls -1t checkpoints/jepa_xy_masked/checkpoint_epoch_*.pt 2>/dev/null | head -1 | \
            xargs ls -lh | awk '{print "  Latest: " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'
    else
        echo "⏳ No checkpoints saved yet (will save after epoch 1 completes)"
    fi
else
    echo "⏳ Checkpoint directory not created yet"
fi
echo ""

# Show latest log lines
echo "Latest Log Activity (last 3 lines):"
tail -3 "$LOG_FILE" | sed 's/^/  /'
echo ""

