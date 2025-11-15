#!/bin/bash
# Quick status checker for smoke test

echo "============================================"
echo "🔬 Smoke Test Status"
echo "============================================"
echo ""

# Check if process is running
if ps aux | grep -q "[t]rain_smoke_test.sh"; then
    echo "✅ Smoke test is RUNNING"
elif ps aux | grep -q "python.*train.py.*smoke"; then
    echo "✅ Training process is RUNNING"
else
    echo "⚠️  Smoke test process not found"
    echo "   (may have finished or failed)"
fi

echo ""

# Check log file
if [ -f smoke_test_run.log ]; then
    LOG_SIZE=$(du -h smoke_test_run.log | cut -f1)
    echo "📄 Log file: smoke_test_run.log (${LOG_SIZE})"
    echo ""
    
    # Show current stage
    if grep -q "Experiment 2:" smoke_test_run.log; then
        echo "📍 Current stage: Experiment 2 (With Isotropy)"
        if grep -q "Quick Analysis" smoke_test_run.log; then
            echo "📍 Status: Running analysis..."
        elif grep -q "Epoch 1:" smoke_test_run.log | tail -1 | grep -q "Epoch 1:"; then
            PROGRESS=$(grep "Epoch 1:" smoke_test_run.log | tail -1)
            echo "📍 Progress: $PROGRESS"
        fi
    elif grep -q "Experiment 1:" smoke_test_run.log; then
        echo "📍 Current stage: Experiment 1 (Baseline)"
        if grep -q "✅ Baseline done" smoke_test_run.log; then
            echo "📍 Status: Baseline complete, starting isotropy..."
        else
            PROGRESS=$(grep "Epoch 1:" smoke_test_run.log | tail -1)
            echo "📍 Progress: $PROGRESS"
        fi
    else
        echo "📍 Status: Starting up..."
    fi
    
    echo ""
    echo "Last 10 lines of log:"
    echo "────────────────────────────────────────"
    tail -10 smoke_test_run.log
    echo "────────────────────────────────────────"
else
    echo "❌ Log file not found: smoke_test_run.log"
fi

echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "   %s: %s%% GPU | %sMB / %sMB RAM\n", $1, $2, $3, $4}'
else
    echo "⚠️  nvidia-smi not available"
fi

echo ""
echo "Commands:"
echo "  tail -f smoke_test_run.log  # Watch live"
echo "  nvidia-smi                  # Check GPU"
echo "  ./scripts/check_smoke_status.sh  # Run this again"
echo ""

