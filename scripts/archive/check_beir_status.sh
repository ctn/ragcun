#!/bin/bash

echo "======================================"
echo "BEIR EVALUATION STATUS"
echo "======================================"
echo ""

# Check if process is running
if pgrep -f "eval_beir_standard.sh" > /dev/null; then
    echo "✅ Evaluation is RUNNING"
else
    echo "⚠️  Evaluation is NOT running"
fi

echo ""
echo "--------------------------------------"
echo "COMPLETED EVALUATIONS:"
echo "--------------------------------------"

# Count completed result files
COMPLETED=$(find results/beir_standard -name "*.json" 2>/dev/null | wc -l)
TOTAL=9

echo "Progress: $COMPLETED / $TOTAL evaluations"
echo ""

# List completed evaluations
if [ -d "results/beir_standard" ]; then
    for file in results/beir_standard/*.json; do
        if [ -f "$file" ]; then
            basename "$file" .json
        fi
    done
fi

echo ""
echo "--------------------------------------"
echo "CURRENT ACTIVITY:"
echo "--------------------------------------"

# Show last 20 lines of log
if [ -f "logs/beir_standard_full.log" ]; then
    tail -20 logs/beir_standard_full.log
else
    echo "No log file found"
fi

echo ""
echo "--------------------------------------"
echo "GPU STATUS:"
echo "--------------------------------------"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "======================================"
echo "To view full log: tail -f logs/beir_standard_full.log"
echo "======================================"

