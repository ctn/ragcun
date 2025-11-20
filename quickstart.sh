#!/bin/bash
# Quickstart script for RAGCUN training

set -e

echo "============================================"
echo "RAGCUN Training Quickstart"
echo "============================================"
echo ""

# Check GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "Warning: No GPU detected"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -e . -q
echo "✅ Dependencies installed"
echo ""

# Prepare sample data
echo "Preparing sample data..."
python scripts/prepare_data.py \
    --documents data/raw/sample_docs.txt \
    --generate_pairs \
    --num_pairs 100 \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed

echo "✅ Data prepared"
echo ""

# Train model
echo "Starting training (this will take a few minutes)..."
python scripts/train/isotropic.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 2 \
    --batch_size 8 \
    --output_dir checkpoints \
    --log_interval 5

echo ""
echo "✅ Training complete!"
echo ""

# Evaluate model
echo "Evaluating model..."
python scripts/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --test_data data/processed/test_eval.json \
    --batch_size 32 \
    --output_file results/quickstart_results.json

echo ""
echo "============================================"
echo "Quickstart Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Check training logs: cat training.log"
echo "  2. View results: cat results/quickstart_results.json"
echo "  3. Try the model in Python:"
echo ""
echo "     from ragcun import IsotropicRetriever"
echo "     retriever = IsotropicRetriever('checkpoints/best_model.pt')"
echo "     retriever.add_documents([...])"
echo "     results = retriever.retrieve('your query', top_k=5)"
echo ""
echo "For more details, see TRAINING_GUIDE.md"
