#!/bin/bash
# Script to generate training data from raw documents

set -e

echo "============================================"
echo "RAGCUN Training Data Generation"
echo "============================================"
echo ""

# Create scripts directory if it doesn't exist
mkdir -p scripts

# Option 1: Generate from tech documents
echo "Option 1: Generate from tech_docs.txt (40 documents)"
echo "----------------------------------------------"
python scripts/prepare_data.py \
    --documents data/raw/tech_docs.txt \
    --generate_pairs \
    --num_pairs 500 \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed/tech_only

echo ""
echo "✅ Tech data prepared in data/processed/tech_only/"
echo ""

# Option 2: Generate from science documents
echo "Option 2: Generate from science_docs.txt (20 documents)"
echo "----------------------------------------------"
python scripts/prepare_data.py \
    --documents data/raw/science_docs.txt \
    --generate_pairs \
    --num_pairs 250 \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed/science_only

echo ""
echo "✅ Science data prepared in data/processed/science_only/"
echo ""

# Option 3: Combine both
echo "Option 3: Combine tech and science documents (60 total)"
echo "----------------------------------------------"
cat data/raw/tech_docs.txt data/raw/science_docs.txt > data/raw/combined_docs.txt
python scripts/prepare_data.py \
    --documents data/raw/combined_docs.txt \
    --generate_pairs \
    --num_pairs 1000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed/combined

echo ""
echo "✅ Combined data prepared in data/processed/combined/"
echo ""

# Option 4: Use pre-made pairs
echo "Option 4: Use pre-made training pairs (20 examples)"
echo "----------------------------------------------"
python scripts/prepare_data.py \
    --input data/raw/training_pairs.json \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed/premade

echo ""
echo "✅ Pre-made pairs prepared in data/processed/premade/"
echo ""

echo "============================================"
echo "Data Generation Complete!"
echo "============================================"
echo ""
echo "Available datasets:"
echo "  1. data/processed/tech_only/      - 500 tech pairs"
echo "  2. data/processed/science_only/   - 250 science pairs"
echo "  3. data/processed/combined/       - 1000 combined pairs"
echo "  4. data/processed/premade/        - 20 curated pairs"
echo ""
echo "Start training with:"
echo "  python scripts/train/isotropic.py --train_data data/processed/combined/train.json --epochs 3"
