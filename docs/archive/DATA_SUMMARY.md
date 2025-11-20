# Generated Training Data Summary

This document describes the training data that has been generated for your RAGCUN project.

## 📊 Available Datasets

### 1. Tech Documents (`data/raw/tech_docs.txt`)
- **Size**: 41 documents (~15KB)
- **Topics**: Programming, AI/ML, DevOps, Cloud, Databases
- **Content**:
  - Programming languages (Python, TypeScript, JavaScript)
  - Machine Learning frameworks (TensorFlow, PyTorch)
  - DevOps tools (Docker, Kubernetes, Git, Terraform, Ansible)
  - Databases (PostgreSQL, MongoDB, Redis, Elasticsearch)
  - Cloud & Architecture (Microservices, Serverless, REST, GraphQL)
  - Development practices (CI/CD, Agile, Scrum, TDD)

**Documents include**:
- Python, Machine Learning, NLP, Deep Learning, Data Science, AI
- Docker, Kubernetes, Git, REST, GraphQL, Microservices
- CI/CD, Agile, Scrum, TDD, APIs, JSON, NoSQL, SQL
- Redis, MongoDB, PostgreSQL, Elasticsearch, Kafka
- TensorFlow, PyTorch, React, Node.js, TypeScript
- WebAssembly, Serverless, Terraform, Ansible, Prometheus

### 2. Science Documents (`data/raw/science_docs.txt`)
- **Size**: 20 documents (~7KB)
- **Topics**: Biology, Physics, Chemistry, Earth Science
- **Content**:
  - Biology: DNA, Evolution, Neurons, Photosynthesis, CRISPR, Vaccines
  - Physics: Quantum Mechanics, General Relativity, Black Holes, Thermodynamics
  - Chemistry: Periodic Table, Antibiotics
  - Earth/Environmental: Climate Change, Plate Tectonics, Ecosystems
  - Technology: Nanotechnology, Renewable Energy, Stem Cells, Nuclear Fusion

**Documents include**:
- Photosynthesis, DNA, Evolution, Neurons, Immune System
- Big Bang, Black Holes, Quantum Mechanics, General Relativity
- Climate Change, Periodic Table, Antibiotics, CRISPR
- Plate Tectonics, Thermodynamics, Ecosystems, Vaccines
- Nuclear Fusion, Stem Cells, Renewable Energy, Nanotechnology

### 3. Pre-made Training Pairs (`data/raw/training_pairs.json`)
- **Size**: 20 curated query-document pairs (~14KB)
- **Format**: JSON with query, positive, and negative examples
- **Quality**: Hand-crafted for optimal training
- **Topics**: Mix of tech and science

**Example pair**:
```json
{
  "query": "What is Python used for?",
  "positive": "Python is a high-level, interpreted programming language...",
  "negative": "Java is an object-oriented programming language..."
}
```

### 4. Sample Documents (`data/raw/sample_docs.txt`)
- **Size**: 10 documents (~2KB)
- **Topics**: General tech topics
- **Purpose**: Quick testing and demos

---

## 🚀 Quick Start Usage

### Option 1: Generate from Tech Documents (Recommended for ML/AI training)

```bash
python scripts/prepare_data.py \
    --documents data/raw/tech_docs.txt \
    --generate_pairs \
    --num_pairs 500 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**Result**: 500 training pairs
- 400 training examples
- 50 validation examples
- 50 test examples

### Option 2: Generate from Science Documents

```bash
python scripts/prepare_data.py \
    --documents data/raw/science_docs.txt \
    --generate_pairs \
    --num_pairs 250 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**Result**: 250 training pairs
- 200 training examples
- 25 validation examples
- 25 test examples

### Option 3: Combine Everything (Maximum Data)

```bash
# Combine all documents
cat data/raw/tech_docs.txt data/raw/science_docs.txt > data/raw/combined_docs.txt

# Generate pairs from combined set
python scripts/prepare_data.py \
    --documents data/raw/combined_docs.txt \
    --generate_pairs \
    --num_pairs 1000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

**Result**: 1000 training pairs from 61 documents
- 800 training examples
- 100 validation examples
- 100 test examples

### Option 4: Use Pre-made Pairs (Highest Quality)

```bash
python scripts/prepare_data.py \
    --input data/raw/training_pairs.json \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed
```

**Result**: 20 curated pairs
- 14 training examples
- 3 validation examples
- 3 test examples

**Note**: Small dataset - good for testing, but too small for real training

### Option 5: Use All Data (Automated Script)

```bash
./scripts/generate_training_data.sh
```

This script generates all 4 datasets automatically!

---

## 📈 Training Recommendations

### For Quick Testing (2-5 minutes)
```bash
# Use pre-made pairs or small sample
python scripts/prepare_data.py \
    --documents data/raw/sample_docs.txt \
    --generate_pairs \
    --num_pairs 50 \
    --split 0.7 0.15 0.15 \
    --output_dir data/processed

python scripts/train.py \
    --train_data data/processed/train.json \
    --epochs 1 \
    --batch_size 8
```

### For Real Training (1-2 hours on T4 GPU)
```bash
# Use combined dataset
cat data/raw/tech_docs.txt data/raw/science_docs.txt > data/raw/combined_docs.txt

python scripts/prepare_data.py \
    --documents data/raw/combined_docs.txt \
    --generate_pairs \
    --num_pairs 1000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed

python scripts/train.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 3 \
    --batch_size 8 \
    --freeze_early_layers \
    --output_dir checkpoints
```

### For Production Training (Add Your Own Data)
```bash
# Add your documents to data/raw/
cp your_documents.txt data/raw/

# Combine with existing data
cat data/raw/tech_docs.txt data/raw/science_docs.txt data/raw/your_documents.txt > data/raw/all_docs.txt

python scripts/prepare_data.py \
    --documents data/raw/all_docs.txt \
    --generate_pairs \
    --num_pairs 5000 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed

python scripts/train.py \
    --train_data data/processed/train.json \
    --val_data data/processed/val.json \
    --epochs 5 \
    --batch_size 8
```

---

## 📊 Data Statistics

### Document Coverage by Topic

**Technology (41 docs)**:
- Programming Languages: 4
- AI/ML Frameworks: 4
- DevOps Tools: 6
- Databases: 5
- Cloud/Architecture: 6
- Development Practices: 6
- Web Technologies: 5
- Infrastructure: 5

**Science (20 docs)**:
- Biology: 8
- Physics: 5
- Chemistry: 2
- Earth Science: 3
- Technology Applications: 2

**Total**: 61 unique documents covering diverse topics

### Query-Pair Quality

Pre-made pairs include:
- Clear, specific questions
- Relevant positive answers
- Realistic negative answers (similar topic but wrong answer)
- Diverse question formats (What, How, Why, etc.)

---

## 🎯 Data Quality Tips

### Good Query-Document Pairs
✅ Query is a natural question
✅ Positive document directly answers the query
✅ Negative document is on a similar topic but doesn't answer the query
✅ Documents have sufficient detail (50-200 words)

### Poor Query-Document Pairs
❌ Query is too vague or general
❌ Positive document doesn't answer the query
❌ Negative document is unrelated (too easy to distinguish)
❌ Documents are too short (<30 words) or too long (>500 words)

---

## 🔧 Customizing Your Data

### Add Your Own Documents

1. Create a text file with one document per line:
```bash
echo "Document 1 text here" > data/raw/my_docs.txt
echo "Document 2 text here" >> data/raw/my_docs.txt
```

2. Generate training pairs:
```bash
python scripts/prepare_data.py \
    --documents data/raw/my_docs.txt \
    --generate_pairs \
    --num_pairs 100 \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

### Create Custom Pairs

Create a JSON file with this format:
```json
[
  {
    "query": "Your question here?",
    "positive": "The correct answer document...",
    "negative": "A similar but incorrect document..."
  }
]
```

Then split it:
```bash
python scripts/prepare_data.py \
    --input data/raw/my_pairs.json \
    --split 0.8 0.1 0.1 \
    --output_dir data/processed
```

---

## 📦 Generated Files

After running `prepare_data.py`, you'll have:

```
data/processed/
├── train.json          # Training pairs
├── val.json           # Validation pairs
├── test.json          # Test pairs (same format as train)
└── test_eval.json     # Test data in evaluation format
```

### File Formats

**Training format** (`train.json`, `val.json`, `test.json`):
```json
[
  {
    "query": "What is Python?",
    "positive": "Python is a programming language...",
    "negative": "Java is a programming language..."
  }
]
```

**Evaluation format** (`test_eval.json`):
```json
{
  "corpus": ["doc1", "doc2", "doc3"],
  "queries": ["query1", "query2"],
  "relevance": [[0], [1, 2]]
}
```

---

## 🎓 Next Steps

1. **Generate your training data**:
   ```bash
   python scripts/prepare_data.py --documents data/raw/tech_docs.txt --generate_pairs --num_pairs 500 --split 0.8 0.1 0.1 --output_dir data/processed
   ```

2. **Train your model**:
   ```bash
   python scripts/train.py --train_data data/processed/train.json --val_data data/processed/val.json --epochs 3
   ```

3. **Evaluate**:
   ```bash
   python scripts/evaluate.py --model_path checkpoints/best_model.pt --test_data data/processed/test_eval.json
   ```

4. **Use in production**:
   ```python
   from ragcun import IsotropicRetriever
   retriever = IsotropicRetriever('checkpoints/best_model.pt')
   retriever.add_documents(your_documents)
   results = retriever.retrieve("your query", top_k=5)
   ```

---

## 📚 Additional Resources

- **Training Guide**: See `TRAINING_GUIDE.md`
- **Scripts Reference**: See `SCRIPTS_README.md`
- **Directory Guide**: See `DIRECTORY_GUIDE.md`
- **Project README**: See `README.md`

Happy training! 🚀
