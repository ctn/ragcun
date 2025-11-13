# LeJEPA Isotropic Gaussian Embeddings for RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ctn/ragcun/blob/main/notebooks/lejepa_training.ipynb)

**Train and use isotropic Gaussian embeddings with LeJEPA for superior RAG retrieval.**

This project implements **isotropic Gaussian distributed embeddings** using Google's EmbeddingGemma-300M fine-tuned with LeJEPA's SIGReg loss. Unlike traditional normalized embeddings (spherical distribution), isotropic Gaussian embeddings provide:

- ✅ **Better retrieval accuracy** - Larger separation between relevant/irrelevant docs
- ✅ **Magnitude = Confidence** - Embedding norm indicates uncertainty
- ✅ **Full dimensional usage** - No dimensional collapse
- ✅ **Semantic compositionality** - Query arithmetic works naturally
- ✅ **Probabilistic scores** - Proper Gaussian likelihood for ranking

## 🚀 Quick Start with Google Colab

Train your isotropic Gaussian embedding model in 3 steps:

1. **[Open Training Notebook](notebooks/lejepa_training.ipynb)** in Google Colab
2. **Run all cells** - Training takes 1-3 hours on free T4 GPU
3. **Download model** - Use in your RAG retrieval system

## 📋 Key Features

- **Isotropic Gaussian Embeddings**: True N(0,I) distribution via LeJEPA SIGReg
- **State-of-the-art Base**: EmbeddingGemma-300M (best MTEB for size)
- **Euclidean Retrieval**: L2 distance instead of cosine similarity
- **No Normalization**: Preserves magnitude as confidence signal
- **Fully Trainable**: Fine-tune on your domain data

## 🔧 Installation

### For Training (Google Colab)

```python
# Install dependencies in Colab
!pip install transformers sentence-transformers datasets faiss-gpu accelerate
!pip install lejepa || pip install git+https://github.com/rbalestr-lab/lejepa.git
```

### For Inference (Local)

```bash
# Clone repository
git clone https://github.com/ctn/ragcun.git
cd ragcun

# Install package
pip install -e .
```

## 💡 Usage

### Step 1: Train Model (in Colab)

```python
# See notebooks/lejepa_training.ipynb for full training code
# After training, download the model weights
```

### Step 2: Use for Retrieval (Local)

```python
from ragcun import GaussianRetriever

# Load your trained model
retriever = GaussianRetriever(model_path='data/embeddings/gaussian_embeddinggemma_final.pt')

# Add documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language."
]
retriever.add_documents(documents)

# Retrieve with Euclidean distance (NOT cosine)
results = retriever.retrieve("What is machine learning?", top_k=3)

for doc, distance in results:
    print(f"[dist={distance:.3f}] {doc}")
```

### Why Euclidean Distance?

```python
# Traditional (spherical embeddings)
cosine_sim(query, doc1) = 0.78  # Good match
cosine_sim(query, doc2) = 0.71  # Bad match
# Difference: only 0.07!

# Isotropic Gaussian embeddings
euclidean_dist(query, doc1) = 0.5   # Good match
euclidean_dist(query, doc2) = 4.2   # Bad match
# Difference: 8.4x larger separation!
```

## 📂 Project Structure

```
ragcun/
├── src/ragcun/                              # Main package
│   ├── __init__.py
│   ├── model.py                             # GaussianEmbeddingGemma model
│   └── retriever.py                         # Gaussian retriever (L2 distance)
├── notebooks/                               # Training & experiments
│   ├── lejepa_training.ipynb               # 🚀 Main training notebook
│   └── document_processing.ipynb
├── data/
│   ├── embeddings/                          # Trained model weights
│   │   └── gaussian_embeddinggemma_final.pt # Put trained model here
│   ├── raw/                                 # Your documents
│   └── processed/                           # Preprocessed data
├── examples/                                # Usage examples
│   └── retrieval_example.py
├── requirements.txt                         # Dependencies
└── README.md
```

## 🎓 Examples

### Running the Basic Example

```bash
python examples/basic_example.py
```

### Using the Colab Notebook

1. Open the [Colab Quickstart Notebook](https://colab.research.google.com/github/ctn/ragcun/blob/main/notebooks/colab_quickstart.ipynb)
2. Follow the step-by-step instructions
3. Experiment with your own documents and queries

## 📊 Working with Data

### Uploading Data in Google Colab

```python
# Method 1: Upload files
from google.colab import files
uploaded = files.upload()

# Method 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Method 3: Download from URL
!wget -O data/raw/document.pdf https://example.com/document.pdf
```

### Local Data Organization

Place your documents in the appropriate directories:
- `data/raw/` - Original documents (PDF, TXT, DOCX, etc.)
- `data/processed/` - Cleaned and processed documents
- `data/embeddings/` - Generated vector embeddings

## ⚙️ Configuration

Copy the example configuration file and customize:

```bash
cp config/config.example.env .env
```

Edit `.env` with your settings:
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MODEL_NAME=gpt-3.5-turbo
TOP_K=5
```

## 🔌 API Reference

### RAGPipeline

Main class for orchestrating retrieval and generation.

```python
pipeline = RAGPipeline(retriever=None, generator=None)
pipeline.add_documents(documents: List[str])
response = pipeline.query(question: str, top_k: int = 5)
```

### Retriever

Handles document retrieval and similarity search.

```python
retriever = Retriever(embedding_model: str = None)
retriever.add_documents(documents: List[str])
docs = retriever.retrieve(query: str, top_k: int = 5)
```

### Generator

Generates responses based on retrieved context.

```python
generator = Generator(model_name: str = None)
response = generator.generate(query: str, context: List[str])
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for easy experimentation with RAG architectures
- Designed with Google Colab compatibility in mind
- Inspired by modern NLP and retrieval systems

## 📚 Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Vector Search](https://github.com/facebookresearch/faiss)

## 🐛 Issues

If you encounter any problems, please [open an issue](https://github.com/ctn/ragcun/issues) on GitHub.
