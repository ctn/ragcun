# RAGCUN - Retrieval-Augmented Generation Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ctn/ragcun/blob/main/notebooks/colab_quickstart.ipynb)

A simple and effective framework for building Retrieval-Augmented Generation (RAG) applications. RAGCUN provides an easy-to-use interface for combining document retrieval with language generation.

## 🚀 Quick Start with Google Colab

The easiest way to get started is using Google Colab:

1. Click the "Open in Colab" badge above
2. Run the cells to install and try RAGCUN
3. Start building your own RAG applications!

## 📋 Features

- **Simple API**: Easy-to-use interface for RAG pipelines
- **Modular Design**: Swap out retrievers and generators as needed
- **Google Colab Ready**: Pre-configured notebooks for quick experimentation
- **Extensible**: Build custom components for your specific use case
- **Well-Documented**: Examples and tutorials to get you started

## 🔧 Installation

### In Google Colab

```python
# Clone and install
!git clone https://github.com/ctn/ragcun.git
%cd ragcun
!pip install -e .
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/ctn/ragcun.git
cd ragcun

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[full]"
```

## 💡 Usage

### Basic Example

```python
from ragcun import RAGPipeline

# Create a pipeline
pipeline = RAGPipeline()

# Add documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language."
]
pipeline.add_documents(documents)

# Query the pipeline
response = pipeline.query("What is machine learning?", top_k=3)
print(response)
```

### Custom Configuration

```python
from ragcun import RAGPipeline, Retriever, Generator

# Create custom components
retriever = Retriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
generator = Generator(model_name="gpt-3.5-turbo")

# Build pipeline with custom components
pipeline = RAGPipeline(retriever=retriever, generator=generator)
```

## 📂 Project Structure

```
ragcun/
├── src/ragcun/          # Main package source code
│   ├── __init__.py      # Package initialization
│   ├── retriever.py     # Document retrieval module
│   ├── generator.py     # Text generation module
│   └── pipeline.py      # RAG pipeline orchestration
├── data/                # Data storage
│   ├── raw/            # Raw input documents
│   ├── processed/      # Processed documents
│   └── embeddings/     # Vector embeddings
├── notebooks/           # Jupyter/Colab notebooks
│   └── colab_quickstart.ipynb
├── examples/            # Example scripts
│   └── basic_example.py
├── config/              # Configuration files
│   ├── __init__.py
│   └── config.example.env
├── requirements.txt     # Package dependencies
├── setup.py            # Package setup configuration
└── README.md           # This file
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
