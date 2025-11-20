# API Documentation

## IsotropicGaussianEncoder

Main model class for training and inference.

### **Initialization**

```python
from ragcun.model import IsotropicGaussianEncoder

# Create model
model = IsotropicGaussianEncoder(
    output_dim=512,
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=True
)
```

**Parameters:**
- `output_dim` (int): Dimension of Gaussian embeddings (default: 512)
- `base_model` (str): Base encoder model (default: 'sentence-transformers/all-mpnet-base-v2')
- `freeze_base` (bool): Freeze base encoder weights (default: False)

---

### **Loading Trained Model**

```python
# Load from checkpoint
model = IsotropicGaussianEncoder.from_pretrained('checkpoints/model.pt')

# Move to GPU
model = model.cuda()
```

---

### **Encoding**

```python
# Encode text
embeddings = model.encode([
    "What is machine learning?",
    "Machine learning is a branch of AI"
])

# Returns: numpy array of shape (2, 512)
```

**Parameters:**
- `texts` (List[str]): List of texts to encode
- `batch_size` (int): Batch size for encoding (default: 32)
- `convert_to_numpy` (bool): Return numpy array (default: True)
- `show_progress` (bool): Show progress bar (default: False)

---

### **Computing Similarity**

```python
import numpy as np

query_emb = model.encode(["machine learning query"])
doc_embs = model.encode(["doc 1", "doc 2", "doc 3"])

# Euclidean distance (use negative for similarity score)
distances = np.linalg.norm(query_emb - doc_embs, axis=1)
similarities = -distances  # Higher = more similar

# Best match
best_idx = similarities.argmax()
```

---

### **Batch Processing**

```python
# Large corpus encoding
corpus = ["doc 1", "doc 2", ...]  # 1 million docs

embeddings = model.encode(
    corpus,
    batch_size=64,
    show_progress=True,
    convert_to_numpy=True
)

# Save embeddings
np.save('corpus_embeddings.npy', embeddings)
```

---

## Training

### **Command Line**

```bash
python scripts/train.py \
    --train_data data/processed/msmarco/train.json \
    --val_data data/processed/msmarco/dev.json \
    --base_model sentence-transformers/all-mpnet-base-v2 \
    --freeze_base False \
    --base_learning_rate 1e-5 \
    --projection_learning_rate 5e-4 \
    --lambda_isotropy 1.0 \
    --lambda_reg 0.1 \
    --epochs 3 \
    --batch_size 16 \
    --output_dim 512 \
    --warmup_steps 1000 \
    --mixed_precision \
    --output_dir checkpoints/model
```

### **Programmatic**

```python
from ragcun.model import IsotropicGaussianEncoder
from ragcun.training import train_model

# Initialize model
model = IsotropicGaussianEncoder(
    output_dim=512,
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=False
)

# Train
train_model(
    model=model,
    train_data='data/processed/msmarco/train.json',
    val_data='data/processed/msmarco/dev.json',
    epochs=3,
    batch_size=16,
    learning_rate=1e-5,
    lambda_isotropy=1.0,
    output_dir='checkpoints/model'
)
```

---

## Evaluation

### **BEIR Evaluation**

```bash
python scripts/evaluate_beir.py \
    --model_path checkpoints/model.pt \
    --datasets scifact nfcorpus arguana \
    --output_file results/beir_results.json
```

### **Custom Evaluation**

```python
from ragcun.model import IsotropicGaussianEncoder
import numpy as np

# Load model
model = IsotropicGaussianEncoder.from_pretrained('checkpoints/model.pt')

# Your queries and documents
queries = ["query 1", "query 2"]
documents = ["doc 1", "doc 2", "doc 3"]

# Encode
query_embs = model.encode(queries)
doc_embs = model.encode(documents)

# Compute distances
distances = np.linalg.norm(
    query_embs[:, None, :] - doc_embs[None, :, :],
    axis=2
)

# Get top-k documents for each query
k = 2
for i, query in enumerate(queries):
    top_k_indices = distances[i].argsort()[:k]
    print(f"Query: {query}")
    for idx in top_k_indices:
        print(f"  - {documents[idx]} (distance: {distances[i][idx]:.3f})")
```

---

## Configuration

### **Model Config**

```python
from ragcun.config import ModelConfig

config = ModelConfig(
    output_dim=512,
    base_model='sentence-transformers/all-mpnet-base-v2',
    freeze_base=True,
    dropout=0.1
)

model = IsotropicGaussianEncoder(config)
```

### **Training Config**

```python
from ragcun.config import TrainingConfig

config = TrainingConfig(
    batch_size=16,
    epochs=3,
    learning_rate=1e-5,
    warmup_steps=1000,
    lambda_isotropy=1.0,
    lambda_reg=0.1,
    mixed_precision=True
)
```

---

## Utilities

### **Get Trainable Parameters**

```python
model = IsotropicGaussianEncoder(freeze_base=True)

base_params, proj_params = model.get_trainable_parameters()

print(f"Base params: {sum(p.numel() for p in base_params):,}")
print(f"Projection params: {sum(p.numel() for p in proj_params):,}")
```

### **Save/Load Model**

```python
# Save
model.save_pretrained('checkpoints/my_model')

# Load
model = IsotropicGaussianEncoder.from_pretrained('checkpoints/my_model')
```

### **Export to HuggingFace**

```python
# Save in HuggingFace format
model.push_to_hub('username/model-name')

# Load from HuggingFace
model = IsotropicGaussianEncoder.from_pretrained('username/model-name')
```

---

## Examples

### **Example 1: Simple Retrieval**

```python
from ragcun.model import IsotropicGaussianEncoder
import numpy as np

# Load model
model = IsotropicGaussianEncoder.from_pretrained('checkpoints/model.pt')

# Query and corpus
query = "What is deep learning?"
corpus = [
    "Deep learning uses neural networks",
    "Machine learning is a subset of AI",
    "Python is a programming language"
]

# Encode
query_emb = model.encode([query])
corpus_embs = model.encode(corpus)

# Find most similar
distances = np.linalg.norm(query_emb - corpus_embs, axis=1)
best_match = corpus[distances.argmin()]

print(f"Query: {query}")
print(f"Best match: {best_match}")
```

### **Example 2: Batch Retrieval**

```python
# Multiple queries
queries = [
    "machine learning basics",
    "python programming",
    "neural networks"
]

# Large corpus
corpus = [...]  # 100K documents

# Encode all
query_embs = model.encode(queries)
corpus_embs = model.encode(corpus, batch_size=128, show_progress=True)

# Retrieve top-10 for each query
for i, query in enumerate(queries):
    distances = np.linalg.norm(query_embs[i] - corpus_embs, axis=1)
    top_10 = distances.argsort()[:10]
    
    print(f"\nQuery: {query}")
    for rank, idx in enumerate(top_10, 1):
        print(f"{rank}. {corpus[idx][:50]}...")
```

### **Example 3: Semantic Search**

```python
class SemanticSearch:
    def __init__(self, model_path):
        self.model = IsotropicGaussianEncoder.from_pretrained(model_path)
        self.corpus = []
        self.corpus_embeddings = None
    
    def index(self, documents):
        """Index documents"""
        self.corpus = documents
        self.corpus_embeddings = self.model.encode(
            documents,
            batch_size=64,
            show_progress=True
        )
    
    def search(self, query, top_k=10):
        """Search for query"""
        query_emb = self.model.encode([query])
        distances = np.linalg.norm(
            query_emb - self.corpus_embeddings,
            axis=1
        )
        top_indices = distances.argsort()[:top_k]
        
        return [
            {
                'document': self.corpus[idx],
                'distance': float(distances[idx])
            }
            for idx in top_indices
        ]

# Use
search = SemanticSearch('checkpoints/model.pt')
search.index(your_documents)
results = search.search("your query")
```

---

## Advanced

### **Custom Loss Weights**

```python
# Train with custom isotropy weight
python scripts/train.py \
    --lambda_isotropy 2.0 \  # Stronger isotropy
    --lambda_reg 0.2 \
    [other args]
```

### **Mixed Precision Training**

```python
# Faster training with FP16
python scripts/train.py \
    --mixed_precision \
    [other args]
```

### **Gradient Checkpointing**

```python
# Save memory at cost of speed
python scripts/train.py \
    --gradient_checkpointing \
    [other args]
```

### **Multi-GPU Training**

```bash
# Use PyTorch DDP
torchrun --nproc_per_node=4 scripts/train.py [args]
```

---

## FAQ

**Q: What distance metric to use?**  
A: Euclidean distance. Use negative distance for similarity scores.

**Q: Can I use cosine similarity?**  
A: No, embeddings are unnormalized. Use Euclidean distance.

**Q: How to normalize embeddings?**  
A: Don't! The whole point is unnormalized Gaussian embeddings.

**Q: What's the output dimension?**  
A: Default 512, configurable via `output_dim` parameter.

**Q: Can I use different base models?**  
A: Yes, any SentenceTransformer model works.

---

For more examples, see: [examples/](../examples/)

