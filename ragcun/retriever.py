"""
GaussianRetriever - Document retrieval using isotropic Gaussian embeddings.

Uses Euclidean distance (L2) instead of cosine similarity.
"""

from typing import List, Optional, Tuple
import torch
import numpy as np
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .model import IsotropicGaussianEncoder


class GaussianRetriever:
    """
    Document retriever using isotropic Gaussian embeddings.

    Key differences from traditional retrievers:
    - Uses Euclidean distance (L2) instead of cosine similarity
    - Embeddings are NOT normalized (magnitude = confidence)
    - Better separation between relevant/irrelevant documents

    Args:
        model_path: Path to trained IsotropicGaussianEncoder weights
        embedding_dim: Dimension of embeddings (default: 512)
        use_gpu: Whether to use GPU for FAISS index

    Example:
        >>> retriever = GaussianRetriever(model_path='model.pt')
        >>> retriever.add_documents(["Python is great", "ML is cool"])
        >>> results = retriever.retrieve("programming language", top_k=1)
        >>> print(results)
        [("Python is great", 0.523)]  # (document, euclidean_distance)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 512,
        use_gpu: bool = False
    ):
        """Initialize retriever with model."""

        if model_path:
            # Load trained model
            self.model = IsotropicGaussianEncoder.from_pretrained(
                model_path,
                output_dim=embedding_dim
            )
        else:
            # Use untrained model (not recommended for production)
            print("⚠️  No model_path provided - using untrained model")
            print("   Train a model first using notebooks/lejepa_training.ipynb")
            self.model = IsotropicGaussianEncoder(output_dim=embedding_dim)

        self.model.eval()
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = None
        self.index = None
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if not HAS_FAISS:
            print("⚠️  FAISS not installed - using slower numpy search")
            print("   Install: pip install faiss-cpu (or faiss-gpu)")

    def add_documents(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Add documents to the retriever.

        Args:
            documents: List of document strings
            batch_size: Batch size for encoding
        """
        if not documents:
            return

        # Encode documents
        print(f"Encoding {len(documents)} documents...")
        with torch.no_grad():
            new_embeddings = self.model.encode(
                documents,
                batch_size=batch_size,
                show_progress=True,
                convert_to_numpy=True
            )

        # Add to collection
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Rebuild index
        self._build_index()
        print(f"✅ Total documents: {len(self.documents)}")

    def _build_index(self) -> None:
        """Build FAISS index for fast L2 search."""
        if self.embeddings is None:
            return

        if HAS_FAISS:
            # FAISS L2 (Euclidean distance) index
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index_cpu = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)

            self.index.add(self.embeddings.astype('float32'))
        else:
            # Fallback to numpy
            self.index = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar documents.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (document, distance) tuples, sorted by distance (lower=better)
        """
        if not self.documents or top_k <= 0:
            return []

        # Encode query
        with torch.no_grad():
            query_emb = self.model.encode(
                [query],
                convert_to_numpy=True
            )

        # Search
        if HAS_FAISS and self.index is not None:
            # FAISS search
            distances, indices = self.index.search(
                query_emb.astype('float32'),
                min(top_k, len(self.documents))
            )
            distances = distances[0]
            indices = indices[0]
        else:
            # Numpy fallback
            distances = np.linalg.norm(
                self.embeddings - query_emb,
                axis=1
            )
            indices = np.argsort(distances)[:top_k]
            distances = distances[indices]

        # Return (document, distance) pairs
        results = [
            (self.documents[idx], float(dist))
            for idx, dist in zip(indices, distances)
        ]

        return results

    def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = None
        self.index = None

    def save_index(self, path: str) -> None:
        """Save documents and embeddings to disk."""
        import pickle

        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"✅ Saved index to {path}")

    def load_index(self, path: str) -> None:
        """Load documents and embeddings from disk."""
        import pickle

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self._build_index()

        print(f"✅ Loaded {len(self.documents)} documents from {path}")
