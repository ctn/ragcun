"""
LeJEPA Isotropic Gaussian Embeddings for RAG

Train and use isotropic Gaussian embeddings with LeJEPA for superior retrieval.
"""

__version__ = "0.2.0"
__author__ = "RAGCUN Team"

from .model import IsotropicGaussianEncoder
from .retriever import GaussianRetriever

__all__ = ["IsotropicGaussianEncoder", "GaussianRetriever"]
