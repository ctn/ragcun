"""
RAGCUN - Retrieval-Augmented Generation framework
A simple and effective framework for building RAG applications.
"""

__version__ = "0.1.0"
__author__ = "RAGCUN Team"

from .retriever import Retriever
from .generator import Generator
from .pipeline import RAGPipeline

__all__ = ["Retriever", "Generator", "RAGPipeline"]
