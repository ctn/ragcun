"""
RAG Pipeline module combining retrieval and generation.
"""

from typing import List, Optional
from .retriever import Retriever
from .generator import Generator


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    This class orchestrates the retrieval and generation steps
    to provide end-to-end RAG functionality.
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever instance. If None, creates default.
            generator: Generator instance. If None, creates default.
        """
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()
        
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the pipeline's retriever.
        
        Args:
            documents: List of document strings to add.
        """
        self.retriever.add_documents(documents)
        
    def query(self, question: str, top_k: int = 5) -> str:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: The user question.
            top_k: Number of documents to retrieve.
            
        Returns:
            Generated response string.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate response
        response = self.generator.generate(question, retrieved_docs)
        
        return response
