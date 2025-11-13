"""
Basic example of using RAGCUN for a simple Q&A system.

This example demonstrates how to:
1. Create a RAG pipeline
2. Add documents
3. Query the system
"""

from ragcun import RAGPipeline


def main():
    """Run a basic RAG example."""
    
    # Initialize the pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Sample documents about artificial intelligence
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process information.",
        "Natural Language Processing (NLP) helps computers understand human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning teaches agents to make decisions through trial and error.",
        "Transfer learning allows models to apply knowledge from one task to another.",
        "Generative AI can create new content like text, images, and music.",
    ]
    
    # Add documents to the pipeline
    print(f"Adding {len(documents)} documents...")
    pipeline.add_documents(documents)
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "Tell me about machine learning",
        "How does deep learning work?",
    ]
    
    # Query the pipeline
    print("\n" + "="*60)
    print("Running queries...")
    print("="*60 + "\n")
    
    for query in queries:
        print(f"Q: {query}")
        response = pipeline.query(query, top_k=3)
        print(f"A: {response}\n")
        print("-"*60 + "\n")


if __name__ == "__main__":
    main()
