"""
Example: Using Gaussian Embeddings for Retrieval

This example shows how to use trained LeJEPA isotropic Gaussian embeddings
for document retrieval with Euclidean distance.
"""

from ragcun import IsotropicRetriever


def main():
    print("=" * 60)
    print("LeJEPA Gaussian Embeddings - Retrieval Example")
    print("=" * 60)

    # Initialize retriever with trained model
    # NOTE: You need to train the model first using notebooks/lejepa_training.ipynb
    model_path = 'data/embeddings/gaussian_embeddinggemma_final.pt'

    print("\n1. Loading model...")
    try:
        retriever = IsotropicRetriever(model_path=model_path)
    except FileNotFoundError:
        print(f"\n⚠️  Model not found at {model_path}")
        print("   Train a model first:")
        print("   1. Open notebooks/lejepa_training.ipynb in Google Colab")
        print("   2. Run all cells to train")
        print("   3. Download and save to data/embeddings/")
        print("\n   For now, using untrained model for demo...")
        retriever = IsotropicRetriever()

    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Retrieval-Augmented Generation combines retrieval with language generation.",
        "JavaScript is primarily used for web development.",
        "Data structures organize and store data efficiently.",
        "Algorithms are step-by-step procedures for solving problems.",
        "Cloud computing provides on-demand computing resources.",
        "DevOps combines software development and IT operations."
    ]

    print(f"\n2. Adding {len(documents)} documents...")
    retriever.add_documents(documents)

    # Example queries
    queries = [
        "What is machine learning?",
        "Programming languages",
        "How does NLP work?"
    ]

    print("\n3. Testing retrieval (Euclidean distance)...")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 60)

        # Retrieve top 3 documents
        results = retriever.retrieve(query, top_k=3)

        for i, (doc, distance) in enumerate(results, 1):
            print(f"\n{i}. [distance={distance:.3f}]")
            print(f"   {doc}")

    print("\n" + "=" * 60)
    print("Why Euclidean Distance?")
    print("=" * 60)
    print("""
Traditional embeddings (normalized, cosine similarity):
  - cosine_sim(query, good_match) = 0.78
  - cosine_sim(query, bad_match) = 0.71
  - Difference: only 0.07 (hard to distinguish!)

Isotropic Gaussian embeddings (unnormalized, L2 distance):
  - euclidean_dist(query, good_match) = 0.5
  - euclidean_dist(query, bad_match) = 4.2
  - Difference: 8.4x larger separation!

Additional benefits:
  ✓ Magnitude indicates confidence
  ✓ No dimensional collapse
  ✓ Better semantic compositionality
  ✓ Proper probabilistic scores
""")

    # Optional: Save index for later use
    print("\n4. Saving index...")
    retriever.save_index('data/processed/retriever_index.pkl')

    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
