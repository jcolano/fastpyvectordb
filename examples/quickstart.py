#!/usr/bin/env python3
"""
FastPyVectorDB Quickstart Guide

This example demonstrates how to use FastPyVectorDB for semantic search and
document retrieval, similar to ChromaDB.

Installation:
    pip install fastpyvectordb[local]  # For local embeddings (no API key needed)
    # or
    pip install fastpyvectordb[openai]  # For OpenAI embeddings

Quick Install from source:
    pip install -e .  # From the repository root
    pip install sentence-transformers  # For local embeddings
"""

import sys
from pathlib import Path

# Add parent directory to path (for running from examples/)
sys.path.insert(0, str(Path(__file__).parent.parent))

import fastpyvectordb


def basic_usage():
    """Basic usage - create collection, add documents, search."""
    print("=" * 60)
    print("BASIC USAGE")
    print("=" * 60)

    # Create a client (data stored in ./vectordb by default)
    client = fastpyvectordb.Client(path="./quickstart_db")

    # Create a collection
    # Uses sentence-transformers by default (no API key needed)
    collection = client.get_or_create_collection("documents")

    # Add some documents
    collection.add(
        documents=[
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with many layers",
            "Natural language processing helps computers understand text",
            "Computer vision enables machines to interpret images",
            "Reinforcement learning trains agents through rewards",
        ],
        ids=["ml", "dl", "nlp", "cv", "rl"],
        metadatas=[
            {"category": "AI", "difficulty": "beginner"},
            {"category": "AI", "difficulty": "advanced"},
            {"category": "NLP", "difficulty": "intermediate"},
            {"category": "CV", "difficulty": "intermediate"},
            {"category": "AI", "difficulty": "advanced"},
        ]
    )

    print(f"Added {collection.count} documents")

    # Search for similar documents
    results = collection.query(
        query_texts="What is artificial intelligence?",
        n_results=3
    )

    print("\nSearch results for 'What is artificial intelligence?':")
    for i, (doc, score) in enumerate(zip(results.documents[0], results.distances[0])):
        print(f"  {i+1}. (score: {score:.4f}) {doc}")

    # Clean up
    client.persist()
    print("\nData persisted to disk")


def filtering_example():
    """Demonstrate metadata filtering."""
    print("\n" + "=" * 60)
    print("FILTERING EXAMPLE")
    print("=" * 60)

    client = fastpyvectordb.Client(path="./quickstart_db")
    collection = client.get_collection("documents")

    # Search with filter
    results = collection.query(
        query_texts="advanced machine learning",
        n_results=5,
        where={"difficulty": "advanced"}  # Only return advanced docs
    )

    print("Search with filter (difficulty=advanced):")
    for doc, meta in zip(results.documents[0], results.metadatas[0]):
        print(f"  - {doc} [{meta}]")


def crud_operations():
    """Demonstrate CRUD operations."""
    print("\n" + "=" * 60)
    print("CRUD OPERATIONS")
    print("=" * 60)

    client = fastpyvectordb.Client(path="./quickstart_db")
    collection = client.get_or_create_collection("crud_demo")

    # Create (Add)
    print("Adding documents...")
    collection.add(
        documents=["First document", "Second document", "Third document"],
        ids=["doc1", "doc2", "doc3"],
        metadatas=[{"version": 1}, {"version": 1}, {"version": 1}]
    )
    print(f"  Count: {collection.count}")

    # Read (Get)
    print("\nGetting document by ID...")
    result = collection.get(ids=["doc1"])
    print(f"  doc1: {result.documents[0]}")

    # Update
    print("\nUpdating document...")
    collection.update(
        ids=["doc1"],
        documents=["First document (updated)"],
        metadatas=[{"version": 2}]
    )
    result = collection.get(ids=["doc1"])
    print(f"  doc1: {result.documents[0]} (version: {result.metadatas[0]['version']})")

    # Delete
    print("\nDeleting document...")
    deleted = collection.delete(ids=["doc3"])
    print(f"  Deleted: {deleted} documents")
    print(f"  Remaining count: {collection.count}")

    # Upsert (Add or Update)
    print("\nUpserting documents...")
    collection.upsert(
        documents=["Doc1 via upsert", "New doc via upsert"],
        ids=["doc1", "doc4"],
        metadatas=[{"version": 3}, {"version": 1}]
    )
    print(f"  Count after upsert: {collection.count}")

    client.persist()


def batch_operations():
    """Demonstrate batch operations for better performance."""
    print("\n" + "=" * 60)
    print("BATCH OPERATIONS")
    print("=" * 60)

    import time

    client = fastpyvectordb.Client(path="./quickstart_db")
    collection = client.get_or_create_collection("batch_demo")

    # Generate sample data
    n_docs = 100
    documents = [f"Document number {i} with some content about topic {i % 10}" for i in range(n_docs)]
    ids = [f"batch_doc_{i}" for i in range(n_docs)]
    metadatas = [{"topic": i % 10, "batch": True} for i in range(n_docs)]

    # Batch add
    start = time.time()
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    elapsed = time.time() - start
    print(f"Added {n_docs} documents in {elapsed:.2f}s ({n_docs/elapsed:.0f} docs/sec)")

    # Batch query
    queries = ["topic 0", "topic 5", "document about something"]
    start = time.time()
    results = collection.query(query_texts=queries, n_results=5)
    elapsed = time.time() - start
    print(f"Queried {len(queries)} texts in {elapsed*1000:.1f}ms")

    for i, query in enumerate(queries):
        print(f"\n  Query: '{query}'")
        for doc in results.documents[i][:2]:
            print(f"    - {doc[:50]}...")

    client.persist()


def multiple_collections():
    """Demonstrate working with multiple collections."""
    print("\n" + "=" * 60)
    print("MULTIPLE COLLECTIONS")
    print("=" * 60)

    client = fastpyvectordb.Client(path="./quickstart_db")

    # Create collections for different types of content
    articles = client.get_or_create_collection("articles")
    products = client.get_or_create_collection("products")

    # Add data to each
    articles.add(
        documents=["Python tutorial for beginners", "Advanced async programming"],
        ids=["art1", "art2"]
    )

    products.add(
        documents=["Laptop with 16GB RAM", "Wireless mouse"],
        ids=["prod1", "prod2"]
    )

    print(f"Collections: {client.list_collections()}")
    print(f"Articles count: {articles.count}")
    print(f"Products count: {products.count}")

    # Search each collection
    query = "computer peripherals"
    print(f"\nSearching for '{query}':")

    art_results = articles.query(query, n_results=1)
    prod_results = products.query(query, n_results=1)

    print(f"  Articles: {art_results.documents[0][0] if art_results.documents[0] else 'No match'}")
    print(f"  Products: {prod_results.documents[0][0] if prod_results.documents[0] else 'No match'}")

    client.persist()


def using_openai_embeddings():
    """Example using OpenAI embeddings (requires OPENAI_API_KEY)."""
    print("\n" + "=" * 60)
    print("OPENAI EMBEDDINGS (requires API key)")
    print("=" * 60)

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("  Skipped: Set OPENAI_API_KEY to run this example")
        return

    client = fastpyvectordb.Client(path="./quickstart_db")

    # Create collection with OpenAI embeddings
    collection = client.create_collection(
        name="openai_docs",
        embedding_model="text-embedding-3-small",
        embedding_provider="openai"
    )

    collection.add(
        documents=["OpenAI makes powerful AI models", "GPT-4 is a large language model"],
        ids=["oai1", "oai2"]
    )

    results = collection.query("What AI models does OpenAI make?", n_results=2)
    print("Results:", results.documents[0])

    client.persist()


def peek_and_get_all():
    """Show how to peek at data and get all documents."""
    print("\n" + "=" * 60)
    print("PEEK AND GET ALL")
    print("=" * 60)

    client = fastpyvectordb.Client(path="./quickstart_db")
    collection = client.get_collection("documents")

    # Peek at first few documents
    sample = collection.peek(limit=3)
    print(f"Peeking at first 3 documents:")
    for doc, meta in zip(sample.documents, sample.metadatas):
        print(f"  - {doc} {meta}")

    # Get all documents
    all_docs = collection.get()
    print(f"\nTotal documents: {len(all_docs.ids)}")

    # Get with filter
    filtered = collection.get(where={"category": "AI"})
    print(f"Documents in 'AI' category: {len(filtered.ids)}")


def cleanup():
    """Clean up example data."""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    import shutil
    if Path("./quickstart_db").exists():
        shutil.rmtree("./quickstart_db")
        print("Cleaned up quickstart_db")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FastPyVectorDB Quickstart Examples")
    print("=" * 60 + "\n")

    try:
        basic_usage()
        filtering_example()
        crud_operations()
        batch_operations()
        multiple_collections()
        using_openai_embeddings()
        peek_and_get_all()
    finally:
        cleanup()

    print("\n" + "=" * 60)
    print("  All examples completed!")
    print("=" * 60)
