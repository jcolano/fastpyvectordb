"""
PyVectorDB - A High-Performance Python Vector Database

A feature-rich, easy-to-use vector database for semantic search, RAG applications,
and embedding-based retrieval. Similar to ChromaDB but with additional features
like graph support, hybrid search, and quantization.

Quick Start:
    import pyvectordb

    # Create a client
    client = pyvectordb.Client()

    # Create a collection (uses local embedding model by default)
    collection = client.create_collection("my_docs")

    # Add documents
    collection.add(
        documents=["Hello world", "Machine learning is great"],
        ids=["doc1", "doc2"]
    )

    # Search
    results = collection.query("AI and ML", n_results=5)
    print(results.documents[0])

Features:
    - Simple ChromaDB-like API
    - Multiple embedding providers (local, OpenAI, Cohere)
    - HNSW indexing for fast similarity search
    - Metadata filtering
    - Hybrid search (vector + keyword)
    - Graph database integration
    - Quantization for memory efficiency
    - REST API server
"""

__version__ = "0.1.0"
__author__ = "PyVectorDB Contributors"

# Main client API
from pyvectordb.client import Client, Collection, QueryResult, GetResult

# Re-export commonly used items from core modules
import sys as _sys
from pathlib import Path as _Path

# Add parent to path for sibling imports
_parent = _Path(__file__).parent.parent
if str(_parent) not in _sys.path:
    _sys.path.insert(0, str(_parent))

# Core components (advanced usage)
from vectordb_optimized import (
    VectorDB,
    Collection as BaseCollection,
    SearchResult,
    Filter,
    DistanceMetric,
    CollectionConfig
)

# Embedding providers
from embeddings import (
    Embedder,
    get_embedder,
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    CohereEmbedder,
    MockEmbedder,
    CachedEmbedder,
    EmbeddingCollection
)

# Define public API
__all__ = [
    # Version
    "__version__",

    # Main API (most users need only these)
    "Client",
    "Collection",
    "QueryResult",
    "GetResult",

    # Core (advanced usage)
    "VectorDB",
    "BaseCollection",
    "SearchResult",
    "Filter",
    "DistanceMetric",
    "CollectionConfig",

    # Embeddings
    "Embedder",
    "get_embedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "MockEmbedder",
    "CachedEmbedder",
    "EmbeddingCollection",

    # Convenience function
    "create_client",
]


def create_client(
    path: str = "./vectordb",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "auto"
) -> Client:
    """
    Convenience function to create a PyVectorDB client.

    Args:
        path: Directory to store database files
        embedding_model: Default embedding model
        embedding_provider: Embedding provider ("auto", "sentence-transformers", "openai", "mock")

    Returns:
        Client instance

    Example:
        client = pyvectordb.create_client("./my_data")
    """
    return Client(
        path=path,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider
    )
