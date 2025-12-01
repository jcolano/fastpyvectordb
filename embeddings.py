"""
Embeddings Integration Module

Provides embedding generation from various providers:
- OpenAI (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
- Sentence Transformers (local models)
- Cohere
- Custom/Mock embeddings for testing

Usage:
    from embeddings import get_embedder, OpenAIEmbedder, SentenceTransformerEmbedder

    # OpenAI
    embedder = OpenAIEmbedder(api_key="sk-...")
    vector = embedder.embed("Hello world")
    vectors = embedder.embed_batch(["Hello", "World"])

    # Sentence Transformers (local)
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vector = embedder.embed("Hello world")

    # Auto-detect from environment
    embedder = get_embedder()  # Uses OPENAI_API_KEY or falls back to local
"""

import os
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np


# =============================================================================
# Base Embedder Interface
# =============================================================================

@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    vector: np.ndarray
    model: str
    dimensions: int
    tokens_used: int = 0


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        pass

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts.

        Default implementation calls embed() for each text.
        Subclasses should override for batch optimization.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return np.array(embeddings)

    def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """Embed with full metadata."""
        vector = self.embed(text)
        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            dimensions=self.dimensions
        )


# =============================================================================
# OpenAI Embedder
# =============================================================================

class OpenAIEmbedder(Embedder):
    """
    OpenAI embeddings via API.

    Models:
    - text-embedding-3-small: 1536 dimensions (default, good balance)
    - text-embedding-3-large: 3072 dimensions (highest quality)
    - text-embedding-ada-002: 1536 dimensions (legacy)

    Requires: pip install openai
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str = None,
        model: str = "text-embedding-3-small",
        dimensions: int = None  # For dimension reduction with v3 models
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        self._model = model
        self._custom_dimensions = dimensions
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("OpenAI package required: pip install openai")
        return self._client

    @property
    def dimensions(self) -> int:
        if self._custom_dimensions:
            return self._custom_dimensions
        return self.MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text using OpenAI API."""
        client = self._get_client()

        kwargs = {"input": text, "model": self._model}
        if self._custom_dimensions and "3-" in self._model:
            kwargs["dimensions"] = self._custom_dimensions

        response = client.embeddings.create(**kwargs)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed multiple texts in batches."""
        client = self._get_client()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            kwargs = {"input": batch, "model": self._model}
            if self._custom_dimensions and "3-" in self._model:
                kwargs["dimensions"] = self._custom_dimensions

            response = client.embeddings.create(**kwargs)

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [np.array(d.embedding, dtype=np.float32) for d in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """Embed with token usage tracking."""
        client = self._get_client()

        kwargs = {"input": text, "model": self._model}
        if self._custom_dimensions and "3-" in self._model:
            kwargs["dimensions"] = self._custom_dimensions

        response = client.embeddings.create(**kwargs)

        return EmbeddingResult(
            vector=np.array(response.data[0].embedding, dtype=np.float32),
            model=self._model,
            dimensions=self.dimensions,
            tokens_used=response.usage.total_tokens
        )


# =============================================================================
# Sentence Transformers Embedder (Local)
# =============================================================================

class SentenceTransformerEmbedder(Embedder):
    """
    Local embeddings using Sentence Transformers.

    Popular models:
    - all-MiniLM-L6-v2: 384 dimensions, fast, good quality
    - all-mpnet-base-v2: 768 dimensions, high quality
    - paraphrase-multilingual-MiniLM-L12-v2: 384 dimensions, multilingual

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimensions = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._dimensions = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required: "
                    "pip install sentence-transformers"
                )
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._load_model()
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text locally."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts with batching."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.astype(np.float32)


# =============================================================================
# Cohere Embedder
# =============================================================================

class CohereEmbedder(Embedder):
    """
    Cohere embeddings via API.

    Models:
    - embed-english-v3.0: 1024 dimensions
    - embed-multilingual-v3.0: 1024 dimensions
    - embed-english-light-v3.0: 384 dimensions (faster)

    Requires: pip install cohere
    """

    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
    }

    def __init__(
        self,
        api_key: str = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document"  # or "search_query"
    ):
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self._api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY or pass api_key.")

        self._model = model
        self._input_type = input_type
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self._api_key)
            except ImportError:
                raise ImportError("Cohere package required: pip install cohere")
        return self._client

    @property
    def dimensions(self) -> int:
        return self.MODEL_DIMENSIONS.get(self._model, 1024)

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text using Cohere API."""
        client = self._get_client()
        response = client.embed(
            texts=[text],
            model=self._model,
            input_type=self._input_type
        )
        return np.array(response.embeddings[0], dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
        """Embed multiple texts in batches."""
        client = self._get_client()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embed(
                texts=batch,
                model=self._model,
                input_type=self._input_type
            )
            all_embeddings.extend(response.embeddings)

        return np.array(all_embeddings, dtype=np.float32)


# =============================================================================
# Mock/Cached Embedder (for testing)
# =============================================================================

class MockEmbedder(Embedder):
    """
    Mock embedder for testing without API calls.

    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return "mock"

    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text hash."""
        # Use hash to generate deterministic but unique embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)

        # Generate and normalize
        vector = rng.randn(self._dimensions).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        return vector


class CachedEmbedder(Embedder):
    """
    Wrapper that caches embeddings to disk.

    Useful for avoiding repeated API calls during development.
    """

    def __init__(self, embedder: Embedder, cache_dir: str = ".embedding_cache"):
        self._embedder = embedder
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    def _load_cache(self):
        cache_file = self._cache_dir / f"{self._embedder.model_name}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                self._cache = {k: np.array(v, dtype=np.float32) for k, v in data.items()}

    def _save_cache(self):
        cache_file = self._cache_dir / f"{self._embedder.model_name}.json"
        data = {k: v.tolist() for k, v in self._cache.items()}
        with open(cache_file, "w") as f:
            json.dump(data, f)

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions

    @property
    def model_name(self) -> str:
        return f"cached:{self._embedder.model_name}"

    def embed(self, text: str) -> np.ndarray:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        vector = self._embedder.embed(text)
        self._cache[key] = vector
        self._save_cache()
        return vector

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self._embedder.embed_batch(uncached_texts, batch_size)
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                key = self._cache_key(text)
                self._cache[key] = embedding
                results.append((uncached_indices[i], embedding))

            self._save_cache()

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results])


# =============================================================================
# Factory Function
# =============================================================================

def get_embedder(
    provider: str = "auto",
    model: str = None,
    api_key: str = None,
    cache: bool = False,
    **kwargs
) -> Embedder:
    """
    Get an embedder instance.

    Args:
        provider: "openai", "sentence-transformers", "cohere", "mock", or "auto"
        model: Model name (provider-specific)
        api_key: API key (for cloud providers)
        cache: Enable disk caching
        **kwargs: Additional provider-specific arguments

    Returns:
        Embedder instance
    """
    embedder = None

    if provider == "auto":
        # Try providers in order of preference
        if os.environ.get("OPENAI_API_KEY") or api_key:
            provider = "openai"
        else:
            # Try to import sentence-transformers
            try:
                import sentence_transformers
                provider = "sentence-transformers"
            except ImportError:
                provider = "mock"

    if provider == "openai":
        embedder = OpenAIEmbedder(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            **kwargs
        )
    elif provider == "sentence-transformers":
        embedder = SentenceTransformerEmbedder(
            model_name=model or "all-MiniLM-L6-v2",
            **kwargs
        )
    elif provider == "cohere":
        embedder = CohereEmbedder(
            api_key=api_key,
            model=model or "embed-english-v3.0",
            **kwargs
        )
    elif provider == "mock":
        embedder = MockEmbedder(dimensions=kwargs.get("dimensions", 384))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if cache and not isinstance(embedder, MockEmbedder):
        embedder = CachedEmbedder(embedder)

    return embedder


# =============================================================================
# Integration with VectorDB
# =============================================================================

class EmbeddingCollection:
    """
    High-level collection that handles embedding automatically.

    Usage:
        from vectordb import VectorDB
        from embeddings import EmbeddingCollection, get_embedder

        db = VectorDB("./my_db")
        embedder = get_embedder()

        # Create embedding-aware collection
        docs = EmbeddingCollection(
            db.create_collection("documents", dimensions=embedder.dimensions),
            embedder
        )

        # Insert with automatic embedding
        docs.add("doc1", "Machine learning is fascinating", {"category": "AI"})

        # Search with text query
        results = docs.search("artificial intelligence", k=5)
    """

    def __init__(self, collection, embedder: Embedder):
        """
        Wrap a collection with automatic embedding.

        Args:
            collection: A Collection instance from vectordb.py
            embedder: An Embedder instance
        """
        self.collection = collection
        self.embedder = embedder

        # Verify dimensions match
        if collection.config.dimensions != embedder.dimensions:
            raise ValueError(
                f"Collection dimensions ({collection.config.dimensions}) "
                f"don't match embedder dimensions ({embedder.dimensions})"
            )

    def add(self, id: str, text: str, metadata: dict = None) -> str:
        """Add a document with automatic embedding."""
        vector = self.embedder.embed(text)

        # Store original text in metadata
        metadata = metadata or {}
        metadata["_text"] = text

        return self.collection.insert(vector, id=id, metadata=metadata)

    def add_batch(self, items: list[tuple[str, str, Optional[dict]]]) -> list[str]:
        """
        Add multiple documents.

        Args:
            items: List of (id, text, metadata) tuples
        """
        ids = [item[0] for item in items]
        texts = [item[1] for item in items]
        metadata_list = [item[2] if len(item) > 2 else {} for item in items]

        # Embed all texts
        vectors = self.embedder.embed_batch(texts)

        # Add original text to metadata
        for i, text in enumerate(texts):
            metadata_list[i] = metadata_list[i] or {}
            metadata_list[i]["_text"] = text

        return self.collection.insert_batch(vectors, ids, metadata_list)

    def search(self, query: str, k: int = 10, filter: dict = None, **kwargs):
        """Search with text query."""
        query_vector = self.embedder.embed(query)
        return self.collection.search(query_vector, k=k, filter=filter, **kwargs)

    def get(self, id: str, include_vector: bool = False):
        """Get a document by ID."""
        return self.collection.get(id, include_vector)

    def delete(self, id: str) -> bool:
        """Delete a document."""
        return self.collection.delete(id)

    def count(self) -> int:
        """Get document count."""
        return self.collection.count()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Embeddings Integration Demo")
    print("=" * 60)

    # Use mock embedder for demo (no API keys needed)
    print("\n--- Mock Embedder (for testing) ---")
    embedder = get_embedder("mock", dimensions=128)
    print(f"Model: {embedder.model_name}")
    print(f"Dimensions: {embedder.dimensions}")

    # Embed some texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing enables computers to understand text",
        "Computer vision analyzes images and videos"
    ]

    print("\nEmbedding texts...")
    embeddings = embedder.embed_batch(texts)
    print(f"Shape: {embeddings.shape}")

    # Show similarity
    print("\nSimilarity matrix (cosine):")
    for i, t1 in enumerate(texts):
        row = []
        for j, t2 in enumerate(texts):
            sim = np.dot(embeddings[i], embeddings[j])
            row.append(f"{sim:.2f}")
        print(f"  {' '.join(row)}  | {t1[:40]}...")

    # Test cached embedder
    print("\n--- Cached Embedder ---")
    cached = CachedEmbedder(embedder, cache_dir="./test_cache")
    v1 = cached.embed("Hello world")
    v2 = cached.embed("Hello world")  # Should be cached
    print(f"Same vector from cache: {np.allclose(v1, v2)}")

    # Integration with VectorDB
    print("\n--- Integration with VectorDB ---")
    try:
        from vectordb import VectorDB

        db = VectorDB("./embedding_demo")

        if "semantic_docs" not in db.list_collections():
            collection = db.create_collection("semantic_docs", dimensions=embedder.dimensions)
        else:
            collection = db.get_collection("semantic_docs")

        # Create embedding collection
        docs = EmbeddingCollection(collection, embedder)

        # Add documents
        items = [
            ("doc1", "Machine learning models can learn from data", {"category": "ML"}),
            ("doc2", "Neural networks are inspired by the human brain", {"category": "ML"}),
            ("doc3", "Databases store and retrieve information efficiently", {"category": "DB"}),
            ("doc4", "Vector search enables semantic similarity matching", {"category": "DB"}),
        ]

        print("\nAdding documents...")
        docs.add_batch(items)
        print(f"Added {docs.count()} documents")

        # Search
        print("\nSearching for 'artificial intelligence learning'...")
        results = docs.search("artificial intelligence learning", k=3)
        for r in results:
            print(f"  {r.id}: score={r.score:.4f}, text={r.metadata.get('_text', '')[:50]}...")

        db.save()
        print("\nDemo complete!")

    except ImportError as e:
        print(f"Skipping VectorDB integration: {e}")

    print("\n" + "=" * 60)
    print("Available providers:")
    print("  - openai: Requires OPENAI_API_KEY")
    print("  - sentence-transformers: Local, requires pip install sentence-transformers")
    print("  - cohere: Requires COHERE_API_KEY")
    print("  - mock: No dependencies, for testing")
