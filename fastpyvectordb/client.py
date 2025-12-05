"""
FastPyVectorDB - High-Level Client API

A simple, ChromaDB-like interface for the FastPyVectorDB vector database.

Usage:
    import fastpyvectordb

    # Create a client
    client = fastpyvectordb.Client()

    # Create a collection
    collection = client.create_collection("my_docs", embedding_model="all-MiniLM-L6-v2")

    # Add documents
    collection.add(
        documents=["Hello world", "Machine learning is great"],
        ids=["doc1", "doc2"],
        metadatas=[{"source": "web"}, {"source": "book"}]
    )

    # Search
    results = collection.query("What is ML?", n_results=5)

    # Get by ID
    doc = collection.get("doc1")
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectordb_optimized import VectorDB, Collection as BaseCollection, Filter, SearchResult, DistanceMetric
from embeddings import (
    Embedder,
    get_embedder,
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    MockEmbedder
)


@dataclass
class QueryResult:
    """Result from a query operation."""
    ids: list[list[str]]
    documents: list[list[Optional[str]]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
    embeddings: Optional[list[list[np.ndarray]]] = None


@dataclass
class GetResult:
    """Result from a get operation."""
    ids: list[str]
    documents: list[Optional[str]]
    metadatas: list[dict]
    embeddings: Optional[list[np.ndarray]] = None


class Collection:
    """
    A document collection with automatic embedding support.

    Similar to ChromaDB's Collection interface. Handles document embedding,
    storage, and semantic search automatically.
    """

    def __init__(
        self,
        name: str,
        base_collection: BaseCollection,
        embedder: Embedder,
        metadata: Optional[dict] = None
    ):
        self.name = name
        self._collection = base_collection
        self._embedder = embedder
        self.metadata = metadata or {}

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def __len__(self) -> int:
        return self.count

    def add(
        self,
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[list[float]]] = None,
        ids: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None
    ) -> list[str]:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts to embed and store
            embeddings: Pre-computed embeddings (optional, skip embedding if provided)
            ids: Unique IDs for each document (auto-generated if not provided)
            metadatas: Metadata dictionaries for each document

        Returns:
            List of document IDs

        Example:
            collection.add(
                documents=["Hello world", "Goodbye world"],
                ids=["id1", "id2"],
                metadatas=[{"source": "a"}, {"source": "b"}]
            )
        """
        if documents is None and embeddings is None:
            raise ValueError("Either documents or embeddings must be provided")

        n = len(documents) if documents else len(embeddings)

        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(n)]
        elif len(ids) != n:
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of documents ({n})")

        # Check for duplicates
        existing = set(ids) & set(self._collection._id_to_label.keys())
        if existing:
            raise ValueError(f"IDs already exist: {existing}")

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        elif len(metadatas) != n:
            raise ValueError(f"Number of metadatas ({len(metadatas)}) must match number of documents ({n})")

        # Store original documents in metadata
        if documents:
            for i, doc in enumerate(documents):
                metadatas[i] = dict(metadatas[i])  # Copy to avoid mutation
                metadatas[i]["_document"] = doc

        # Get or compute embeddings
        if embeddings is not None:
            vectors = np.array(embeddings, dtype=np.float32)
        else:
            vectors = self._embedder.embed_batch(documents)

        # Insert into collection
        return self._collection.insert_batch(vectors, ids, metadatas)

    def upsert(
        self,
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[list[float]]] = None,
        ids: list[str] = None,
        metadatas: Optional[list[dict]] = None
    ) -> list[str]:
        """
        Add or update documents in the collection.

        Same as add(), but updates existing documents if IDs already exist.
        """
        if ids is None:
            raise ValueError("IDs must be provided for upsert")

        # Delete existing IDs first
        for id in ids:
            if id in self._collection._id_to_label:
                self._collection.delete(id)

        # Now add
        return self.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def query(
        self,
        query_texts: Optional[Union[str, list[str]]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        include: list[str] = None
    ) -> QueryResult:
        """
        Query the collection for similar documents.

        Args:
            query_texts: Text(s) to search for (will be embedded)
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return per query
            where: Filter conditions (e.g., {"category": "tech"})
            include: What to include in results: ["documents", "metadatas", "embeddings", "distances"]

        Returns:
            QueryResult with matching documents

        Example:
            results = collection.query(
                query_texts="What is machine learning?",
                n_results=5,
                where={"category": "tech"}
            )
        """
        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        include = include or ["documents", "metadatas", "distances"]

        # Handle single query text
        if isinstance(query_texts, str):
            query_texts = [query_texts]

        # Get query embeddings
        if query_embeddings is not None:
            queries = np.array(query_embeddings, dtype=np.float32)
        else:
            queries = self._embedder.embed_batch(query_texts)

        # Prepare filter
        filter_obj = Filter.from_dict(where) if where else None

        # Execute searches
        include_vectors = "embeddings" in include

        all_ids = []
        all_documents = []
        all_metadatas = []
        all_distances = []
        all_embeddings = []

        for query in queries:
            results = self._collection.search(
                query,
                k=n_results,
                filter=filter_obj,
                include_vectors=include_vectors
            )

            ids = []
            documents = []
            metadatas = []
            distances = []
            embeddings = []

            for r in results:
                ids.append(r.id)
                distances.append(r.score)
                metadatas.append({k: v for k, v in r.metadata.items() if not k.startswith("_")})
                documents.append(r.metadata.get("_document"))
                if r.vector is not None:
                    embeddings.append(r.vector)

            all_ids.append(ids)
            all_documents.append(documents)
            all_metadatas.append(metadatas)
            all_distances.append(distances)
            if embeddings:
                all_embeddings.append(embeddings)

        return QueryResult(
            ids=all_ids,
            documents=all_documents if "documents" in include else [[None] * len(all_ids[i]) for i in range(len(all_ids))],
            metadatas=all_metadatas if "metadatas" in include else [[{}] * len(all_ids[i]) for i in range(len(all_ids))],
            distances=all_distances if "distances" in include else [[0.0] * len(all_ids[i]) for i in range(len(all_ids))],
            embeddings=all_embeddings if "embeddings" in include and all_embeddings else None
        )

    def get(
        self,
        ids: Optional[Union[str, list[str]]] = None,
        where: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include: list[str] = None
    ) -> GetResult:
        """
        Get documents by ID or filter.

        Args:
            ids: Single ID or list of IDs to retrieve
            where: Filter conditions
            limit: Maximum number of results
            offset: Number of results to skip
            include: What to include: ["documents", "metadatas", "embeddings"]

        Returns:
            GetResult with matching documents

        Example:
            # Get by ID
            result = collection.get(ids=["doc1", "doc2"])

            # Get with filter
            result = collection.get(where={"category": "tech"}, limit=10)
        """
        include = include or ["documents", "metadatas"]
        include_vectors = "embeddings" in include

        result_ids = []
        result_documents = []
        result_metadatas = []
        result_embeddings = []

        if ids is not None:
            # Get by specific IDs
            if isinstance(ids, str):
                ids = [ids]

            results = self._collection.get_batch(ids, include_vectors=include_vectors)

            for id, r in zip(ids, results):
                if r is not None:
                    result_ids.append(r["id"])
                    metadata = {k: v for k, v in r["metadata"].items() if not k.startswith("_")}
                    result_metadatas.append(metadata)
                    result_documents.append(r["metadata"].get("_document"))
                    if include_vectors and r.get("vector") is not None:
                        result_embeddings.append(r["vector"])
        else:
            # Get all with optional filter
            all_ids = self._collection.list_ids(limit=limit or 1000, offset=offset)
            filter_obj = Filter.from_dict(where) if where else None

            for id in all_ids:
                r = self._collection.get(id, include_vector=include_vectors)
                if r is None:
                    continue

                metadata = r["metadata"]
                if filter_obj and not filter_obj.evaluate(metadata):
                    continue

                result_ids.append(r["id"])
                result_metadatas.append({k: v for k, v in metadata.items() if not k.startswith("_")})
                result_documents.append(metadata.get("_document"))
                if include_vectors and r.get("vector") is not None:
                    result_embeddings.append(r["vector"])

                if limit and len(result_ids) >= limit:
                    break

        return GetResult(
            ids=result_ids,
            documents=result_documents if "documents" in include else [None] * len(result_ids),
            metadatas=result_metadatas if "metadatas" in include else [{}] * len(result_ids),
            embeddings=result_embeddings if "embeddings" in include and result_embeddings else None
        )

    def update(
        self,
        ids: list[str],
        documents: Optional[list[str]] = None,
        embeddings: Optional[list[list[float]]] = None,
        metadatas: Optional[list[dict]] = None
    ):
        """
        Update existing documents.

        Args:
            ids: IDs of documents to update
            documents: New document texts (will be re-embedded)
            embeddings: New embeddings
            metadatas: New metadata (merged with existing)
        """
        for i, id in enumerate(ids):
            existing = self._collection.get(id, include_vector=True)
            if existing is None:
                raise ValueError(f"Document with ID '{id}' not found")

            # Get new embedding
            if embeddings is not None:
                vector = np.array(embeddings[i], dtype=np.float32)
            elif documents is not None:
                vector = self._embedder.embed(documents[i])
            else:
                vector = existing["vector"]

            # Merge metadata
            new_metadata = dict(existing["metadata"])
            if metadatas is not None and i < len(metadatas):
                new_metadata.update(metadatas[i])
            if documents is not None and i < len(documents):
                new_metadata["_document"] = documents[i]

            # Upsert
            self._collection.upsert(vector, id, new_metadata)

    def delete(
        self,
        ids: Optional[Union[str, list[str]]] = None,
        where: Optional[dict] = None
    ) -> int:
        """
        Delete documents from the collection.

        Args:
            ids: Single ID or list of IDs to delete
            where: Filter conditions (delete all matching)

        Returns:
            Number of documents deleted
        """
        if ids is None and where is None:
            raise ValueError("Either ids or where must be provided")

        to_delete = []

        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            to_delete.extend(ids)

        if where is not None:
            # Find matching IDs
            filter_obj = Filter.from_dict(where)
            for id in self._collection.list_ids(limit=100000):
                r = self._collection.get(id)
                if r and filter_obj.evaluate(r["metadata"]):
                    to_delete.append(id)

        return self._collection.delete_batch(list(set(to_delete)))

    def peek(self, limit: int = 10) -> GetResult:
        """
        Get a sample of documents from the collection.

        Args:
            limit: Number of documents to return

        Returns:
            GetResult with sample documents
        """
        return self.get(limit=limit)


class Client:
    """
    FastPyVectorDB Client - Main entry point for the vector database.

    Similar to ChromaDB's Client interface. Manages collections and
    provides a simple API for document storage and semantic search.

    Example:
        import fastpyvectordb

        # Create client (data stored in ./vectordb by default)
        client = fastpyvectordb.Client()

        # Or specify a path
        client = fastpyvectordb.Client(path="./my_data")

        # Create a collection with auto-embedding
        collection = client.create_collection(
            name="documents",
            embedding_model="all-MiniLM-L6-v2"  # Local model, no API key needed
        )

        # Add documents
        collection.add(
            documents=["Machine learning is amazing", "Python is great"],
            ids=["ml_doc", "python_doc"],
            metadatas=[{"topic": "AI"}, {"topic": "programming"}]
        )

        # Search
        results = collection.query("artificial intelligence", n_results=5)
        print(results.documents[0])  # Top matches
    """

    def __init__(
        self,
        path: str = "./vectordb",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "auto"
    ):
        """
        Initialize the FastPyVectorDB client.

        Args:
            path: Directory to store database files
            embedding_model: Default embedding model for new collections
            embedding_provider: Embedding provider ("auto", "sentence-transformers", "openai", "mock")
        """
        self.path = Path(path)
        self._db = VectorDB(str(self.path))
        self._embedders: dict[str, Embedder] = {}
        self._collections: dict[str, Collection] = {}
        self._default_embedding_model = embedding_model
        self._default_embedding_provider = embedding_provider

    def _get_embedder(
        self,
        model: str = None,
        provider: str = None
    ) -> Embedder:
        """Get or create an embedder instance."""
        model = model or self._default_embedding_model
        provider = provider or self._default_embedding_provider

        key = f"{provider}:{model}"

        if key not in self._embedders:
            self._embedders[key] = get_embedder(
                provider=provider,
                model=model
            )

        return self._embedders[key]

    def create_collection(
        self,
        name: str,
        embedding_model: str = None,
        embedding_provider: str = None,
        metadata: Optional[dict] = None,
        distance_metric: str = "cosine"
    ) -> Collection:
        """
        Create a new collection.

        Args:
            name: Unique name for the collection
            embedding_model: Model to use for embeddings (default: all-MiniLM-L6-v2)
            embedding_provider: Provider ("auto", "sentence-transformers", "openai", "mock")
            metadata: Optional metadata to store with the collection
            distance_metric: Distance metric ("cosine", "l2", "ip")

        Returns:
            Collection instance

        Example:
            # Using local sentence-transformers (no API key needed)
            collection = client.create_collection(
                name="my_docs",
                embedding_model="all-MiniLM-L6-v2"
            )

            # Using OpenAI embeddings
            collection = client.create_collection(
                name="my_docs",
                embedding_model="text-embedding-3-small",
                embedding_provider="openai"
            )
        """
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists. Use get_collection() or get_or_create_collection().")

        # Get embedder and dimensions
        embedder = self._get_embedder(embedding_model, embedding_provider)

        # Create base collection
        base_collection = self._db.create_collection(
            name=name,
            dimensions=embedder.dimensions,
            metric=distance_metric
        )

        # Create wrapper
        collection = Collection(
            name=name,
            base_collection=base_collection,
            embedder=embedder,
            metadata=metadata
        )

        self._collections[name] = collection
        return collection

    def get_collection(
        self,
        name: str,
        embedding_model: str = None,
        embedding_provider: str = None
    ) -> Collection:
        """
        Get an existing collection.

        Args:
            name: Name of the collection
            embedding_model: Model to use for embeddings
            embedding_provider: Provider for embeddings

        Returns:
            Collection instance

        Raises:
            ValueError: If collection doesn't exist
        """
        # Return cached if available
        if name in self._collections:
            return self._collections[name]

        # Get from database
        base_collection = self._db.get_collection(name)

        # Get embedder
        embedder = self._get_embedder(embedding_model, embedding_provider)

        # Verify dimensions match
        if embedder.dimensions != base_collection.config.dimensions:
            raise ValueError(
                f"Embedder dimensions ({embedder.dimensions}) don't match "
                f"collection dimensions ({base_collection.config.dimensions}). "
                f"Use the same embedding model that was used to create the collection."
            )

        collection = Collection(
            name=name,
            base_collection=base_collection,
            embedder=embedder
        )

        self._collections[name] = collection
        return collection

    def get_or_create_collection(
        self,
        name: str,
        embedding_model: str = None,
        embedding_provider: str = None,
        metadata: Optional[dict] = None,
        distance_metric: str = "cosine"
    ) -> Collection:
        """
        Get a collection if it exists, or create it.

        Args:
            name: Name of the collection
            embedding_model: Model to use for embeddings
            embedding_provider: Provider for embeddings
            metadata: Metadata for new collection
            distance_metric: Distance metric for new collection

        Returns:
            Collection instance
        """
        try:
            return self.get_collection(name, embedding_model, embedding_provider)
        except ValueError:
            return self.create_collection(
                name=name,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                metadata=metadata,
                distance_metric=distance_metric
            )

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Name of the collection to delete

        Returns:
            True if deleted, False if not found
        """
        if name in self._collections:
            del self._collections[name]

        return self._db.delete_collection(name)

    def list_collections(self) -> list[str]:
        """
        List all collection names.

        Returns:
            List of collection names
        """
        return self._db.list_collections()

    def heartbeat(self) -> int:
        """
        Check if the database is accessible.

        Returns:
            Current timestamp in nanoseconds
        """
        import time
        return int(time.time() * 1e9)

    def persist(self):
        """
        Persist all collections to disk.

        Call this to ensure all data is saved.
        """
        self._db.save()

    def reset(self):
        """
        Delete all collections and reset the database.

        Warning: This is destructive and cannot be undone.
        """
        for name in list(self._collections.keys()):
            self.delete_collection(name)

        for name in self._db.list_collections():
            self._db.delete_collection(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.persist()
