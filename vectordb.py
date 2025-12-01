"""
PyVectorDB - A Python Vector Database Implementation
Phase 2: Metadata Filtering, Batch Ops, Collections

Dependencies:
    pip install numpy hnswlib

Usage:
    from vectordb import VectorDB, Collection

    db = VectorDB("./my_db")
    collection = db.create_collection("documents", dimensions=384)
    collection.insert(vector, id="doc1", metadata={"category": "tech"})
    results = collection.search(query_vector, k=10, filter={"category": "tech"})
"""

import numpy as np
import hnswlib
import json
import pickle
import uuid
import os
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Callable
from enum import Enum
import operator
import re


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class VectorEntry:
    """A vector with its ID and metadata."""
    id: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    vector: Optional[np.ndarray] = None


class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "l2"
    DOT_PRODUCT = "ip"  # inner product


# =============================================================================
# Filter Expression Engine
# =============================================================================

class FilterOp(Enum):
    EQ = "eq"       # equals
    NE = "ne"       # not equals
    GT = "gt"       # greater than
    GTE = "gte"     # greater than or equal
    LT = "lt"       # less than
    LTE = "lte"     # less than or equal
    IN = "in"       # in list
    NIN = "nin"     # not in list
    CONTAINS = "contains"  # string contains
    REGEX = "regex"        # regex match


@dataclass
class FilterCondition:
    """Single filter condition: field op value"""
    field: str
    op: FilterOp
    value: Any

    def evaluate(self, metadata: dict) -> bool:
        """Evaluate this condition against metadata."""
        if self.field not in metadata:
            return False

        actual = metadata[self.field]
        expected = self.value

        if self.op == FilterOp.EQ:
            return actual == expected
        elif self.op == FilterOp.NE:
            return actual != expected
        elif self.op == FilterOp.GT:
            return actual > expected
        elif self.op == FilterOp.GTE:
            return actual >= expected
        elif self.op == FilterOp.LT:
            return actual < expected
        elif self.op == FilterOp.LTE:
            return actual <= expected
        elif self.op == FilterOp.IN:
            return actual in expected
        elif self.op == FilterOp.NIN:
            return actual not in expected
        elif self.op == FilterOp.CONTAINS:
            return expected in str(actual)
        elif self.op == FilterOp.REGEX:
            return bool(re.search(expected, str(actual)))

        return False


class Filter:
    """
    Composable filter expression.

    Usage:
        # Simple equality
        filter = Filter.eq("category", "tech")

        # Compound filters
        filter = Filter.and_([
            Filter.eq("category", "tech"),
            Filter.gte("year", 2020)
        ])

        # Dict shorthand (equality only)
        filter = Filter.from_dict({"category": "tech", "author": "John"})
    """

    def __init__(self, evaluate_fn: Callable[[dict], bool]):
        self._evaluate = evaluate_fn

    def evaluate(self, metadata: dict) -> bool:
        return self._evaluate(metadata)

    @staticmethod
    def eq(field: str, value: Any) -> 'Filter':
        cond = FilterCondition(field, FilterOp.EQ, value)
        return Filter(cond.evaluate)

    @staticmethod
    def ne(field: str, value: Any) -> 'Filter':
        cond = FilterCondition(field, FilterOp.NE, value)
        return Filter(cond.evaluate)

    @staticmethod
    def gt(field: str, value: Any) -> 'Filter':
        cond = FilterCondition(field, FilterOp.GT, value)
        return Filter(cond.evaluate)

    @staticmethod
    def gte(field: str, value: Any) -> 'Filter':
        cond = FilterCondition(field, FilterOp.GTE, value)
        return Filter(cond.evaluate)

    @staticmethod
    def lt(field: str, value: Any) -> 'Filter':
        cond = FilterCondition(field, FilterOp.LT, value)
        return Filter(cond.evaluate)

    @staticmethod
    def lte(field: str, value: Any) -> 'Filter':
        cond = FilterCondition(field, FilterOp.LTE, value)
        return Filter(cond.evaluate)

    @staticmethod
    def in_(field: str, values: list) -> 'Filter':
        cond = FilterCondition(field, FilterOp.IN, values)
        return Filter(cond.evaluate)

    @staticmethod
    def nin(field: str, values: list) -> 'Filter':
        cond = FilterCondition(field, FilterOp.NIN, values)
        return Filter(cond.evaluate)

    @staticmethod
    def contains(field: str, substring: str) -> 'Filter':
        cond = FilterCondition(field, FilterOp.CONTAINS, substring)
        return Filter(cond.evaluate)

    @staticmethod
    def regex(field: str, pattern: str) -> 'Filter':
        cond = FilterCondition(field, FilterOp.REGEX, pattern)
        return Filter(cond.evaluate)

    @staticmethod
    def and_(filters: list['Filter']) -> 'Filter':
        return Filter(lambda m: all(f.evaluate(m) for f in filters))

    @staticmethod
    def or_(filters: list['Filter']) -> 'Filter':
        return Filter(lambda m: any(f.evaluate(m) for f in filters))

    @staticmethod
    def not_(filter_: 'Filter') -> 'Filter':
        return Filter(lambda m: not filter_.evaluate(m))

    @staticmethod
    def from_dict(d: dict) -> 'Filter':
        """Create filter from dict (equality conditions, AND'd together)."""
        if not d:
            return Filter(lambda m: True)
        filters = [Filter.eq(k, v) for k, v in d.items()]
        return Filter.and_(filters)


# =============================================================================
# Collection - Single namespace of vectors
# =============================================================================

@dataclass
class CollectionConfig:
    """Configuration for a collection."""
    name: str
    dimensions: int
    metric: DistanceMetric = DistanceMetric.COSINE
    M: int = 16                    # HNSW: connections per layer
    ef_construction: int = 200     # HNSW: construction quality
    ef_search: int = 50            # HNSW: search quality
    max_elements: int = 1_000_000  # Max vectors


class Collection:
    """
    A collection of vectors with HNSW indexing.

    Thread-safe for concurrent reads, serialized writes.
    """

    def __init__(self, config: CollectionConfig, base_path: Path):
        self.config = config
        self.base_path = base_path / config.name
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        # Storage
        self._index: Optional[hnswlib.Index] = None
        self._metadata: dict[str, dict] = {}      # id -> metadata
        self._id_to_label: dict[str, int] = {}    # id -> hnsw label
        self._label_to_id: dict[int, str] = {}    # hnsw label -> id
        self._vectors: dict[str, np.ndarray] = {} # id -> vector (for retrieval)
        self._next_label: int = 0

        # Load existing data
        self._load()

    def _init_index(self):
        """Initialize HNSW index."""
        space = self.config.metric.value
        self._index = hnswlib.Index(space=space, dim=self.config.dimensions)
        self._index.init_index(
            max_elements=self.config.max_elements,
            ef_construction=self.config.ef_construction,
            M=self.config.M
        )
        self._index.set_ef(self.config.ef_search)

    def _load(self):
        """Load collection from disk."""
        index_path = self.base_path / "index.bin"
        meta_path = self.base_path / "metadata.json"
        vectors_path = self.base_path / "vectors.npy"
        state_path = self.base_path / "state.json"

        if index_path.exists() and state_path.exists():
            # Load state
            with open(state_path, "r") as f:
                state = json.load(f)
            self._id_to_label = state["id_to_label"]
            self._label_to_id = {int(k): v for k, v in state["label_to_id"].items()}
            self._next_label = state["next_label"]

            # Load index
            self._init_index()
            self._index.load_index(str(index_path))

            # Load metadata
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    self._metadata = json.load(f)

            # Load vectors
            if vectors_path.exists():
                vectors_data = np.load(vectors_path, allow_pickle=True).item()
                self._vectors = vectors_data
        else:
            self._init_index()

    def save(self):
        """Persist collection to disk."""
        with self._lock:
            # Save index
            self._index.save_index(str(self.base_path / "index.bin"))

            # Save metadata
            with open(self.base_path / "metadata.json", "w") as f:
                json.dump(self._metadata, f)

            # Save vectors
            np.save(self.base_path / "vectors.npy", self._vectors)

            # Save state
            state = {
                "id_to_label": self._id_to_label,
                "label_to_id": {str(k): v for k, v in self._label_to_id.items()},
                "next_label": self._next_label
            }
            with open(self.base_path / "state.json", "w") as f:
                json.dump(state, f)

            # Save config
            with open(self.base_path / "config.json", "w") as f:
                json.dump({
                    "name": self.config.name,
                    "dimensions": self.config.dimensions,
                    "metric": self.config.metric.value,
                    "M": self.config.M,
                    "ef_construction": self.config.ef_construction,
                    "ef_search": self.config.ef_search,
                    "max_elements": self.config.max_elements
                }, f)

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def insert(self, vector: np.ndarray, id: str = None,
               metadata: dict = None) -> str:
        """Insert a single vector."""
        with self._lock:
            id = id or str(uuid.uuid4())

            if id in self._id_to_label:
                raise ValueError(f"ID '{id}' already exists. Use upsert() to update.")

            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape[0] != self.config.dimensions:
                raise ValueError(
                    f"Vector has {vector.shape[0]} dimensions, "
                    f"expected {self.config.dimensions}"
                )

            label = self._next_label
            self._next_label += 1

            # Add to HNSW index
            self._index.add_items(vector.reshape(1, -1), [label])

            # Store mappings
            self._id_to_label[id] = label
            self._label_to_id[label] = id
            self._vectors[id] = vector

            if metadata:
                self._metadata[id] = metadata

            return id

    def insert_batch(self, vectors: np.ndarray, ids: list[str] = None,
                     metadata_list: list[dict] = None) -> list[str]:
        """
        Insert multiple vectors efficiently.

        Args:
            vectors: (N, D) array of vectors
            ids: Optional list of IDs (auto-generated if None)
            metadata_list: Optional list of metadata dicts

        Returns:
            List of IDs
        """
        with self._lock:
            vectors = np.asarray(vectors, dtype=np.float32)
            n = vectors.shape[0]

            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                n = 1

            if vectors.shape[1] != self.config.dimensions:
                raise ValueError(
                    f"Vectors have {vectors.shape[1]} dimensions, "
                    f"expected {self.config.dimensions}"
                )

            # Generate IDs if needed
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(n)]

            if len(ids) != n:
                raise ValueError(f"Got {len(ids)} IDs for {n} vectors")

            # Check for duplicates
            for id in ids:
                if id in self._id_to_label:
                    raise ValueError(f"ID '{id}' already exists")

            # Generate labels
            labels = list(range(self._next_label, self._next_label + n))
            self._next_label += n

            # Batch add to HNSW
            self._index.add_items(vectors, labels)

            # Store mappings
            for i, (id, label) in enumerate(zip(ids, labels)):
                self._id_to_label[id] = label
                self._label_to_id[label] = id
                self._vectors[id] = vectors[i]

                if metadata_list and i < len(metadata_list) and metadata_list[i]:
                    self._metadata[id] = metadata_list[i]

            return ids

    def upsert(self, vector: np.ndarray, id: str, metadata: dict = None) -> str:
        """Insert or update a vector."""
        with self._lock:
            if id in self._id_to_label:
                self.delete(id)
            return self.insert(vector, id, metadata)

    def get(self, id: str, include_vector: bool = False) -> Optional[dict]:
        """Get a vector entry by ID."""
        if id not in self._id_to_label:
            return None

        result = {
            "id": id,
            "metadata": self._metadata.get(id, {})
        }

        if include_vector:
            result["vector"] = self._vectors.get(id)

        return result

    def get_batch(self, ids: list[str], include_vectors: bool = False) -> list[Optional[dict]]:
        """Get multiple vectors by ID."""
        return [self.get(id, include_vectors) for id in ids]

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        with self._lock:
            if id not in self._id_to_label:
                return False

            label = self._id_to_label[id]

            # Mark as deleted in HNSW (can't actually remove)
            self._index.mark_deleted(label)

            # Remove from mappings
            del self._id_to_label[id]
            del self._label_to_id[label]
            del self._vectors[id]

            if id in self._metadata:
                del self._metadata[id]

            return True

    def delete_batch(self, ids: list[str]) -> int:
        """Delete multiple vectors. Returns count of deleted."""
        deleted = 0
        for id in ids:
            if self.delete(id):
                deleted += 1
        return deleted

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10,
               filter: Filter | dict = None,
               include_vectors: bool = False,
               ef_search: int = None) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query: Query vector
            k: Number of results
            filter: Filter expression or dict for metadata filtering
            include_vectors: Include vectors in results
            ef_search: Override ef_search parameter (higher = more accurate)

        Returns:
            List of SearchResult sorted by score (lower is better)
        """
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        if query.shape[1] != self.config.dimensions:
            raise ValueError(
                f"Query has {query.shape[1]} dimensions, "
                f"expected {self.config.dimensions}"
            )

        # Convert dict filter to Filter object
        if isinstance(filter, dict):
            filter = Filter.from_dict(filter)

        # Set ef_search if provided
        if ef_search:
            self._index.set_ef(ef_search)

        # If filtering, we need to fetch more candidates
        fetch_k = k * 10 if filter else k
        fetch_k = min(fetch_k, len(self._id_to_label))

        if fetch_k == 0:
            return []

        # HNSW search
        labels, distances = self._index.knn_query(query, k=fetch_k)

        # Reset ef_search
        if ef_search:
            self._index.set_ef(self.config.ef_search)

        # Build results with filtering
        results = []
        for label, distance in zip(labels[0], distances[0]):
            label = int(label)
            if label not in self._label_to_id:
                continue

            id = self._label_to_id[label]
            metadata = self._metadata.get(id, {})

            # Apply filter
            if filter and not filter.evaluate(metadata):
                continue

            result = SearchResult(
                id=id,
                score=float(distance),
                metadata=metadata
            )

            if include_vectors:
                result.vector = self._vectors.get(id)

            results.append(result)

            if len(results) >= k:
                break

        return results

    def search_batch(self, queries: np.ndarray, k: int = 10,
                     filter: Filter | dict = None) -> list[list[SearchResult]]:
        """Search for multiple queries."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        return [self.search(q, k, filter) for q in queries]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def count(self) -> int:
        """Number of vectors in collection."""
        return len(self._id_to_label)

    def __len__(self) -> int:
        return self.count()

    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List vector IDs."""
        ids = list(self._id_to_label.keys())
        return ids[offset:offset + limit]

    def set_ef_search(self, ef: int):
        """Set search quality parameter."""
        self.config.ef_search = ef
        self._index.set_ef(ef)


# =============================================================================
# VectorDB - Multi-collection database
# =============================================================================

class VectorDB:
    """
    Multi-collection vector database.

    Usage:
        db = VectorDB("./my_database")

        # Create collection
        docs = db.create_collection("documents", dimensions=384)

        # Insert vectors
        docs.insert(embedding, id="doc1", metadata={"title": "Hello"})

        # Search
        results = docs.search(query_embedding, k=10)

        # With filters
        results = docs.search(query, k=10, filter={"category": "tech"})

        # Save to disk
        db.save()
    """

    def __init__(self, path: str = "./vectordb"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, Collection] = {}
        self._load_collections()

    def _load_collections(self):
        """Load existing collections from disk."""
        for item in self.path.iterdir():
            if item.is_dir():
                config_path = item / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config_data = json.load(f)

                    config = CollectionConfig(
                        name=config_data["name"],
                        dimensions=config_data["dimensions"],
                        metric=DistanceMetric(config_data.get("metric", "cosine")),
                        M=config_data.get("M", 16),
                        ef_construction=config_data.get("ef_construction", 200),
                        ef_search=config_data.get("ef_search", 50),
                        max_elements=config_data.get("max_elements", 1_000_000)
                    )

                    self._collections[config.name] = Collection(config, self.path)

    def create_collection(self, name: str, dimensions: int,
                          metric: str = "cosine",
                          **kwargs) -> Collection:
        """
        Create a new collection.

        Args:
            name: Collection name
            dimensions: Vector dimensions
            metric: Distance metric ("cosine", "l2", "ip")
            **kwargs: Additional HNSW parameters (M, ef_construction, etc.)
        """
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")

        config = CollectionConfig(
            name=name,
            dimensions=dimensions,
            metric=DistanceMetric(metric),
            **kwargs
        )

        collection = Collection(config, self.path)
        self._collections[name] = collection
        collection.save()

        return collection

    def get_collection(self, name: str) -> Collection:
        """Get a collection by name."""
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' not found")
        return self._collections[name]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name not in self._collections:
            return False

        import shutil
        collection_path = self.path / name
        if collection_path.exists():
            shutil.rmtree(collection_path)

        del self._collections[name]
        return True

    def list_collections(self) -> list[str]:
        """List all collection names."""
        return list(self._collections.keys())

    def save(self):
        """Save all collections to disk."""
        for collection in self._collections.values():
            collection.save()

    def __getitem__(self, name: str) -> Collection:
        return self.get_collection(name)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("PyVectorDB Demo")
    print("=" * 60)

    # Create database
    db = VectorDB("./demo_db")

    # Create collection
    if "documents" not in db.list_collections():
        docs = db.create_collection("documents", dimensions=128)
    else:
        docs = db.get_collection("documents")

    # Generate sample data
    np.random.seed(42)
    n_vectors = 10000
    dimensions = 128

    print(f"\nInserting {n_vectors} vectors...")
    start = time.time()

    # Batch insert
    vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
    # Normalize for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    ids = [f"doc_{i}" for i in range(n_vectors)]
    metadata_list = [
        {
            "category": np.random.choice(["tech", "science", "business", "sports"]),
            "year": int(np.random.randint(2015, 2025)),
            "score": float(np.random.rand()),
            "index": i
        }
        for i in range(n_vectors)
    ]

    docs.insert_batch(vectors, ids, metadata_list)

    print(f"Inserted in {time.time() - start:.2f}s")
    print(f"Collection size: {len(docs)}")

    # Save
    db.save()
    print("Saved to disk")

    # Search without filter
    query = np.random.randn(dimensions).astype(np.float32)
    query = query / np.linalg.norm(query)

    print("\n--- Search without filter ---")
    start = time.time()
    results = docs.search(query, k=5)
    print(f"Search time: {(time.time() - start) * 1000:.2f}ms")
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}")

    # Search with dict filter
    print("\n--- Search with filter: category='tech' ---")
    start = time.time()
    results = docs.search(query, k=5, filter={"category": "tech"})
    print(f"Search time: {(time.time() - start) * 1000:.2f}ms")
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}")

    # Search with complex filter
    print("\n--- Search with filter: category='tech' AND year >= 2020 ---")
    complex_filter = Filter.and_([
        Filter.eq("category", "tech"),
        Filter.gte("year", 2020)
    ])
    start = time.time()
    results = docs.search(query, k=5, filter=complex_filter)
    print(f"Search time: {(time.time() - start) * 1000:.2f}ms")
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}, year={r.metadata.get('year')}")

    # Search with OR filter
    print("\n--- Search with filter: category IN ['tech', 'science'] ---")
    results = docs.search(query, k=5, filter=Filter.in_("category", ["tech", "science"]))
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}")

    print("\n" + "=" * 60)
    print("Demo complete!")
