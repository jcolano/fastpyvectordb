"""
Hybrid Search: Vector Similarity + BM25 Keyword Matching

This module adds keyword search capability that can be combined with
vector similarity for better retrieval results.

Usage:
    from hybrid_search import HybridCollection

    collection = HybridCollection("docs", dimensions=384, text_field="content")

    collection.insert(
        vector=embedding,
        id="doc1",
        metadata={"content": "Machine learning is a subset of AI", "category": "tech"}
    )

    # Hybrid search: combines vector similarity with keyword matching
    results = collection.hybrid_search(
        vector=query_embedding,
        query_text="machine learning",
        k=10,
        alpha=0.5  # 0 = pure keyword, 1 = pure vector
    )
"""

import numpy as np
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from vectordb import Collection, CollectionConfig, SearchResult, Filter
from pathlib import Path
import json


# =============================================================================
# BM25 Implementation
# =============================================================================

@dataclass
class BM25Config:
    """BM25 configuration parameters."""
    k1: float = 1.5     # Term frequency saturation
    b: float = 0.75     # Length normalization


class BM25Index:
    """
    BM25 (Best Matching 25) inverted index for keyword search.

    BM25 is a ranking function used in information retrieval.
    It's based on term frequency (TF) and inverse document frequency (IDF).
    """

    def __init__(self, config: BM25Config = None):
        self.config = config or BM25Config()

        # Document storage
        self._docs: dict[str, list[str]] = {}  # id -> tokens

        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)

        # Statistics
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0
        self._total_docs: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_document(self, doc_id: str, text: str):
        """Add a document to the index."""
        if doc_id in self._docs:
            self.remove_document(doc_id)

        tokens = self._tokenize(text)
        self._docs[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)

        # Update inverted index
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            self._inverted_index[term][doc_id] = count

        # Update statistics
        self._total_docs = len(self._docs)
        self._avg_doc_length = sum(self._doc_lengths.values()) / max(1, self._total_docs)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._docs:
            return False

        tokens = self._docs[doc_id]
        term_counts = Counter(tokens)

        # Remove from inverted index
        for term in term_counts:
            if term in self._inverted_index:
                self._inverted_index[term].pop(doc_id, None)
                if not self._inverted_index[term]:
                    del self._inverted_index[term]

        del self._docs[doc_id]
        del self._doc_lengths[doc_id]

        # Update statistics
        self._total_docs = len(self._docs)
        self._avg_doc_length = sum(self._doc_lengths.values()) / max(1, self._total_docs) if self._total_docs > 0 else 0

        return True

    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency."""
        if term not in self._inverted_index:
            return 0

        n = self._total_docs
        df = len(self._inverted_index[term])  # Document frequency

        # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _score_document(self, doc_id: str, query_terms: list[str]) -> float:
        """Calculate BM25 score for a document."""
        if doc_id not in self._docs:
            return 0

        score = 0
        doc_length = self._doc_lengths[doc_id]
        k1 = self.config.k1
        b = self.config.b

        for term in query_terms:
            if term not in self._inverted_index:
                continue
            if doc_id not in self._inverted_index[term]:
                continue

            tf = self._inverted_index[term][doc_id]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_length / self._avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search for documents matching the query."""
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # Find candidate documents (those containing at least one query term)
        candidates = set()
        for term in query_terms:
            if term in self._inverted_index:
                candidates.update(self._inverted_index[term].keys())

        # Score all candidates
        scores = []
        for doc_id in candidates:
            score = self._score_document(doc_id, query_terms)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]

    def to_dict(self) -> dict:
        """Serialize index to dict."""
        return {
            "docs": self._docs,
            "doc_lengths": self._doc_lengths,
            "avg_doc_length": self._avg_doc_length,
            "total_docs": self._total_docs,
            "inverted_index": {k: dict(v) for k, v in self._inverted_index.items()},
            "config": {"k1": self.config.k1, "b": self.config.b}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BM25Index':
        """Deserialize index from dict."""
        config = BM25Config(**data.get("config", {}))
        index = cls(config)
        index._docs = data.get("docs", {})
        index._doc_lengths = data.get("doc_lengths", {})
        index._avg_doc_length = data.get("avg_doc_length", 0)
        index._total_docs = data.get("total_docs", 0)
        index._inverted_index = defaultdict(dict)
        for term, docs in data.get("inverted_index", {}).items():
            index._inverted_index[term] = docs
        return index


# =============================================================================
# Hybrid Collection
# =============================================================================

@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    id: str
    score: float                    # Combined score
    vector_score: float             # Vector similarity score
    keyword_score: float            # BM25 score
    metadata: dict = field(default_factory=dict)
    vector: Optional[np.ndarray] = None


class HybridCollection(Collection):
    """
    Collection with both vector similarity and keyword (BM25) search.

    Supports three search modes:
    1. Vector search only (standard similarity search)
    2. Keyword search only (BM25)
    3. Hybrid search (weighted combination)
    """

    def __init__(self, config: CollectionConfig, base_path: Path,
                 text_fields: list[str] = None):
        """
        Initialize hybrid collection.

        Args:
            config: Collection configuration
            base_path: Storage path
            text_fields: Fields to index for keyword search (default: all string fields)
        """
        super().__init__(config, base_path)
        self.text_fields = text_fields or []
        self._bm25 = BM25Index()
        self._load_bm25()

    def _load_bm25(self):
        """Load BM25 index from disk."""
        bm25_path = self.base_path / "bm25_index.json"
        if bm25_path.exists():
            with open(bm25_path, "r") as f:
                data = json.load(f)
            self._bm25 = BM25Index.from_dict(data)

    def _save_bm25(self):
        """Save BM25 index to disk."""
        bm25_path = self.base_path / "bm25_index.json"
        with open(bm25_path, "w") as f:
            json.dump(self._bm25.to_dict(), f)

    def save(self):
        """Save collection and BM25 index."""
        super().save()
        self._save_bm25()

    def _extract_text(self, metadata: dict) -> str:
        """Extract text from metadata for BM25 indexing."""
        if not metadata:
            return ""

        texts = []
        for key, value in metadata.items():
            # Index specified fields, or all string fields if none specified
            if self.text_fields:
                if key in self.text_fields and isinstance(value, str):
                    texts.append(value)
            else:
                if isinstance(value, str):
                    texts.append(value)

        return " ".join(texts)

    def insert(self, vector: np.ndarray, id: str = None,
               metadata: dict = None) -> str:
        """Insert vector with BM25 indexing."""
        id = super().insert(vector, id, metadata)

        # Index for BM25
        if metadata:
            text = self._extract_text(metadata)
            if text:
                self._bm25.add_document(id, text)

        return id

    def insert_batch(self, vectors: np.ndarray, ids: list[str] = None,
                     metadata_list: list[dict] = None) -> list[str]:
        """Insert batch with BM25 indexing."""
        ids = super().insert_batch(vectors, ids, metadata_list)

        # Index for BM25
        if metadata_list:
            for id, meta in zip(ids, metadata_list):
                if meta:
                    text = self._extract_text(meta)
                    if text:
                        self._bm25.add_document(id, text)

        return ids

    def delete(self, id: str) -> bool:
        """Delete vector and remove from BM25 index."""
        if super().delete(id):
            self._bm25.remove_document(id)
            return True
        return False

    def keyword_search(self, query: str, k: int = 10,
                       filter: Filter | dict = None) -> list[HybridSearchResult]:
        """
        Pure keyword (BM25) search.

        Args:
            query: Search query text
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of results sorted by BM25 score
        """
        # Convert dict filter
        if isinstance(filter, dict):
            filter = Filter.from_dict(filter)

        # Get more candidates if filtering
        fetch_k = k * 10 if filter else k
        bm25_results = self._bm25.search(query, k=fetch_k)

        results = []
        for doc_id, score in bm25_results:
            metadata = self._metadata.get(doc_id, {})

            # Apply filter
            if filter and not filter.evaluate(metadata):
                continue

            results.append(HybridSearchResult(
                id=doc_id,
                score=score,
                vector_score=0,
                keyword_score=score,
                metadata=metadata
            ))

            if len(results) >= k:
                break

        return results

    def hybrid_search(
        self,
        vector: np.ndarray,
        query_text: str = None,
        k: int = 10,
        alpha: float = 0.5,
        filter: Filter | dict = None,
        include_vectors: bool = False,
        vector_weight: float = None,
        keyword_weight: float = None
    ) -> list[HybridSearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.

        The final score is: alpha * vector_score + (1 - alpha) * keyword_score

        Both scores are normalized to [0, 1] range before combining.

        Args:
            vector: Query vector
            query_text: Query text for keyword search (optional)
            k: Number of results
            alpha: Balance between vector (1.0) and keyword (0.0) search
                   Default 0.5 = equal weight
            filter: Optional metadata filter
            include_vectors: Include vectors in results
            vector_weight: Alternative to alpha (overrides alpha if set)
            keyword_weight: Alternative to alpha (used with vector_weight)

        Returns:
            List of results sorted by combined score
        """
        # Handle weight parameters
        if vector_weight is not None and keyword_weight is not None:
            total = vector_weight + keyword_weight
            alpha = vector_weight / total if total > 0 else 0.5

        # If no query text, fall back to pure vector search
        if not query_text:
            vector_results = self.search(vector, k=k, filter=filter,
                                         include_vectors=include_vectors)
            return [
                HybridSearchResult(
                    id=r.id,
                    score=r.score,
                    vector_score=r.score,
                    keyword_score=0,
                    metadata=r.metadata,
                    vector=r.vector
                )
                for r in vector_results
            ]

        # Convert filter
        if isinstance(filter, dict):
            filter = Filter.from_dict(filter)

        # Get candidates from both searches (fetch more for better coverage)
        fetch_k = k * 5

        # Vector search results
        vector_results = self.search(vector, k=fetch_k, filter=None,
                                     include_vectors=include_vectors)

        # Keyword search results
        keyword_results = self._bm25.search(query_text, k=fetch_k)

        # Normalize scores
        # Vector scores: convert distance to similarity (lower distance = higher similarity)
        vector_scores = {}
        if vector_results:
            max_dist = max(r.score for r in vector_results) or 1
            for r in vector_results:
                # Convert distance to normalized similarity [0, 1]
                vector_scores[r.id] = 1 - (r.score / max_dist) if max_dist > 0 else 1

        # Keyword scores: normalize to [0, 1]
        keyword_scores = {}
        if keyword_results:
            max_score = max(score for _, score in keyword_results) or 1
            for doc_id, score in keyword_results:
                keyword_scores[doc_id] = score / max_score if max_score > 0 else 0

        # Combine candidates
        all_candidates = set(vector_scores.keys()) | set(keyword_scores.keys())

        # Calculate combined scores
        combined = []
        for doc_id in all_candidates:
            v_score = vector_scores.get(doc_id, 0)
            k_score = keyword_scores.get(doc_id, 0)

            # Combined score (higher is better)
            combined_score = alpha * v_score + (1 - alpha) * k_score

            metadata = self._metadata.get(doc_id, {})

            # Apply filter
            if filter and not filter.evaluate(metadata):
                continue

            result = HybridSearchResult(
                id=doc_id,
                score=combined_score,
                vector_score=v_score,
                keyword_score=k_score,
                metadata=metadata
            )

            if include_vectors:
                result.vector = self._vectors.get(doc_id)

            combined.append(result)

        # Sort by combined score (descending - higher is better)
        combined.sort(key=lambda x: x.score, reverse=True)

        return combined[:k]


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import time
    from pathlib import Path
    import shutil

    print("=" * 60)
    print("Hybrid Search Demo")
    print("=" * 60)

    # Clean up previous demo
    demo_path = Path("./hybrid_demo")
    if demo_path.exists():
        shutil.rmtree(demo_path)

    # Create collection with text fields
    config = CollectionConfig(
        name="articles",
        dimensions=128,
        max_elements=10000
    )

    collection = HybridCollection(config, demo_path, text_fields=["title", "content"])

    # Sample documents
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "category": "AI"
        },
        {
            "title": "Deep Learning Neural Networks",
            "content": "Deep learning uses neural networks with many layers to model complex patterns in data.",
            "category": "AI"
        },
        {
            "title": "Natural Language Processing Basics",
            "content": "NLP enables computers to understand, interpret, and generate human language.",
            "category": "NLP"
        },
        {
            "title": "Computer Vision Applications",
            "content": "Computer vision algorithms can analyze images and videos to extract meaningful information.",
            "category": "CV"
        },
        {
            "title": "Reinforcement Learning in Games",
            "content": "RL agents learn to play games by receiving rewards for good actions and penalties for bad ones.",
            "category": "AI"
        },
        {
            "title": "Database Optimization Techniques",
            "content": "Optimizing database queries involves indexing, query planning, and efficient data structures.",
            "category": "DB"
        },
        {
            "title": "Vector Databases for AI",
            "content": "Vector databases store embeddings and enable fast similarity search for AI applications.",
            "category": "DB"
        },
        {
            "title": "Transformer Architecture Explained",
            "content": "Transformers use self-attention mechanisms to process sequential data in parallel.",
            "category": "AI"
        }
    ]

    # Insert documents with random vectors (in practice, use real embeddings)
    print("\nInserting documents...")
    np.random.seed(42)

    for i, doc in enumerate(documents):
        vector = np.random.randn(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        collection.insert(vector, id=f"doc_{i}", metadata=doc)

    print(f"Inserted {len(documents)} documents")

    # Query vector (random for demo)
    query_vector = np.random.randn(128).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    # 1. Pure vector search
    print("\n--- Pure Vector Search ---")
    results = collection.search(query_vector, k=3)
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, title={r.metadata.get('title', '')[:40]}")

    # 2. Pure keyword search
    print("\n--- Pure Keyword Search: 'machine learning' ---")
    results = collection.keyword_search("machine learning", k=3)
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, title={r.metadata.get('title', '')[:40]}")

    # 3. Hybrid search (balanced)
    print("\n--- Hybrid Search: 'machine learning' (alpha=0.5) ---")
    results = collection.hybrid_search(query_vector, "machine learning", k=3, alpha=0.5)
    for r in results:
        print(f"  {r.id}: combined={r.score:.4f}, vec={r.vector_score:.4f}, kw={r.keyword_score:.4f}")
        print(f"         title={r.metadata.get('title', '')[:50]}")

    # 4. Hybrid search (favor keywords)
    print("\n--- Hybrid Search: 'neural networks' (alpha=0.2, favor keywords) ---")
    results = collection.hybrid_search(query_vector, "neural networks", k=3, alpha=0.2)
    for r in results:
        print(f"  {r.id}: combined={r.score:.4f}, vec={r.vector_score:.4f}, kw={r.keyword_score:.4f}")
        print(f"         title={r.metadata.get('title', '')[:50]}")

    # 5. Hybrid search with filter
    print("\n--- Hybrid Search: 'learning' + filter(category='AI') ---")
    results = collection.hybrid_search(
        query_vector,
        "learning",
        k=3,
        alpha=0.5,
        filter={"category": "AI"}
    )
    for r in results:
        print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}")
        print(f"         title={r.metadata.get('title', '')[:50]}")

    # Save
    collection.save()
    print("\n" + "=" * 60)
    print("Demo complete!")
