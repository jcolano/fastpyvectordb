#!/usr/bin/env python3
"""
PyVectorDB RAG Demo - Complete Integration Example
===================================================

This demo shows how to build a complete Retrieval-Augmented Generation (RAG)
application using all PyVectorDB modules:

1. VectorDB (vectordb_optimized.py) - Main vector storage with HNSW indexing
2. Quantization (quantization.py) - Memory compression for large datasets
3. Parallel Search (parallel_search.py) - Multi-core acceleration
4. Graph DB (graph.py) - Knowledge graph for relationships
5. Hybrid Search (hybrid_search.py) - Combined vector + graph search

Architecture:
=============

    ┌─────────────────────────────────────────────────────────────────┐
    │                         RAG Application                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
    │   │   Embedder   │───▶│   VectorDB   │◀───│ Quantization │     │
    │   │  (text→vec)  │    │   (HNSW)     │    │  (compress)  │     │
    │   └──────────────┘    └──────┬───────┘    └──────────────┘     │
    │                              │                                   │
    │                              ▼                                   │
    │   ┌──────────────┐    ┌──────────────┐                         │
    │   │  Graph DB    │◀──▶│Hybrid Search │                         │
    │   │(relationships)│    │(vec + graph) │                         │
    │   └──────────────┘    └──────┬───────┘                         │
    │                              │                                   │
    │                              ▼                                   │
    │   ┌──────────────┐    ┌──────────────┐                         │
    │   │   Parallel   │───▶│   Results    │                         │
    │   │   Search     │    │  (top-k)     │                         │
    │   └──────────────┘    └──────────────┘                         │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Run:
    python rag_demo.py

Dependencies:
    pip install numpy hnswlib
"""

import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import our modules
from vectordb_optimized import VectorDB, Collection, Filter, SearchResult
from quantization import ScalarQuantizer, BinaryQuantizer
from parallel_search import ParallelSearchEngine, MemoryMappedVectors
from graph import GraphDB as Graph, NodeBuilder, EdgeBuilder
from embeddings import MockEmbedder as SimpleEmbedder


# =============================================================================
# Synthetic Data Generator
# =============================================================================

@dataclass
class Document:
    """A document with content and metadata."""
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    author: str
    date: str


def generate_synthetic_corpus(n_docs: int = 1000) -> List[Document]:
    """
    Generate a synthetic corpus of technical documentation.

    Creates documents about:
    - Programming languages (Python, Rust, JavaScript, Go)
    - Databases (PostgreSQL, MongoDB, Redis, Vector DBs)
    - Machine Learning (Neural Networks, Transformers, RAG)
    - DevOps (Docker, Kubernetes, CI/CD)
    """
    np.random.seed(42)

    # Topic templates
    topics = {
        "python": {
            "category": "programming",
            "tags": ["python", "programming", "scripting"],
            "templates": [
                "Python is a versatile programming language known for {feature}. It excels at {use_case} and provides {benefit}.",
                "When working with Python, developers often use {tool} for {purpose}. This enables {outcome}.",
                "The Python ecosystem includes {library} which helps with {task}. Key features include {features}.",
                "Best practices in Python development include {practice}. This ensures {benefit} and improves {metric}.",
            ],
            "features": ["dynamic typing", "clean syntax", "extensive libraries", "rapid prototyping"],
            "use_cases": ["data science", "web development", "automation", "machine learning"],
            "tools": ["pip", "virtualenv", "pytest", "black", "mypy"],
            "libraries": ["NumPy", "Pandas", "FastAPI", "Django", "SQLAlchemy"],
        },
        "rust": {
            "category": "programming",
            "tags": ["rust", "systems", "performance"],
            "templates": [
                "Rust provides memory safety through {feature}. This eliminates {problem} while maintaining {benefit}.",
                "Rust's ownership system ensures {guarantee}. Developers benefit from {advantage} without {tradeoff}.",
                "High-performance applications use Rust for {use_case}. Key benefits include {benefits}.",
                "The Rust compiler catches {error_type} at compile time, preventing {issue} in production.",
            ],
            "features": ["ownership", "borrowing", "lifetimes", "zero-cost abstractions"],
            "problems": ["memory leaks", "data races", "null pointer errors", "buffer overflows"],
            "use_cases": ["systems programming", "WebAssembly", "embedded systems", "game engines"],
        },
        "vector_db": {
            "category": "database",
            "tags": ["vector", "embeddings", "similarity"],
            "templates": [
                "Vector databases store {data_type} for {purpose}. They use {algorithm} for fast similarity search.",
                "HNSW indexing provides {benefit} for vector search. It achieves {performance} with {tradeoff}.",
                "Semantic search uses vector embeddings to find {target}. This enables {capability} beyond keyword matching.",
                "Vector quantization reduces memory by {factor}x while maintaining {metric} recall.",
            ],
            "data_types": ["embeddings", "feature vectors", "semantic representations"],
            "algorithms": ["HNSW", "IVF", "PQ", "LSH"],
            "purposes": ["similarity search", "recommendation", "RAG", "semantic matching"],
        },
        "machine_learning": {
            "category": "ai",
            "tags": ["ml", "neural-networks", "deep-learning"],
            "templates": [
                "Neural networks learn {what} through {how}. They excel at {task} by identifying {patterns}.",
                "Transformers use {mechanism} to process {data}. This architecture enables {capability}.",
                "RAG combines {component1} with {component2} for {purpose}. This improves {metric} significantly.",
                "Fine-tuning adapts {model} for {task} using {data}. Results show {improvement} in performance.",
            ],
            "mechanisms": ["self-attention", "multi-head attention", "positional encoding"],
            "tasks": ["text generation", "classification", "question answering", "translation"],
        },
        "devops": {
            "category": "infrastructure",
            "tags": ["docker", "kubernetes", "cicd"],
            "templates": [
                "Containerization with Docker provides {benefit}. Applications run consistently across {environments}.",
                "Kubernetes orchestrates {what} for {purpose}. It handles {task} automatically.",
                "CI/CD pipelines automate {process}. This reduces {metric} and improves {outcome}.",
                "Infrastructure as Code manages {resource} through {tool}. Changes are {adjective} and reproducible.",
            ],
            "benefits": ["isolation", "portability", "scalability", "reproducibility"],
            "processes": ["testing", "deployment", "rollback", "monitoring"],
        },
    }

    authors = ["Alice Chen", "Bob Smith", "Carol Davis", "David Lee", "Eva Martinez"]
    dates = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in [1, 10, 20]]

    documents = []

    for i in range(n_docs):
        # Pick a random topic
        topic_name = np.random.choice(list(topics.keys()))
        topic = topics[topic_name]

        # Generate content from template
        template = np.random.choice(topic["templates"])

        # Fill in template variables
        content = template
        for key in ["feature", "features", "use_case", "use_cases", "benefit", "benefits",
                    "tool", "tools", "library", "libraries", "purpose", "purposes",
                    "task", "tasks", "outcome", "problem", "problems", "guarantee",
                    "advantage", "tradeoff", "error_type", "issue", "data_type", "data_types",
                    "algorithm", "algorithms", "target", "capability", "factor", "metric",
                    "what", "how", "patterns", "mechanism", "mechanisms", "data",
                    "component1", "component2", "model", "improvement", "environments",
                    "resource", "adjective", "process", "processes"]:
            if "{" + key + "}" in content:
                if key in topic:
                    value = np.random.choice(topic[key]) if isinstance(topic[key], list) else topic[key]
                else:
                    # Default values
                    defaults = {
                        "benefit": "improved performance",
                        "outcome": "better results",
                        "metric": "accuracy",
                        "purpose": "efficient processing",
                        "capability": "advanced functionality",
                        "factor": str(np.random.choice([4, 8, 16, 32])),
                        "what": "complex patterns",
                        "how": "gradient descent",
                        "data": "training examples",
                        "model": "pre-trained model",
                        "improvement": "significant gains",
                        "environments": "development and production",
                        "resource": "infrastructure",
                        "adjective": "version-controlled",
                        "target": "relevant documents",
                        "patterns": "hidden structures",
                        "component1": "retrieval",
                        "component2": "generation",
                        "guarantee": "thread safety",
                        "advantage": "performance",
                        "tradeoff": "garbage collection",
                        "error_type": "memory errors",
                        "issue": "runtime crashes",
                    }
                    value = defaults.get(key, "important aspects")
                content = content.replace("{" + key + "}", value)

        # Create document
        doc = Document(
            id=f"doc_{i:04d}",
            title=f"{topic_name.replace('_', ' ').title()} Guide #{i}",
            content=content,
            category=topic["category"],
            tags=topic["tags"] + [f"doc-{i % 10}"],
            author=np.random.choice(authors),
            date=np.random.choice(dates),
        )
        documents.append(doc)

    return documents


# =============================================================================
# RAG Application
# =============================================================================

class RAGApplication:
    """
    Complete RAG application integrating all PyVectorDB modules.

    Features:
    - Vector search with HNSW indexing
    - Optional quantization for memory efficiency
    - Parallel search for large datasets
    - Knowledge graph for entity relationships
    - Hybrid search combining vectors and graph
    """

    def __init__(
        self,
        db_path: str,
        dimensions: int = 128,
        use_quantization: bool = True,
        use_graph: bool = True,
    ):
        """
        Initialize RAG application.

        Args:
            db_path: Path for database storage
            dimensions: Embedding dimensions
            use_quantization: Enable memory compression
            use_graph: Enable knowledge graph
        """
        self.db_path = Path(db_path)
        self.dimensions = dimensions
        self.use_quantization = use_quantization
        self.use_graph = use_graph

        # Initialize components
        print("Initializing RAG components...")

        # 1. Vector Database (main storage)
        self.db = VectorDB(str(self.db_path / "vectordb"))
        self.collection = self.db.create_collection(
            "documents",
            dimensions=dimensions,
            metric="cosine",
            ef_construction=200,
            M=16,
        )
        print("  ✓ VectorDB initialized")

        # 2. Embedder (text → vectors)
        self.embedder = SimpleEmbedder(dimensions=dimensions)
        print("  ✓ Embedder initialized")

        # 3. Quantization (optional memory compression)
        self.quantizer: Optional[ScalarQuantizer] = None
        self.quantized_vectors: Optional[np.ndarray] = None
        if use_quantization:
            self.quantizer = ScalarQuantizer(dimensions)
            print("  ✓ Quantizer initialized (4x compression)")

        # 4. Parallel Search Engine
        self.search_engine = ParallelSearchEngine()
        print("  ✓ Parallel search engine initialized")

        # 5. Knowledge Graph (optional)
        self.graph: Optional[Graph] = None
        if use_graph:
            self.graph = Graph()
            print("  ✓ Knowledge graph initialized")

        # Document storage
        self.documents: Dict[str, Document] = {}
        self.vectors: Optional[np.ndarray] = None

        print("RAG application ready!\n")

    def index_documents(self, documents: List[Document], show_progress: bool = True):
        """
        Index documents into the RAG system.

        Steps:
        1. Generate embeddings for all documents
        2. Insert into vector database
        3. Build quantized index (if enabled)
        4. Build knowledge graph (if enabled)
        """
        n_docs = len(documents)

        if show_progress:
            print(f"Indexing {n_docs:,} documents...")

        # Store documents
        for doc in documents:
            self.documents[doc.id] = doc

        # 1. Generate embeddings
        start = time.perf_counter()
        texts = [f"{doc.title}. {doc.content}" for doc in documents]
        vectors = self.embedder.embed_batch(texts)
        self.vectors = vectors
        embed_time = time.perf_counter() - start

        if show_progress:
            print(f"  ✓ Generated {n_docs:,} embeddings ({embed_time:.2f}s)")

        # 2. Insert into vector database
        start = time.perf_counter()
        ids = [doc.id for doc in documents]
        metadata_list = [
            {
                "title": doc.title,
                "category": doc.category,
                "author": doc.author,
                "date": doc.date,
                "tags": ",".join(doc.tags),
            }
            for doc in documents
        ]
        self.collection.insert_batch(vectors, ids, metadata_list)
        insert_time = time.perf_counter() - start

        if show_progress:
            print(f"  ✓ Indexed in VectorDB ({insert_time:.2f}s, {n_docs/insert_time:,.0f} docs/sec)")

        # 3. Build quantized index (if enabled)
        if self.quantizer:
            start = time.perf_counter()
            self.quantizer.train(vectors)
            self.quantized_vectors = self.quantizer.encode(vectors)
            quant_time = time.perf_counter() - start

            memory_original = vectors.nbytes / 1024 / 1024
            memory_quantized = self.quantized_vectors.nbytes / 1024 / 1024

            if show_progress:
                print(f"  ✓ Built quantized index ({quant_time:.2f}s)")
                print(f"    Memory: {memory_original:.1f} MB → {memory_quantized:.1f} MB "
                      f"({memory_original/memory_quantized:.1f}x compression)")

        # 4. Build knowledge graph (if enabled)
        if self.graph:
            start = time.perf_counter()
            self._build_knowledge_graph(documents)
            graph_time = time.perf_counter() - start

            if show_progress:
                print(f"  ✓ Built knowledge graph ({graph_time:.2f}s)")
                print(f"    Nodes: {len(self.graph._nodes):,}, Edges: {len(self.graph._edges):,}")

        if show_progress:
            print(f"\nIndexing complete!")

    def _build_knowledge_graph(self, documents: List[Document]):
        """Build knowledge graph from documents."""
        # Create category nodes
        categories = set(doc.category for doc in documents)
        for cat in categories:
            self.graph.create_node(
                NodeBuilder(f"cat_{cat}")
                .label("Category")
                .property("name", cat)
                .build()
            )

        # Create author nodes
        authors = set(doc.author for doc in documents)
        for author in authors:
            self.graph.create_node(
                NodeBuilder(f"author_{author.replace(' ', '_')}")
                .label("Author")
                .property("name", author)
                .build()
            )

        # Create tag nodes
        all_tags = set()
        for doc in documents:
            all_tags.update(doc.tags)
        for tag in all_tags:
            self.graph.create_node(
                NodeBuilder(f"tag_{tag}")
                .label("Tag")
                .property("name", tag)
                .build()
            )

        # Create document nodes and relationships
        for doc in documents:
            # Document node
            self.graph.create_node(
                NodeBuilder(doc.id)
                .label("Document")
                .property("title", doc.title)
                .property("date", doc.date)
                .build()
            )

            # Document → Category
            self.graph.create_edge(
                EdgeBuilder(doc.id, f"cat_{doc.category}", "BELONGS_TO")
                .build()
            )

            # Document → Author
            self.graph.create_edge(
                EdgeBuilder(doc.id, f"author_{doc.author.replace(' ', '_')}", "WRITTEN_BY")
                .build()
            )

            # Document → Tags
            for tag in doc.tags:
                self.graph.create_edge(
                    EdgeBuilder(doc.id, f"tag_{tag}", "TAGGED_WITH")
                    .build()
                )

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hnsw",
        filter_dict: Dict = None,
        expand_graph: bool = False,
    ) -> List[Dict]:
        """
        Search for relevant documents.

        Args:
            query: Search query text
            k: Number of results
            method: Search method ("hnsw", "quantized", "parallel", "hybrid")
            filter_dict: Metadata filter (e.g., {"category": "programming"})
            expand_graph: Include graph-related documents

        Returns:
            List of results with document info and scores
        """
        # Embed query
        query_vector = self.embedder.embed(query)

        results = []

        if method == "hnsw":
            # Standard HNSW search
            filter_obj = Filter.from_dict(filter_dict) if filter_dict else None
            search_results = self.collection.search(
                query_vector, k=k, filter=filter_obj
            )
            results = self._format_results(search_results)

        elif method == "quantized" and self.quantizer:
            # Search using quantized vectors
            query_quantized = self.quantizer.encode(query_vector.reshape(1, -1))
            distances = self.quantizer.distances_l2(
                query_vector, self.quantized_vectors
            )

            # Get top-k
            top_k_idx = np.argpartition(distances, k)[:k]
            top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

            ids = list(self.documents.keys())
            results = [
                {
                    "id": ids[idx],
                    "score": float(distances[idx]),
                    "document": self.documents[ids[idx]],
                }
                for idx in top_k_idx
            ]

        elif method == "parallel":
            # Parallel brute-force search
            parallel_results = self.search_engine.search_parallel(
                query_vector, self.vectors, k=k, metric="cosine"
            )

            ids = list(self.documents.keys())
            results = [
                {
                    "id": ids[r.index],
                    "score": r.distance,
                    "document": self.documents[ids[r.index]],
                }
                for r in parallel_results
            ]

        elif method == "hybrid" and self.graph:
            # HNSW search + graph expansion
            search_results = self.collection.search(query_vector, k=k)
            results = self._format_results(search_results)

            if expand_graph and results:
                # Find related documents through graph
                related_ids = set()
                for r in results:
                    doc_id = r["id"]

                    # Get documents by same author
                    neighbors = self.graph.neighbors(doc_id, direction="out")
                    for neighbor in neighbors:
                        if neighbor.labels and "Author" in neighbor.labels:
                            author_docs = self.graph.neighbors(neighbor.id, direction="in")
                            for ad in author_docs:
                                if ad.id != doc_id and ad.id in self.documents:
                                    related_ids.add(ad.id)

                    # Get documents with same tags
                    for neighbor in neighbors:
                        if neighbor.labels and "Tag" in neighbor.labels:
                            tag_docs = self.graph.neighbors(neighbor.id, direction="in")
                            for td in tag_docs:
                                if td.id != doc_id and td.id in self.documents:
                                    related_ids.add(td.id)

                # Add related documents
                existing_ids = {r["id"] for r in results}
                for rel_id in list(related_ids)[:k]:
                    if rel_id not in existing_ids:
                        results.append({
                            "id": rel_id,
                            "score": 0.5,  # Lower score for graph-expanded results
                            "document": self.documents[rel_id],
                            "source": "graph_expansion",
                        })

        return results

    def _format_results(self, search_results: List[SearchResult]) -> List[Dict]:
        """Format search results with document info."""
        return [
            {
                "id": r.id,
                "score": r.score,
                "document": self.documents.get(r.id),
                "metadata": r.metadata,
            }
            for r in search_results
            if r.id in self.documents
        ]

    def benchmark(self, n_queries: int = 100):
        """
        Benchmark different search methods.
        """
        print(f"\n{'='*60}")
        print("  SEARCH BENCHMARK")
        print(f"{'='*60}")
        print(f"Documents: {len(self.documents):,}")
        print(f"Queries: {n_queries}")

        # Generate random queries
        np.random.seed(123)
        sample_docs = np.random.choice(list(self.documents.values()), min(n_queries, len(self.documents)), replace=False)
        queries = [f"{doc.title}" for doc in sample_docs]

        methods = ["hnsw", "parallel"]
        if self.quantizer:
            methods.append("quantized")

        for method in methods:
            times = []
            for q in queries:
                start = time.perf_counter()
                _ = self.search(q, k=5, method=method)
                times.append(time.perf_counter() - start)

            avg_time = np.mean(times) * 1000
            p99_time = np.percentile(times, 99) * 1000

            print(f"\n{method.upper()}:")
            print(f"  Avg latency: {avg_time:.2f} ms")
            print(f"  P99 latency: {p99_time:.2f} ms")
            print(f"  QPS: {1000/avg_time:.0f}")


# =============================================================================
# Demo Application
# =============================================================================

def run_demo():
    """Run the complete RAG demo."""
    print("=" * 70)
    print("  PyVectorDB RAG Demo - Complete Integration Example")
    print("=" * 70)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temp directory: {temp_dir}")

    try:
        # 1. Generate synthetic corpus
        print("\n" + "-" * 60)
        print("Step 1: Generating Synthetic Corpus")
        print("-" * 60)

        n_docs = 2000
        documents = generate_synthetic_corpus(n_docs)

        print(f"Generated {n_docs:,} documents")
        print(f"\nSample document:")
        print(f"  ID: {documents[0].id}")
        print(f"  Title: {documents[0].title}")
        print(f"  Category: {documents[0].category}")
        print(f"  Tags: {documents[0].tags}")
        print(f"  Content: {documents[0].content[:100]}...")

        # 2. Initialize RAG application
        print("\n" + "-" * 60)
        print("Step 2: Initializing RAG Application")
        print("-" * 60)

        rag = RAGApplication(
            db_path=temp_dir,
            dimensions=128,
            use_quantization=True,
            use_graph=True,
        )

        # 3. Index documents
        print("\n" + "-" * 60)
        print("Step 3: Indexing Documents")
        print("-" * 60)

        rag.index_documents(documents)

        # 4. Demo searches
        print("\n" + "-" * 60)
        print("Step 4: Demo Searches")
        print("-" * 60)

        demo_queries = [
            ("Python machine learning libraries", "hnsw", None),
            ("Rust memory safety ownership", "hnsw", None),
            ("Vector database HNSW indexing", "quantized", None),
            ("Docker container orchestration", "parallel", None),
            ("Programming best practices", "hnsw", {"category": "programming"}),
            ("Neural network transformers", "hybrid", None),
        ]

        for query, method, filter_dict in demo_queries:
            print(f"\nQuery: \"{query}\"")
            print(f"Method: {method}" + (f", Filter: {filter_dict}" if filter_dict else ""))

            start = time.perf_counter()
            results = rag.search(
                query, k=3, method=method,
                filter_dict=filter_dict,
                expand_graph=(method == "hybrid")
            )
            latency = (time.perf_counter() - start) * 1000

            print(f"Results ({latency:.2f}ms):")
            for i, r in enumerate(results[:3], 1):
                doc = r.get("document")
                if doc:
                    source = r.get("source", "vector_search")
                    print(f"  {i}. [{r['score']:.4f}] {doc.title}")
                    print(f"     Category: {doc.category} | Source: {source}")

        # 5. Benchmark
        print("\n" + "-" * 60)
        print("Step 5: Performance Benchmark")
        print("-" * 60)

        rag.benchmark(n_queries=50)

        # 6. Memory usage
        print("\n" + "-" * 60)
        print("Step 6: Memory Usage Summary")
        print("-" * 60)

        vector_memory = rag.vectors.nbytes / 1024 / 1024
        quantized_memory = rag.quantized_vectors.nbytes / 1024 / 1024 if rag.quantized_vectors is not None else 0

        print(f"\nMemory Usage:")
        print(f"  Original vectors: {vector_memory:.2f} MB")
        print(f"  Quantized vectors: {quantized_memory:.2f} MB")
        print(f"  Compression ratio: {vector_memory/quantized_memory:.1f}x" if quantized_memory > 0 else "")
        print(f"  Documents stored: {len(rag.documents):,}")
        if rag.graph:
            print(f"  Graph nodes: {len(rag.graph._nodes):,}")
            print(f"  Graph edges: {len(rag.graph._edges):,}")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)


# =============================================================================
# API Examples
# =============================================================================

def show_api_examples():
    """Print API usage examples."""
    print("""
================================================================================
  PyVectorDB API Quick Reference
================================================================================

1. BASIC VECTOR DATABASE
------------------------
from vectordb_optimized import VectorDB, Filter

# Create database
db = VectorDB("./my_database")
collection = db.create_collection("docs", dimensions=384)

# Insert vectors
collection.insert(vector, id="doc1", metadata={"category": "tech"})
collection.insert_batch(vectors, ids, metadata_list)

# Search
results = collection.search(query_vector, k=10)
results = collection.search(query_vector, k=10, filter=Filter.eq("category", "tech"))

# Filter operations
Filter.eq("field", value)      # Equal
Filter.gt("field", value)      # Greater than
Filter.in_("field", [values])  # In list
Filter.and_([f1, f2])          # AND
Filter.or_([f1, f2])           # OR

2. QUANTIZATION (Memory Compression)
------------------------------------
from quantization import ScalarQuantizer, BinaryQuantizer

# Scalar: 4x compression, 97%+ recall
sq = ScalarQuantizer(dimensions=384)
sq.train(vectors)
quantized = sq.encode(vectors)
distances = sq.distances_l2(query, quantized)

# Binary: 32x compression, fast hamming
bq = BinaryQuantizer(dimensions=384)
bq.train(vectors)
binary = bq.encode(vectors)
distances = bq.distances_hamming(query, binary)

3. PARALLEL SEARCH (Multi-core)
-------------------------------
from parallel_search import ParallelSearchEngine, MemoryMappedVectors

# Parallel brute-force search
engine = ParallelSearchEngine(n_workers=8)
results = engine.search_parallel(query, vectors, k=10, metric="cosine")

# Batch search (GEMM optimized)
all_results = engine.search_batch_parallel(queries, vectors, k=10)

# Memory-mapped for large datasets
mmap = MemoryMappedVectors("./large_data", dimensions=384)
mmap.create(n_vectors=100_000_000)
mmap.append_batch(vectors)
results = mmap.search_parallel(query, k=10)

4. KNOWLEDGE GRAPH
------------------
from graph import GraphDB as Graph, NodeBuilder, EdgeBuilder

graph = Graph()

# Create nodes
graph.create_node(NodeBuilder("user1").label("User").property("name", "Alice"))
graph.create_node(NodeBuilder("doc1").label("Document").property("title", "Guide"))

# Create edges
graph.create_edge(EdgeBuilder("user1", "doc1", "AUTHORED"))

# Query
neighbors = graph.neighbors("user1", direction="out")
path = graph.traverse("user1", max_depth=3)

5. HYBRID SEARCH (Vector + Graph)
---------------------------------
from hybrid_search import HybridSearch

hybrid = HybridSearch(collection, graph)
results = hybrid.search(
    query_vector,
    k=10,
    graph_boost=0.3,  # Weight for graph relationships
    expand_neighbors=True,
)

6. COMPLETE RAG APPLICATION
---------------------------
from rag_demo import RAGApplication

rag = RAGApplication(
    db_path="./rag_db",
    dimensions=384,
    use_quantization=True,
    use_graph=True,
)

# Index documents
rag.index_documents(documents)

# Search with different methods
results = rag.search("query", method="hnsw")      # Fast approximate
results = rag.search("query", method="quantized") # Memory efficient
results = rag.search("query", method="parallel")  # Exact search
results = rag.search("query", method="hybrid")    # Vector + graph

================================================================================
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        show_api_examples()
    else:
        run_demo()
