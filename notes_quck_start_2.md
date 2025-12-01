# PyVectorDB Quick Start Guide

A high-performance Python vector database with HNSW indexing, quantization, parallel processing, and knowledge graph support.

## Table of Contents

- [Installation](#installation)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Integration Patterns](#integration-patterns)
- [Performance Tuning](#performance-tuning)

## Installation

```bash
# Required
pip install numpy hnswlib

# Optional (for advanced features)
pip install sentence-transformers  # Real embeddings
pip install openai                 # OpenAI embeddings
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PyVectorDB System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  vectordb_       │    │   quantization   │                   │
│  │  optimized.py    │    │      .py         │                   │
│  │  ──────────────  │    │  ──────────────  │                   │
│  │  • VectorDB      │    │  • Scalar (4x)   │                   │
│  │  • Collection    │    │  • Binary (32x)  │                   │
│  │  • HNSW Index    │    │  • Product (8x)  │                   │
│  │  • Filters       │    │                  │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────────────────────────────┐                   │
│  │           parallel_search.py              │                   │
│  │  ────────────────────────────────────────│                   │
│  │  • ParallelSearchEngine (BLAS/GEMM)      │                   │
│  │  • MemoryMappedVectors (>RAM datasets)   │                   │
│  │  • ConcurrentHNSWSearcher                │                   │
│  └────────────────────┬─────────────────────┘                   │
│                       │                                          │
│           ┌───────────┴───────────┐                             │
│           ▼                       ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │    graph.py      │    │  hybrid_search   │                   │
│  │  ──────────────  │    │      .py         │                   │
│  │  • GraphDB       │    │  ──────────────  │                   │
│  │  • Nodes/Edges   │    │  Vector + Graph  │                   │
│  │  • Traversal     │    │  Combined Search │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Vector Database

```python
from vectordb_optimized import VectorDB, Filter
import numpy as np

# Create database
db = VectorDB("./my_database")
collection = db.create_collection("documents", dimensions=384)

# Insert vectors
vector = np.random.randn(384).astype(np.float32)
collection.insert(vector, id="doc1", metadata={"category": "tech", "author": "Alice"})

# Batch insert (faster)
vectors = np.random.randn(1000, 384).astype(np.float32)
ids = [f"doc_{i}" for i in range(1000)]
metadata_list = [{"category": "tech"} for _ in range(1000)]
collection.insert_batch(vectors, ids, metadata_list)

# Search
query = np.random.randn(384).astype(np.float32)
results = collection.search(query, k=10)

for r in results:
    print(f"ID: {r.id}, Score: {r.score:.4f}")

# Filtered search
results = collection.search(
    query,
    k=10,
    filter=Filter.eq("category", "tech")
)

# Save to disk
db.save()
```

### 2. Memory Compression with Quantization

```python
from quantization import ScalarQuantizer, BinaryQuantizer
import numpy as np

vectors = np.random.randn(100000, 384).astype(np.float32)

# Scalar Quantization: 4x compression, 97%+ recall
sq = ScalarQuantizer(dimensions=384)
sq.train(vectors)
quantized = sq.encode(vectors)  # uint8 instead of float32

print(f"Original: {vectors.nbytes / 1e6:.1f} MB")
print(f"Quantized: {quantized.nbytes / 1e6:.1f} MB")

# Search with quantized vectors
query = np.random.randn(384).astype(np.float32)
distances = sq.distances_l2(query, quantized)
top_k = np.argpartition(distances, 10)[:10]

# Binary Quantization: 32x compression, ultra-fast hamming distance
bq = BinaryQuantizer(dimensions=384)
bq.train(vectors)
binary = bq.encode(vectors)  # 1-bit per dimension

distances = bq.distances_hamming(query, binary)
```

### 3. Parallel Processing for Large Datasets

```python
from parallel_search import ParallelSearchEngine, MemoryMappedVectors
import numpy as np

vectors = np.random.randn(1000000, 128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

# Parallel search with BLAS (67x faster than naive)
engine = ParallelSearchEngine(n_workers=8)
results = engine.search_parallel(query, vectors, k=10, metric="cosine")

# Batch search with GEMM (2x faster for multiple queries)
queries = np.random.randn(100, 128).astype(np.float32)
all_results = engine.search_batch_parallel(queries, vectors, k=10)

# Memory-mapped for datasets larger than RAM
mmap = MemoryMappedVectors("./large_dataset", dimensions=128)
mmap.create(n_vectors=100_000_000)  # 100M vectors
mmap.append_batch(vectors)
results = mmap.search_parallel(query, k=10, engine=engine)
```

### 4. Knowledge Graph

```python
from graph import GraphDB, NodeBuilder, EdgeBuilder

graph = GraphDB()

# Create nodes
graph.create_node(
    NodeBuilder("user_1")
    .label("User")
    .property("name", "Alice")
    .property("role", "engineer")
    .build()
)

graph.create_node(
    NodeBuilder("doc_1")
    .label("Document")
    .property("title", "Vector DB Guide")
    .build()
)

# Create relationships
graph.create_edge(
    EdgeBuilder("user_1", "doc_1", "AUTHORED")
    .property("date", "2024-01-15")
    .build()
)

# Query neighbors
neighbors = graph.neighbors("user_1", direction="out")
for node in neighbors:
    print(f"Connected to: {node.id}")

# Traverse graph
path = graph.traverse("user_1", max_depth=3)
```

### 5. Complete RAG Application

```python
from rag_demo import RAGApplication

# Initialize with all features
rag = RAGApplication(
    db_path="./rag_database",
    dimensions=384,
    use_quantization=True,  # 4x memory savings
    use_graph=True,         # Knowledge graph
)

# Index documents
documents = [...]  # Your documents
rag.index_documents(documents)

# Search with different methods
results = rag.search("machine learning", method="hnsw")       # Fast approximate
results = rag.search("machine learning", method="quantized")  # Memory efficient
results = rag.search("machine learning", method="parallel")   # Exact brute-force
results = rag.search("machine learning", method="hybrid")     # Vector + graph

# Filtered search
results = rag.search(
    "Python programming",
    method="hnsw",
    filter_dict={"category": "programming"}
)
```

## Module Reference

### vectordb_optimized.py

| Class | Description |
|-------|-------------|
| `VectorDB` | Multi-collection database manager |
| `Collection` | Single collection with HNSW index |
| `Filter` | Metadata filtering (eq, gt, in_, and_, or_) |
| `SearchResult` | Search result with id, score, metadata |

**Key Methods:**
```python
# VectorDB
db.create_collection(name, dimensions, metric="cosine")
db.get_collection(name)
db.list_collections()
db.save()

# Collection
collection.insert(vector, id, metadata)
collection.insert_batch(vectors, ids, metadata_list)
collection.search(query, k=10, filter=None)
collection.search_batch(queries, k=10)
collection.get(id, include_vector=False)
collection.delete(id)
```

### quantization.py

| Class | Compression | Recall | Speed |
|-------|-------------|--------|-------|
| `ScalarQuantizer` | 4x | 97%+ | Moderate |
| `BinaryQuantizer` | 32x | ~5% | Very Fast |
| `ProductQuantizer` | 8-16x | ~90% | Fast |

**Key Methods:**
```python
quantizer.train(vectors)           # Learn quantization parameters
quantizer.encode(vectors)          # Compress vectors
quantizer.distances_l2(query, db)  # Compute distances
```

### parallel_search.py

| Class | Use Case |
|-------|----------|
| `ParallelSearchEngine` | Multi-core brute-force search |
| `MemoryMappedVectors` | Datasets larger than RAM |
| `ConcurrentHNSWSearcher` | Parallel HNSW queries |

**Key Methods:**
```python
engine.search_parallel(query, vectors, k, metric)
engine.search_batch_parallel(queries, vectors, k)
engine.search_chunked_parallel(query, vectors, k)  # Very large datasets

mmap.create(n_vectors, dimensions)
mmap.append_batch(vectors)
mmap.search_parallel(query, k, engine)
```

### graph.py

| Class | Description |
|-------|-------------|
| `GraphDB` | Main graph database |
| `NodeBuilder` | Fluent node creation |
| `EdgeBuilder` | Fluent edge creation |
| `Node` | Graph node with labels and properties |
| `Edge` | Directed edge with type and properties |

**Key Methods:**
```python
graph.create_node(NodeBuilder(...).build())
graph.create_edge(EdgeBuilder(...).build())
graph.neighbors(node_id, direction="out")  # "in", "out", "both"
graph.traverse(start, max_depth=3)
graph.query("MATCH (n:User) RETURN n")
```

## Integration Patterns

### Pattern 1: HNSW + Quantization (Memory Efficient)

```python
from vectordb_optimized import VectorDB
from quantization import ScalarQuantizer

# Store vectors in HNSW for fast search
db = VectorDB("./db")
collection = db.create_collection("docs", dimensions=384)
collection.insert_batch(vectors, ids)

# Also keep quantized copy for memory efficiency
sq = ScalarQuantizer(384)
sq.train(vectors)
quantized = sq.encode(vectors)

# Use HNSW for candidate generation, quantized for re-ranking
candidates = collection.search(query, k=100)
# Re-rank candidates with quantized distances...
```

### Pattern 2: HNSW + Graph (Hybrid Search)

```python
from vectordb_optimized import VectorDB
from graph import GraphDB, NodeBuilder, EdgeBuilder

db = VectorDB("./db")
collection = db.create_collection("docs", dimensions=384)
graph = GraphDB()

# Index document
collection.insert(vector, id="doc1", metadata={"author": "alice"})
graph.create_node(NodeBuilder("doc1").label("Document").build())
graph.create_edge(EdgeBuilder("doc1", "user_alice", "AUTHORED").build())

# Hybrid search: vector similarity + graph expansion
vector_results = collection.search(query, k=10)
for r in vector_results:
    related = graph.neighbors(r.id, direction="both")
    # Include related documents in results
```

### Pattern 3: Memory-Mapped + Parallel (Massive Datasets)

```python
from parallel_search import ParallelSearchEngine, MemoryMappedVectors

# Create memory-mapped storage for 100M vectors
mmap = MemoryMappedVectors("./massive_dataset", dimensions=128)
mmap.create(n_vectors=100_000_000)

# Stream data in batches
for batch in data_stream:
    mmap.append_batch(batch)

# Search with parallel engine
engine = ParallelSearchEngine(n_workers=16)
results = mmap.search_parallel(query, k=10, engine=engine)
```

## Performance Tuning

### HNSW Parameters

| Parameter | Default | Description | Tradeoff |
|-----------|---------|-------------|----------|
| `M` | 16 | Connections per node | Higher = better recall, more memory |
| `ef_construction` | 200 | Build quality | Higher = better index, slower build |
| `ef_search` | 50 | Search quality | Higher = better recall, slower search |

```python
collection = db.create_collection(
    "docs",
    dimensions=384,
    M=32,               # More connections
    ef_construction=400, # Better index quality
)
collection.set_ef_search(100)  # Higher recall at search time
```

### Quantization Selection

| Use Case | Quantizer | Why |
|----------|-----------|-----|
| Production (balanced) | Scalar | 4x compression, 97%+ recall |
| Ultra-fast filtering | Binary | 32x compression, very fast |
| Research/Analytics | Product | Good compression/recall balance |

### Memory vs Speed

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100K vectors | HNSW only |
| 100K - 1M | HNSW + Scalar quantization |
| 1M - 10M | Memory-mapped + HNSW |
| > 10M | Memory-mapped + Binary quantization + HNSW candidates |

## Running the Demo

```bash
# Full RAG demo
python rag_demo.py

# API reference
python rag_demo.py --api

# Performance benchmarks
python benchmark_parallel.py
python benchmark_quantization.py
```

## Benchmarks

Performance on 100K vectors, 128 dimensions:

| Method | Latency | QPS | Memory |
|--------|---------|-----|--------|
| Naive Python | 450 ms | 2 | 48 MB |
| Vectorized BLAS | 6 ms | 167 | 48 MB |
| HNSW | 0.17 ms | 5,773 | 48 MB |
| Scalar Quantized | 6 ms | 167 | 12 MB |
| Binary Quantized | 0.8 ms | 1,250 | 1.5 MB |

## Next Steps

1. See `rag_demo.py` for a complete application example
2. Run benchmarks to understand performance characteristics
3. Choose the right combination of modules for your use case
4. Integrate with your embedding model (OpenAI, Sentence Transformers, etc.)
