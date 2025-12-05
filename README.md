# FastPyVectorDB

A high-performance Python vector database with a simple, ChromaDB-like API. Features HNSW indexing, multiple embedding providers, quantization, knowledge graphs, and more.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Installation

```bash
# Install with pip (from source)
pip install -e .

# With local embeddings (recommended - no API key needed)
pip install -e ".[local]"

# With OpenAI embeddings
pip install -e ".[openai]"

# With all optional dependencies
pip install -e ".[all]"
```

**Quick install core dependencies only:**
```bash
pip install numpy hnswlib sentence-transformers
```

## Quick Start

FastPyVectorDB provides a simple, intuitive API similar to ChromaDB:

```python
import fastpyvectordb

# Create a client
client = fastpyvectordb.Client()

# Create a collection (uses local embeddings by default)
collection = client.create_collection("my_documents")

# Add documents - embeddings are generated automatically
collection.add(
    documents=[
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing helps computers understand text"
    ],
    ids=["ml", "dl", "nlp"],
    metadatas=[
        {"category": "AI", "level": "beginner"},
        {"category": "AI", "level": "advanced"},
        {"category": "NLP", "level": "intermediate"}
    ]
)

# Search with natural language
results = collection.query(
    query_texts="What is AI?",
    n_results=3
)

# Print results
for doc, score in zip(results.documents[0], results.distances[0]):
    print(f"Score: {score:.4f} - {doc}")
```

## Core API

### Client Operations

```python
import fastpyvectordb

# Create client (data stored in ./vectordb by default)
client = fastpyvectordb.Client(path="./my_data")

# Create a new collection
collection = client.create_collection("documents")

# Get existing collection
collection = client.get_collection("documents")

# Get or create (safe for repeated calls)
collection = client.get_or_create_collection("documents")

# List all collections
print(client.list_collections())

# Delete a collection
client.delete_collection("documents")

# Save all data to disk
client.persist()
```

### Collection Operations

```python
# Add documents
collection.add(
    documents=["Hello world", "Goodbye world"],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "greeting"}, {"source": "farewell"}]
)

# Add with pre-computed embeddings
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ids=["doc1", "doc2"]
)

# Upsert (add or update)
collection.upsert(
    documents=["Updated document"],
    ids=["doc1"]
)

# Query/Search
results = collection.query(
    query_texts="search query",    # or query_embeddings=[...]
    n_results=10,
    where={"category": "tech"}     # optional filter
)

# Access results
print(results.ids)        # [["id1", "id2", ...]]
print(results.documents)  # [["doc1", "doc2", ...]]
print(results.distances)  # [[0.1, 0.2, ...]]
print(results.metadatas)  # [[{...}, {...}, ...]]

# Get by ID
result = collection.get(ids=["doc1", "doc2"])
result = collection.get(where={"category": "tech"}, limit=10)

# Update existing documents
collection.update(
    ids=["doc1"],
    documents=["New content"],
    metadatas=[{"version": 2}]
)

# Delete documents
collection.delete(ids=["doc1", "doc2"])
collection.delete(where={"category": "old"})

# Get count
print(f"Documents: {collection.count}")

# Peek at sample
sample = collection.peek(limit=5)
```

### Using Different Embedding Models

```python
# Local embeddings (default) - no API key needed
collection = client.create_collection(
    name="docs",
    embedding_model="all-MiniLM-L6-v2"  # Fast, 384 dimensions
)

# Higher quality local model
collection = client.create_collection(
    name="docs",
    embedding_model="all-mpnet-base-v2"  # Better quality, 768 dimensions
)

# OpenAI embeddings (requires OPENAI_API_KEY)
collection = client.create_collection(
    name="docs",
    embedding_model="text-embedding-3-small",
    embedding_provider="openai"
)

# Cohere embeddings (requires COHERE_API_KEY)
collection = client.create_collection(
    name="docs",
    embedding_model="embed-english-v3.0",
    embedding_provider="cohere"
)
```

### Filtering

```python
# Simple equality filter
results = collection.query(
    query_texts="search",
    where={"category": "tech"}
)

# Multiple conditions (AND)
results = collection.query(
    query_texts="search",
    where={"category": "tech", "year": 2024}
)

# Using Filter class for complex queries
from fastpyvectordb import Filter

results = collection.query(
    query_texts="search",
    filter=Filter.and_([
        Filter.eq("category", "tech"),
        Filter.gte("score", 0.8),
        Filter.in_("status", ["published", "draft"])
    ])
)
```

## Examples

See the `examples/` directory for complete examples:

```bash
# Run quickstart examples
python examples/quickstart.py

# RAG application demo
python examples/rag_demo.py

# News intelligence demo
python examples/news_intelligence_demo.py
```

---

## Advanced Usage

For advanced features like quantization, parallel search, and knowledge graphs, see below.

## Features

- **Simple API** — ChromaDB-like interface for easy adoption
- **Multiple Embeddings** — Local (Sentence Transformers), OpenAI, Cohere
- **HNSW Indexing** — Sub-millisecond approximate nearest neighbor search
- **Metadata Filtering** — Filter by any metadata field
- **Quantization** — 4-32x memory compression with scalar, binary, and product quantizers
- **Parallel Search** — Multi-core BLAS/GEMM acceleration (67x speedup)
- **Knowledge Graph** — Nodes, edges, traversal, and Cypher-like queries
- **Hybrid Search** — Combined vector similarity + graph relationships
- **REST API** — FastAPI server with WebSocket real-time updates
- **Persistence** — Save/load to disk with automatic recovery

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastPyVectorDB System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  fastpyvectordb/│    │   quantization   │                  │
│  │  client.py       │    │      .py         │                  │
│  │  ──────────────  │    │  ──────────────  │                  │
│  │  • Client        │    │  • Scalar (4x)   │                  │
│  │  • Collection    │    │  • Binary (32x)  │                  │
│  │  • Simple API    │    │  • Product (8x)  │                  │
│  │  • Auto-embed    │    │                  │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌──────────────────────────────────────────┐                  │
│  │       vectordb_optimized.py              │                  │
│  │  ────────────────────────────────────────│                  │
│  │  • VectorDB / Collection (Core)          │                  │
│  │  • HNSW Index                            │                  │
│  │  • Filter Engine                         │                  │
│  └────────────────────┬─────────────────────┘                  │
│                       │                                         │
│           ┌───────────┴───────────┐                            │
│           ▼                       ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │    graph.py      │    │  parallel_search │                  │
│  │  ──────────────  │    │      .py         │                  │
│  │  • GraphDB       │    │  ──────────────  │                  │
│  │  • Nodes/Edges   │    │  • Multi-core    │                  │
│  │  • Traversal     │    │  • Memory-mapped │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Low-Level API

For more control, you can use the low-level API directly:

### Basic Vector Database

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
    query, k=10,
    filter=Filter.eq("category", "tech")
)

# Save to disk
db.save()
```

### Memory Compression with Quantization

```python
from quantization import ScalarQuantizer, BinaryQuantizer
import numpy as np

vectors = np.random.randn(100000, 384).astype(np.float32)

# Scalar Quantization: 4x compression, 97%+ recall
sq = ScalarQuantizer(dimensions=384)
sq.train(vectors)
quantized = sq.encode(vectors)

print(f"Original: {vectors.nbytes / 1e6:.1f} MB")
print(f"Quantized: {quantized.nbytes / 1e6:.1f} MB")

# Search with quantized vectors
query = np.random.randn(384).astype(np.float32)
distances = sq.distances_l2(query, quantized)
top_k = np.argpartition(distances, 10)[:10]

# Binary Quantization: 32x compression, ultra-fast hamming distance
bq = BinaryQuantizer(dimensions=384)
bq.train(vectors)
binary = bq.encode(vectors)
distances = bq.distances_hamming(query, binary)
```

### Parallel Processing for Large Datasets

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
mmap.create(n_vectors=100_000_000)
mmap.append_batch(vectors)
results = mmap.search_parallel(query, k=10, engine=engine)
```

### Knowledge Graph

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
paths = graph.traverse("user_1", max_depth=3)
```

### Embeddings Integration

```python
from embeddings import get_embedder, EmbeddingCollection
from vectordb_optimized import VectorDB

# Auto-detect embedder (uses OPENAI_API_KEY if set, else local)
embedder = get_embedder()

# Or specify provider
embedder = get_embedder("openai", model="text-embedding-3-small")
embedder = get_embedder("sentence-transformers", model="all-MiniLM-L6-v2")

# Create embedding-aware collection
db = VectorDB("./my_db")
collection = db.create_collection("docs", dimensions=embedder.dimensions)
docs = EmbeddingCollection(collection, embedder)

# Insert with automatic embedding
docs.add("doc1", "Machine learning is fascinating", {"category": "AI"})

# Search with text query
results = docs.search("artificial intelligence", k=5)
```

### REST API Server

```bash
# Start server
uvicorn server:app --reload --port 8000

# Or run directly
python server.py
```

```python
# Client usage
from client import VectorDBClient

client = VectorDBClient("http://localhost:8000")

# Create collection
client.create_collection("docs", dimensions=384)

# Insert
client.insert("docs", vector=[0.1, 0.2, ...], metadata={"title": "Hello"})

# Search
results = client.search("docs", vector=[0.1, 0.2, ...], k=10)
```

## Module Reference

| Module | Description |
|--------|-------------|
| `fastpyvectordb/` | High-level client API (recommended) |
| `vectordb_optimized.py` | Core vector database with HNSW indexing |
| `quantization.py` | Scalar, binary, and product quantization |
| `parallel_search.py` | Multi-core search engine and memory-mapped vectors |
| `graph.py` | Knowledge graph with nodes, edges, and traversal |
| `hybrid_search.py` | Combined vector + graph search |
| `embeddings.py` | OpenAI, Sentence Transformers, Cohere integration |
| `server.py` | FastAPI REST server |
| `client.py` | Python HTTP client |
| `realtime.py` | WebSocket real-time subscriptions |

### Filter Operations

```python
from fastpyvectordb import Filter

Filter.eq("field", value)       # Equal
Filter.ne("field", value)       # Not equal
Filter.gt("field", value)       # Greater than
Filter.gte("field", value)      # Greater than or equal
Filter.lt("field", value)       # Less than
Filter.lte("field", value)      # Less than or equal
Filter.in_("field", [values])   # In list
Filter.contains("field", "sub") # Contains substring
Filter.regex("field", "^pat.*") # Regex match
Filter.and_([f1, f2])           # AND
Filter.or_([f1, f2])            # OR
Filter.not_(f1)                 # NOT
```

### Quantization Comparison

| Quantizer | Compression | Recall | Speed | Use Case |
|-----------|-------------|--------|-------|----------|
| `ScalarQuantizer` | 4x | 97%+ | Moderate | Production (balanced) |
| `BinaryQuantizer` | 32x | ~85% | Very Fast | Ultra-fast filtering |
| `ProductQuantizer` | 8-16x | ~90% | Fast | Research/Analytics |

## Benchmarks

Performance on 100K vectors, 128 dimensions:

| Method | Latency | QPS | Memory |
|--------|---------|-----|--------|
| Naive Python | 450 ms | 2 | 48 MB |
| Vectorized BLAS | 6 ms | 167 | 48 MB |
| HNSW | 0.17 ms | 5,773 | 48 MB |
| Scalar Quantized | 6 ms | 167 | 12 MB |
| Binary Quantized | 0.8 ms | 1,250 | 1.5 MB |

Speedup vs Naive (100K vectors):

| Method | Speedup |
|--------|---------|
| BLAS | 89x |
| Parallel Engine | 87x |
| Batch GEMM | 267x |
| HNSW | 2,388x |
| Hybrid | 939x |

### Running Benchmarks

```bash
# Quick benchmark (10K vectors)
python examples/benchmark.py --quick

# Standard benchmark (100K vectors)
python examples/benchmark.py --medium

# Stress test (1M vectors)
python examples/benchmark.py --stress

# Parallel search benchmark
python examples/benchmark_parallel.py

# Quantization benchmark
python examples/benchmark_quantization.py
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
    M=32,
    ef_construction=400,
)
collection.set_ef_search(100)
```

### Memory vs Speed Guidelines

| Dataset Size | Recommendation |
|--------------|----------------|
| < 100K vectors | HNSW only |
| 100K - 1M | HNSW + Scalar quantization |
| 1M - 10M | Memory-mapped + HNSW |
| > 10M | Memory-mapped + Binary quantization + HNSW candidates |

## Requirements

Core:
```
numpy>=1.24.0
hnswlib>=0.8.0
```

Optional:
```
sentence-transformers>=2.2.0  # Local embeddings
openai>=1.0.0                  # OpenAI embeddings
fastapi>=0.109.0               # REST API
uvicorn>=0.27.0                # ASGI server
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
