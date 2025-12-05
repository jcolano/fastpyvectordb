#!/usr/bin/env python3
"""
Quantization Benchmark Suite for PyVectorDB

Comprehensive performance testing for vector quantization methods.
Tests scalar, binary, and product quantization with HNSW integration.

Usage:
    python benchmark_quantization.py                     # Run all benchmarks with defaults (50K)
    python benchmark_quantization.py --records 1000000   # Test with 1M records
    python benchmark_quantization.py --quick             # Quick benchmark (10K records)
    python benchmark_quantization.py --medium            # Medium benchmark (100K records)
    python benchmark_quantization.py --large             # Large benchmark (500K records)
    python benchmark_quantization.py --stress            # Stress test (1M records)
    python benchmark_quantization.py --only baseline scalar_hybrid  # Run only specific benchmarks
    python benchmark_quantization.py --skip pq memory    # Skip specific benchmarks
    python benchmark_quantization.py --dimension 384     # Test with 384-dim vectors
    python benchmark_quantization.py --compare file1.json file2.json  # Compare results
"""

import argparse
import gc
import json
import os
import sys
import time
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import platform

import numpy as np

# Import our modules
from vectordb_optimized import VectorDB, Collection, Filter, SearchResult
from quantization import ScalarQuantizer, BinaryQuantizer, ProductQuantizer


# =============================================================================
# Configuration & Presets
# =============================================================================

PRESETS = {
    "quick": {"records": 10_000, "description": "Quick smoke test"},
    "default": {"records": 50_000, "description": "Default benchmark"},
    "medium": {"records": 100_000, "description": "Medium scale test"},
    "large": {"records": 500_000, "description": "Large scale test"},
    "stress": {"records": 1_000_000, "description": "Stress test (1M records)"},
    "extreme": {"records": 2_000_000, "description": "Extreme test (2M records)"},
}

# Benchmark categories that can be selected/skipped
BENCHMARK_CATEGORIES = [
    "baseline",       # HNSW only (reference)
    "scalar_bf",      # Scalar quantization brute force
    "binary_bf",      # Binary quantization brute force
    "scalar_hybrid",  # HNSW + Scalar re-ranking
    "binary_hybrid",  # HNSW + Binary re-ranking
    "pq",             # Product quantization
    "recall_k",       # Recall analysis at varying k
    "memory",         # Memory scaling comparison
]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    num_records: int = 50_000
    dimension: int = 128
    num_search_queries: int = 100
    num_recall_queries: int = 50      # Queries for recall calculation
    k: int = 10                        # Default top-k
    hnsw_candidates: int = 100         # Candidates for hybrid search
    pq_subspaces: int = 8              # PQ subspaces
    pq_centroids: int = 256            # PQ centroids per subspace
    pq_train_size: int = 10_000        # PQ training sample size
    progress_interval: int = 0         # Auto-calculated if 0
    chunk_size: int = 50_000           # Generate vectors in chunks

    def __post_init__(self):
        # Auto-calculate progress interval based on record count
        if self.progress_interval == 0:
            if self.num_records <= 10_000:
                self.progress_interval = 5_000
            elif self.num_records <= 100_000:
                self.progress_interval = 10_000
            elif self.num_records <= 500_000:
                self.progress_interval = 50_000
            else:
                self.progress_interval = 100_000

        # Scale recall queries with dataset size
        if self.num_records <= 10_000:
            self.num_recall_queries = 50
        elif self.num_records <= 100_000:
            self.num_recall_queries = 100
        else:
            self.num_recall_queries = 200


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    records: int
    dimension: int
    duration_seconds: float
    operations_per_second: float
    memory_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    recall_at_k: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"  {self.name}",
            f"    Records: {self.records:,}",
            f"    Dimension: {self.dimension}",
            f"    Duration: {self.duration_seconds:.3f}s",
            f"    Throughput: {self.operations_per_second:,.1f} ops/sec",
        ]
        if self.memory_mb is not None and self.memory_mb > 0:
            lines.append(f"    Memory: {self.memory_mb:.2f} MB")
        if self.compression_ratio is not None and self.compression_ratio > 0:
            lines.append(f"    Compression: {self.compression_ratio:.1f}x")
        if self.recall_at_k is not None:
            lines.append(f"    Recall@k: {self.recall_at_k:.1f}%")
        if self.p50_latency_ms is not None:
            lines.append(f"    Latency p50/p95/p99: {self.p50_latency_ms:.3f}/{self.p95_latency_ms:.3f}/{self.p99_latency_ms:.3f} ms")
        if self.min_latency_ms is not None:
            lines.append(f"    Latency min/max: {self.min_latency_ms:.3f}/{self.max_latency_ms:.3f} ms")
        for k, v in self.extra.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "records": self.records,
            "dimension": self.dimension,
            "duration_seconds": self.duration_seconds,
            "operations_per_second": self.operations_per_second,
            "memory_mb": self.memory_mb,
            "compression_ratio": self.compression_ratio,
            "recall_at_k": self.recall_at_k,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "extra": self.extra,
        }


# =============================================================================
# Utilities
# =============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_system_info() -> dict:
    """Get system information for benchmark context."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["total_memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass
    return info


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles from sampled data."""
    if not latencies:
        return {}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    return {
        "p50": latencies_sorted[n // 2],
        "p95": latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[-1],
        "p99": latencies_sorted[int(n * 0.99)] if n > 1 else latencies_sorted[-1],
        "min": latencies_sorted[0],
        "max": latencies_sorted[-1],
    }


def generate_vectors_chunked(n: int, dim: int, seed: int = 42, chunk_size: int = 50_000):
    """Generator that yields normalized vectors in chunks."""
    rng = np.random.default_rng(seed)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        count = end - start

        vectors = rng.standard_normal((count, dim), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        yield start, vectors


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def generate_metadata_batch(start_idx: int, count: int, seed: int = 42) -> List[Dict]:
    """Generate metadata for a batch of records."""
    rng = np.random.default_rng(seed + start_idx)
    categories = ["electronics", "clothing", "books", "sports", "home", "toys"]

    metadata = []
    for i in range(count):
        idx = start_idx + i
        metadata.append({
            "category": categories[idx % len(categories)],
            "price": float(rng.uniform(10, 1000)),
            "rating": float(rng.uniform(1, 5)),
            "in_stock": bool(rng.random() > 0.3),
            "index": idx,
        })
    return metadata


def compute_recall(true_indices: Set[int], approx_indices: Set[int]) -> float:
    """Compute recall@k."""
    if not true_indices:
        return 0.0
    return len(true_indices & approx_indices) / len(true_indices)


def get_ground_truth(vectors: np.ndarray, query: np.ndarray, k: int) -> Set[int]:
    """Compute ground truth top-k using brute force L2."""
    distances = np.linalg.norm(vectors - query, axis=1)
    return set(np.argpartition(distances, k)[:k])


def estimate_benchmark_time(num_records: int) -> str:
    """Estimate total benchmark time based on record count."""
    # Rough estimates for quantization benchmarks
    base_time = num_records / 10000  # ~10K inserts/sec
    quant_time = num_records / 50000 * 5  # Quantization overhead
    search_time = 5  # Fixed search benchmarks
    pq_train_time = 30 if num_records >= 100_000 else 10  # PQ training

    total_seconds = base_time + quant_time + search_time + pq_train_time

    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds / 60)} minutes"
    else:
        return f"~{total_seconds / 3600:.1f} hours"


# =============================================================================
# Quantized Collection Wrapper
# =============================================================================

class QuantizedCollection:
    """
    Collection wrapper that integrates quantization with HNSW.

    Demonstrates proper integration pattern:
    1. Use HNSW for fast candidate retrieval (O(log n))
    2. Use quantized vectors for memory-efficient storage
    3. Optionally re-rank candidates with quantized distances
    """

    def __init__(self, collection: Collection, quantizer_type: str = "scalar"):
        self.collection = collection
        self.quantizer_type = quantizer_type

        # Initialize quantizer
        if quantizer_type == "scalar":
            self.quantizer = ScalarQuantizer(collection.config.dimensions)
        elif quantizer_type == "binary":
            self.quantizer = BinaryQuantizer(collection.config.dimensions)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

        # Quantized vector storage
        self._quantized_vectors: np.ndarray = None
        self._id_list: List[str] = None
        self._id_to_idx: Dict[str, int] = None
        self._trained = False

    def build_quantized_index(self, vectors: np.ndarray, ids: List[str]):
        """Build quantized index from vectors."""
        self.quantizer.train(vectors)
        self._quantized_vectors = self.quantizer.encode(vectors)
        self._id_list = ids
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        self._trained = True

    def search_hybrid(self, query: np.ndarray, k: int = 10,
                      hnsw_candidates: int = 100) -> List[SearchResult]:
        """
        Hybrid search: HNSW candidates + quantized re-ranking.
        """
        candidates = self.collection.search(query, k=hnsw_candidates)

        if not candidates or not self._trained:
            return candidates[:k]

        # Get quantized vectors for candidates
        candidate_indices = []
        for result in candidates:
            if result.id in self._id_to_idx:
                candidate_indices.append(self._id_to_idx[result.id])

        if not candidate_indices:
            return candidates[:k]

        candidate_quantized = self._quantized_vectors[candidate_indices]

        # Compute quantized distances
        if self.quantizer_type == "scalar":
            distances = self.quantizer.distances_l2(query, candidate_quantized)
        else:  # binary
            query_bits = self.quantizer.encode_query(query)
            distances = self.quantizer.hamming_distances(query_bits, candidate_quantized)

        # Sort and return top k
        sorted_indices = np.argsort(distances)[:k]

        results = []
        for idx in sorted_indices:
            original_idx = candidate_indices[idx]
            result = SearchResult(
                id=self._id_list[original_idx],
                score=float(distances[idx]),
                metadata=self.collection._metadata.get(self._id_list[original_idx], {})
            )
            results.append(result)

        return results

    def search_quantized_only(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search using only quantized vectors (brute force)."""
        if not self._trained:
            raise ValueError("Quantized index not built.")

        if self.quantizer_type == "scalar":
            distances = self.quantizer.distances_l2(query, self._quantized_vectors)
        else:
            query_bits = self.quantizer.encode_query(query)
            distances = self.quantizer.hamming_distances(query_bits, self._quantized_vectors)

        if k < len(distances):
            top_k_idx = np.argpartition(distances, k)[:k]
            top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
        else:
            top_k_idx = np.argsort(distances)

        return top_k_idx, distances[top_k_idx]

    def memory_usage(self) -> dict:
        """Get memory usage statistics."""
        n = len(self._id_list) if self._id_list else 0
        d = self.collection.config.dimensions

        float32_bytes = n * d * 4
        quantized_bytes = self._quantized_vectors.nbytes if self._quantized_vectors is not None else 0

        return {
            "original_mb": float32_bytes / 1024 / 1024,
            "quantized_mb": quantized_bytes / 1024 / 1024,
            "compression": float32_bytes / quantized_bytes if quantized_bytes > 0 else 0,
            "savings_percent": (1 - quantized_bytes / float32_bytes) * 100 if float32_bytes > 0 else 0
        }


# =============================================================================
# Benchmarker Class
# =============================================================================

class QuantizationBenchmarker:
    """Runs comprehensive quantization benchmarks."""

    def __init__(self, config: BenchmarkConfig, verbose: bool = True,
                 only: List[str] = None, skip: List[str] = None):
        self.config = config
        self.verbose = verbose
        self.only = set(only) if only else None
        self.skip = set(skip) if skip else set()
        self.results: List[BenchmarkResult] = []

        # Create temp directory for benchmark databases
        self.temp_dir = tempfile.mkdtemp(prefix="quantbench_")
        self._db_counter = 0

        # Will be populated during run
        self.db: Optional[VectorDB] = None
        self.collection: Optional[Collection] = None
        self.vectors: Optional[np.ndarray] = None
        self.ids: Optional[List[str]] = None
        self.query_vectors: Optional[np.ndarray] = None
        self.ground_truths: Optional[List[Set[int]]] = None

    def _get_db_path(self) -> str:
        """Get a unique database path for each benchmark."""
        self._db_counter += 1
        return os.path.join(self.temp_dir, f"db_{self._db_counter}")

    def cleanup(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def should_run(self, category: str) -> bool:
        """Check if a benchmark category should run."""
        if self.only is not None:
            return category in self.only
        return category not in self.skip

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.log(f"\n{'='*70}")
        self.log(f"  QUANTIZATION BENCHMARK SUITE")
        self.log(f"  Records: {self.config.num_records:,} | Dimension: {self.config.dimension}")
        self.log(f"  Estimated time: {estimate_benchmark_time(self.config.num_records)}")
        self.log(f"{'='*70}\n")

        # Setup: Generate data and create HNSW index
        self._setup_data()

        # Run benchmarks based on selection
        benchmark_methods = [
            ("baseline", self.benchmark_baseline),
            ("scalar_bf", self.benchmark_scalar_brute_force),
            ("binary_bf", self.benchmark_binary_brute_force),
            ("scalar_hybrid", self.benchmark_scalar_hybrid),
            ("binary_hybrid", self.benchmark_binary_hybrid),
            ("pq", self.benchmark_product_quantization),
            ("recall_k", self.benchmark_recall_at_k),
            ("memory", self.benchmark_memory_scaling),
        ]

        for category, method in benchmark_methods:
            if self.should_run(category):
                try:
                    method()
                except Exception as e:
                    self.log(f"  ERROR in {category}: {e}\n")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()

        self.print_summary()
        return self.results

    def _setup_data(self):
        """Generate vectors, create HNSW index, compute ground truths."""
        self.log("--- Setup: Generating Data & Building Index ---")

        # Generate vectors in chunks for memory efficiency
        self.log(f"  Generating {self.config.num_records:,} vectors...")
        start = time.perf_counter()

        all_vectors = []
        for chunk_start, chunk_vectors in generate_vectors_chunked(
            self.config.num_records,
            self.config.dimension,
            seed=42,
            chunk_size=self.config.chunk_size
        ):
            all_vectors.append(chunk_vectors)
            if (chunk_start + len(chunk_vectors)) % self.config.progress_interval == 0:
                self.log(f"    Generated {chunk_start + len(chunk_vectors):,} vectors...")

        self.vectors = np.vstack(all_vectors)
        self.ids = [f"vec_{i}" for i in range(self.config.num_records)]

        gen_time = time.perf_counter() - start
        self.log(f"  Vector generation: {gen_time:.2f}s")

        # Generate query vectors
        self.log(f"  Generating {self.config.num_search_queries} query vectors...")
        self.query_vectors = generate_vectors(
            self.config.num_search_queries,
            self.config.dimension,
            seed=999
        )

        # Create HNSW index
        self.log(f"  Building HNSW index...")
        start = time.perf_counter()

        self.db = VectorDB(self._get_db_path())
        self.collection = self.db.create_collection(
            "benchmark",
            dimensions=self.config.dimension
        )

        # Insert in batches
        batch_size = min(50_000, self.config.num_records)
        for i in range(0, self.config.num_records, batch_size):
            end = min(i + batch_size, self.config.num_records)
            batch_vectors = self.vectors[i:end]
            batch_ids = self.ids[i:end]
            metadata = generate_metadata_batch(i, end - i)
            self.collection.insert_batch(batch_vectors, batch_ids, metadata)

            if end % self.config.progress_interval == 0:
                self.log(f"    Inserted {end:,} vectors...")

        insert_time = time.perf_counter() - start
        self.log(f"  HNSW index built: {insert_time:.2f}s ({self.config.num_records/insert_time:,.0f} vec/sec)")

        # Compute ground truths for recall calculation
        self.log(f"  Computing ground truths for {self.config.num_recall_queries} queries...")
        start = time.perf_counter()

        self.ground_truths = []
        for i in range(self.config.num_recall_queries):
            gt = get_ground_truth(self.vectors, self.query_vectors[i], self.config.k)
            self.ground_truths.append(gt)

        gt_time = time.perf_counter() - start
        self.log(f"  Ground truth computation: {gt_time:.2f}s\n")

    def _compute_recall(self, approx_indices: Set[int], query_idx: int) -> float:
        """Compute recall for a single query."""
        if query_idx >= len(self.ground_truths):
            return 0.0
        return compute_recall(self.ground_truths[query_idx], approx_indices)

    def benchmark_baseline(self):
        """Benchmark HNSW only (baseline reference)."""
        self.log("--- Benchmark: HNSW Baseline ---")

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, query in enumerate(self.query_vectors):
            t0 = time.perf_counter()
            results = self.collection.search(query, k=self.config.k)
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < self.config.num_recall_queries:
                result_indices = set(int(r.id.split("_")[1]) for r in results)
                recalls.append(self._compute_recall(result_indices, i))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        # Memory: HNSW index + vectors
        memory_mb = (self.config.num_records * self.config.dimension * 4) / (1024 * 1024)

        result = BenchmarkResult(
            name="HNSW Baseline",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=len(self.query_vectors) / duration,
            memory_mb=memory_mb,
            compression_ratio=1.0,
            recall_at_k=np.mean(recalls) * 100 if recalls else None,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"num_queries": len(self.query_vectors)}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_scalar_brute_force(self):
        """Benchmark scalar quantization with brute force search."""
        self.log("--- Benchmark: Scalar Quantization (Brute Force) ---")

        # Train and encode
        self.log("  Training scalar quantizer...")
        sq = ScalarQuantizer(self.config.dimension)
        sq.train(self.vectors)

        self.log("  Encoding vectors...")
        sq_vectors = sq.encode(self.vectors)

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, query in enumerate(self.query_vectors):
            t0 = time.perf_counter()
            distances = sq.distances_l2(query, sq_vectors)
            top_k_idx = np.argpartition(distances, self.config.k)[:self.config.k]
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < self.config.num_recall_queries:
                recalls.append(self._compute_recall(set(top_k_idx), i))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)
        mem_info = sq.memory_usage(self.config.num_records)

        result = BenchmarkResult(
            name="Scalar Quantization (BF)",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=len(self.query_vectors) / duration,
            memory_mb=mem_info['quantized_bytes'] / (1024 * 1024),
            compression_ratio=mem_info['compression_ratio'],
            recall_at_k=np.mean(recalls) * 100 if recalls else None,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"quantizer": "uint8"}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_binary_brute_force(self):
        """Benchmark binary quantization with brute force search."""
        self.log("--- Benchmark: Binary Quantization (Brute Force) ---")

        # Train and encode
        self.log("  Training binary quantizer...")
        bq = BinaryQuantizer(self.config.dimension)
        bq.train(self.vectors, use_median=True)

        self.log("  Encoding vectors...")
        bq_vectors = bq.encode(self.vectors)

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, query in enumerate(self.query_vectors):
            t0 = time.perf_counter()
            top_k_idx, _ = bq.search(query, bq_vectors, k=self.config.k)
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < self.config.num_recall_queries:
                recalls.append(self._compute_recall(set(top_k_idx), i))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)
        mem_info = bq.memory_usage(self.config.num_records)

        result = BenchmarkResult(
            name="Binary Quantization (BF)",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=len(self.query_vectors) / duration,
            memory_mb=mem_info['quantized_bytes'] / (1024 * 1024),
            compression_ratio=mem_info['compression_ratio'],
            recall_at_k=np.mean(recalls) * 100 if recalls else None,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"quantizer": "1-bit"}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_scalar_hybrid(self):
        """Benchmark HNSW + Scalar re-ranking."""
        self.log("--- Benchmark: HNSW + Scalar Re-ranking ---")

        # Build quantized collection
        self.log("  Building quantized index...")
        qc = QuantizedCollection(self.collection, quantizer_type="scalar")
        qc.build_quantized_index(self.vectors, self.ids)

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, query in enumerate(self.query_vectors):
            t0 = time.perf_counter()
            results = qc.search_hybrid(
                query,
                k=self.config.k,
                hnsw_candidates=self.config.hnsw_candidates
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < self.config.num_recall_queries:
                result_indices = set(int(r.id.split("_")[1]) for r in results)
                recalls.append(self._compute_recall(result_indices, i))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)
        mem_info = qc.memory_usage()

        result = BenchmarkResult(
            name="HNSW + Scalar Re-rank",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=len(self.query_vectors) / duration,
            memory_mb=mem_info['quantized_mb'],
            compression_ratio=mem_info['compression'],
            recall_at_k=np.mean(recalls) * 100 if recalls else None,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"hnsw_candidates": self.config.hnsw_candidates}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_binary_hybrid(self):
        """Benchmark HNSW + Binary re-ranking."""
        self.log("--- Benchmark: HNSW + Binary Re-ranking ---")

        # Build quantized collection
        self.log("  Building quantized index...")
        qc = QuantizedCollection(self.collection, quantizer_type="binary")
        qc.build_quantized_index(self.vectors, self.ids)

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, query in enumerate(self.query_vectors):
            t0 = time.perf_counter()
            results = qc.search_hybrid(
                query,
                k=self.config.k,
                hnsw_candidates=self.config.hnsw_candidates
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < self.config.num_recall_queries:
                result_indices = set(int(r.id.split("_")[1]) for r in results)
                recalls.append(self._compute_recall(result_indices, i))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)
        mem_info = qc.memory_usage()

        result = BenchmarkResult(
            name="HNSW + Binary Re-rank",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=len(self.query_vectors) / duration,
            memory_mb=mem_info['quantized_mb'],
            compression_ratio=mem_info['compression'],
            recall_at_k=np.mean(recalls) * 100 if recalls else None,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"hnsw_candidates": self.config.hnsw_candidates}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_product_quantization(self):
        """Benchmark product quantization."""
        self.log("--- Benchmark: Product Quantization ---")

        # Ensure dimensions are divisible by subspaces
        if self.config.dimension % self.config.pq_subspaces != 0:
            self.log(f"  Skipping: dimension {self.config.dimension} not divisible by {self.config.pq_subspaces}\n")
            return

        # Train PQ
        self.log(f"  Training PQ codebooks (M={self.config.pq_subspaces}, K={self.config.pq_centroids})...")
        train_start = time.perf_counter()

        pq = ProductQuantizer(
            self.config.dimension,
            num_subspaces=self.config.pq_subspaces,
            num_centroids=self.config.pq_centroids
        )

        # Use subset for training
        train_size = min(self.config.pq_train_size, self.config.num_records)
        pq.train(self.vectors[:train_size], n_iter=10)

        train_time = time.perf_counter() - train_start
        self.log(f"  PQ training: {train_time:.2f}s")

        # Encode all vectors
        self.log("  Encoding vectors...")
        pq_codes = pq.encode(self.vectors)

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, query in enumerate(self.query_vectors):
            t0 = time.perf_counter()
            top_k_idx, _ = pq.search(query, pq_codes, k=self.config.k)
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < self.config.num_recall_queries:
                recalls.append(self._compute_recall(set(top_k_idx), i))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)
        mem_info = pq.memory_usage(self.config.num_records)

        result = BenchmarkResult(
            name=f"Product Quantization (PQ{self.config.pq_subspaces})",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=len(self.query_vectors) / duration,
            memory_mb=mem_info['quantized_bytes'] / (1024 * 1024),
            compression_ratio=mem_info['compression_ratio'],
            recall_at_k=np.mean(recalls) * 100 if recalls else None,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={
                "pq_subspaces": self.config.pq_subspaces,
                "pq_centroids": self.config.pq_centroids,
                "train_time_s": round(train_time, 2)
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_recall_at_k(self):
        """Benchmark recall at different k values."""
        self.log("--- Benchmark: Recall at Varying K ---")

        k_values = [1, 5, 10, 20, 50, 100]
        k_values = [k for k in k_values if k <= self.config.num_records]

        # Initialize quantizers
        sq = ScalarQuantizer(self.config.dimension)
        sq.train(self.vectors)
        sq_vectors = sq.encode(self.vectors)

        bq = BinaryQuantizer(self.config.dimension)
        bq.train(self.vectors)
        bq_vectors = bq.encode(self.vectors)

        num_queries = min(50, self.config.num_recall_queries)

        for k in k_values:
            # Compute ground truths for this k
            k_ground_truths = []
            for i in range(num_queries):
                gt = get_ground_truth(self.vectors, self.query_vectors[i], k)
                k_ground_truths.append(gt)

            # Test each method
            methods = [
                ("HNSW", lambda q: set(int(r.id.split("_")[1]) for r in self.collection.search(q, k=k))),
                ("Scalar", lambda q: set(np.argpartition(sq.distances_l2(q, sq_vectors), k)[:k])),
                ("Binary", lambda q: set(bq.search(q, bq_vectors, k=k)[0])),
            ]

            for method_name, search_fn in methods:
                recalls = []
                start = time.perf_counter()

                for i in range(num_queries):
                    approx = search_fn(self.query_vectors[i])
                    recalls.append(compute_recall(k_ground_truths[i], approx))

                duration = time.perf_counter() - start

                result = BenchmarkResult(
                    name=f"Recall@{k} ({method_name})",
                    records=self.config.num_records,
                    dimension=self.config.dimension,
                    duration_seconds=duration,
                    operations_per_second=num_queries / duration,
                    recall_at_k=np.mean(recalls) * 100,
                    extra={"k": k, "method": method_name, "num_queries": num_queries}
                )
                self.results.append(result)
                self.log(str(result))

        self.log("")

    def benchmark_memory_scaling(self):
        """Benchmark memory and recall at different dataset sizes."""
        self.log("--- Benchmark: Memory Scaling ---")

        # Dynamic scales based on target record count
        base_scales = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
        scales = [s for s in base_scales if s <= self.config.num_records]

        if len(scales) < 2:
            scales = [1000, self.config.num_records]

        num_test_queries = 20

        for scale in scales:
            gc.collect()
            time.sleep(0.1)

            # Use subset of vectors
            subset_vectors = self.vectors[:scale]
            subset_ids = self.ids[:scale]

            # Create quantizers
            sq = ScalarQuantizer(self.config.dimension)
            sq.train(subset_vectors)
            sq_vectors = sq.encode(subset_vectors)

            bq = BinaryQuantizer(self.config.dimension)
            bq.train(subset_vectors)
            bq_vectors = bq.encode(subset_vectors)

            # Compute ground truths for this subset
            scale_gts = []
            for i in range(num_test_queries):
                gt = get_ground_truth(subset_vectors, self.query_vectors[i], self.config.k)
                scale_gts.append(gt)

            # Test recalls
            sq_recalls = []
            bq_recalls = []

            for i in range(num_test_queries):
                # Scalar
                sq_dists = sq.distances_l2(self.query_vectors[i], sq_vectors)
                sq_top_k = set(np.argpartition(sq_dists, self.config.k)[:self.config.k])
                sq_recalls.append(compute_recall(scale_gts[i], sq_top_k))

                # Binary
                bq_top_k, _ = bq.search(self.query_vectors[i], bq_vectors, k=self.config.k)
                bq_recalls.append(compute_recall(scale_gts[i], set(bq_top_k)))

            # Memory stats
            sq_mem = sq.memory_usage(scale)
            bq_mem = bq.memory_usage(scale)

            # Record results
            result_sq = BenchmarkResult(
                name=f"Memory@{scale:,} (Scalar)",
                records=scale,
                dimension=self.config.dimension,
                duration_seconds=0,
                operations_per_second=0,
                memory_mb=sq_mem['quantized_bytes'] / (1024 * 1024),
                compression_ratio=sq_mem['compression_ratio'],
                recall_at_k=np.mean(sq_recalls) * 100,
                extra={"bytes_per_vector": sq_mem['quantized_bytes'] / scale}
            )
            self.results.append(result_sq)
            self.log(str(result_sq))

            result_bq = BenchmarkResult(
                name=f"Memory@{scale:,} (Binary)",
                records=scale,
                dimension=self.config.dimension,
                duration_seconds=0,
                operations_per_second=0,
                memory_mb=bq_mem['quantized_bytes'] / (1024 * 1024),
                compression_ratio=bq_mem['compression_ratio'],
                recall_at_k=np.mean(bq_recalls) * 100,
                extra={"bytes_per_vector": bq_mem['quantized_bytes'] / scale}
            )
            self.results.append(result_bq)
            self.log(str(result_bq))

        self.log("")

    def print_summary(self):
        """Print a summary table of all results."""
        self.log("\n" + "="*70)
        self.log("  BENCHMARK SUMMARY")
        self.log("="*70 + "\n")

        # Group by category
        categories = {
            "Baseline": ["HNSW Baseline"],
            "Brute Force": ["Scalar Quantization (BF)", "Binary Quantization (BF)"],
            "Hybrid": ["HNSW + Scalar", "HNSW + Binary"],
            "Product Quantization": ["Product Quantization"],
            "Recall Analysis": ["Recall@"],
            "Memory Scaling": ["Memory@"],
        }

        for category, prefixes in categories.items():
            relevant = [r for r in self.results
                       if any(r.name.startswith(p) for p in prefixes)]
            if relevant:
                self.log(f"  {category}:")
                for r in relevant:
                    parts = []
                    if r.operations_per_second > 0:
                        parts.append(f"{r.operations_per_second:,.0f} ops/s")
                    if r.p50_latency_ms:
                        parts.append(f"p50={r.p50_latency_ms:.2f}ms")
                    if r.compression_ratio and r.compression_ratio > 1:
                        parts.append(f"{r.compression_ratio:.1f}x compression")
                    if r.recall_at_k:
                        parts.append(f"recall={r.recall_at_k:.1f}%")
                    if r.memory_mb and r.memory_mb > 0:
                        parts.append(f"mem={r.memory_mb:.2f}MB")

                    self.log(f"    {r.name}: {' | '.join(parts)}")
                self.log("")

        # Key metrics summary
        baseline = next((r for r in self.results if r.name == "HNSW Baseline"), None)
        scalar_hybrid = next((r for r in self.results if "Scalar Re-rank" in r.name), None)
        binary_bf = next((r for r in self.results if "Binary Quantization (BF)" in r.name), None)

        self.log("  KEY INSIGHTS:")
        if baseline:
            self.log(f"    Baseline QPS: {baseline.operations_per_second:,.0f}")
            self.log(f"    Baseline Recall@{self.config.k}: {baseline.recall_at_k:.1f}%")
        if scalar_hybrid:
            self.log(f"    Scalar Hybrid Compression: {scalar_hybrid.compression_ratio:.1f}x")
            self.log(f"    Scalar Hybrid Recall: {scalar_hybrid.recall_at_k:.1f}%")
        if binary_bf:
            self.log(f"    Binary Compression: {binary_bf.compression_ratio:.1f}x")
            self.log(f"    Binary Recall: {binary_bf.recall_at_k:.1f}%")

    def export_json(self, path: str):
        """Export results to JSON with full metadata."""
        data = {
            "benchmark_version": "1.0",
            "benchmark_type": "quantization",
            "timestamp": datetime.now().isoformat(),
            "system_info": get_system_info(),
            "config": {
                "num_records": self.config.num_records,
                "dimension": self.config.dimension,
                "k": self.config.k,
                "hnsw_candidates": self.config.hnsw_candidates,
                "pq_subspaces": self.config.pq_subspaces,
                "pq_centroids": self.config.pq_centroids,
            },
            "results": [r.to_dict() for r in self.results]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.log(f"\nResults exported to: {path}")


# =============================================================================
# Comparison Tool
# =============================================================================

def compare_benchmarks(file1: str, file2: str):
    """Compare two benchmark result files."""
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    print(f"\n{'='*70}")
    print(f"  QUANTIZATION BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"\n  File 1: {file1}")
    print(f"  File 2: {file2}")
    print(f"\n  Config 1: {data1['config']['num_records']:,} records, dim={data1['config']['dimension']}")
    print(f"  Config 2: {data2['config']['num_records']:,} records, dim={data2['config']['dimension']}")

    # Build result maps
    results1 = {r['name']: r for r in data1['results']}
    results2 = {r['name']: r for r in data2['results']}

    # Find common benchmarks
    common = set(results1.keys()) & set(results2.keys())

    print(f"\n  {'Benchmark':<35} {'File1':>12} {'File2':>12} {'Change':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    for name in sorted(common):
        r1 = results1[name]
        r2 = results2[name]

        # Compare QPS
        ops1 = r1.get('operations_per_second', 0)
        ops2 = r2.get('operations_per_second', 0)

        if ops1 > 0 and ops2 > 0:
            change = ((ops2 - ops1) / ops1) * 100
            change_str = f"{change:+.1f}%"
            print(f"  {name:<35} {ops1:>12,.0f} {ops2:>12,.0f} {change_str:>10}")

        # Compare recall
        recall1 = r1.get('recall_at_k')
        recall2 = r2.get('recall_at_k')
        if recall1 and recall2:
            recall_change = recall2 - recall1
            print(f"    └─ Recall: {recall1:>10.1f}% {recall2:>10.1f}% {recall_change:>+9.1f}%")

    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantization Benchmark Suite for PyVectorDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  --quick     10K records (smoke test)
  --medium    100K records
  --large     500K records
  --stress    1M records
  --extreme   2M records

Benchmark Categories:
  baseline      - HNSW only (reference)
  scalar_bf     - Scalar quantization brute force
  binary_bf     - Binary quantization brute force
  scalar_hybrid - HNSW + Scalar re-ranking
  binary_hybrid - HNSW + Binary re-ranking
  pq            - Product quantization
  recall_k      - Recall analysis at varying k
  memory        - Memory scaling comparison

Examples:
  python benchmark_quantization.py --quick
  python benchmark_quantization.py --records 250000
  python benchmark_quantization.py --large --only baseline scalar_hybrid
  python benchmark_quantization.py --stress --skip pq memory
  python benchmark_quantization.py --compare run1.json run2.json
        """
    )

    # Size options
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--records", "-n", type=int, default=50_000,
                           help="Number of records (default: 50000)")
    size_group.add_argument("--quick", "-q", action="store_true",
                           help="Quick benchmark (10K records)")
    size_group.add_argument("--medium", "-m", action="store_true",
                           help="Medium benchmark (100K records)")
    size_group.add_argument("--large", "-l", action="store_true",
                           help="Large benchmark (500K records)")
    size_group.add_argument("--stress", "-s", action="store_true",
                           help="Stress test (1M records)")
    size_group.add_argument("--extreme", "-x", action="store_true",
                           help="Extreme test (2M records)")

    # Other options
    parser.add_argument("--dimension", "-d", type=int, default=128,
                       help="Vector dimension (default: 128)")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k for search (default: 10)")
    parser.add_argument("--hnsw-candidates", type=int, default=100,
                       help="HNSW candidates for hybrid search (default: 100)")
    parser.add_argument("--pq-subspaces", type=int, default=8,
                       help="PQ subspaces (default: 8)")
    parser.add_argument("--output", "-o", type=str,
                       help="Export results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")

    # Benchmark selection
    parser.add_argument("--only", nargs="+", choices=BENCHMARK_CATEGORIES,
                       help="Run only specified benchmarks")
    parser.add_argument("--skip", nargs="+", choices=BENCHMARK_CATEGORIES,
                       help="Skip specified benchmarks")

    # Comparison mode
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"),
                       help="Compare two benchmark result files")

    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        compare_benchmarks(args.compare[0], args.compare[1])
        return

    # Determine record count from preset or explicit value
    if args.quick:
        num_records = PRESETS["quick"]["records"]
    elif args.medium:
        num_records = PRESETS["medium"]["records"]
    elif args.large:
        num_records = PRESETS["large"]["records"]
    elif args.stress:
        num_records = PRESETS["stress"]["records"]
    elif args.extreme:
        num_records = PRESETS["extreme"]["records"]
    else:
        num_records = args.records

    config = BenchmarkConfig(
        num_records=num_records,
        dimension=args.dimension,
        k=args.k,
        hnsw_candidates=args.hnsw_candidates,
        pq_subspaces=args.pq_subspaces,
    )

    benchmarker = QuantizationBenchmarker(
        config=config,
        verbose=not args.quiet,
        only=args.only,
        skip=args.skip,
    )

    try:
        results = benchmarker.run_all()

        if args.output:
            benchmarker.export_json(args.output)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted.")
        sys.exit(1)
    finally:
        benchmarker.cleanup()


if __name__ == "__main__":
    main()
