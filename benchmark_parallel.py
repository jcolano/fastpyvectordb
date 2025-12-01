#!/usr/bin/env python3
"""
PyVectorDB Phase 3: Parallel Processing Benchmark Suite

Comprehensive benchmarks for multi-core parallel search optimizations.

Usage:
    python benchmark_parallel.py                      # Run all benchmarks with defaults (50K)
    python benchmark_parallel.py --records 100000     # Test with 100K records
    python benchmark_parallel.py --quick              # Quick benchmark (10K records)
    python benchmark_parallel.py --medium             # Medium benchmark (100K records)
    python benchmark_parallel.py --large              # Large benchmark (500K records)
    python benchmark_parallel.py --stress             # Stress test (1M records)
    python benchmark_parallel.py --only naive vectorized hnsw  # Run specific benchmarks
    python benchmark_parallel.py --skip mmap          # Skip specific benchmarks
    python benchmark_parallel.py --workers 8          # Set worker count
    python benchmark_parallel.py --chunk-size 50000   # Set chunk size
    python benchmark_parallel.py --output results.json  # Export results
    python benchmark_parallel.py --compare file1.json file2.json  # Compare results
"""

import argparse
import gc
import json
import os
import sys
import time
import tempfile
import shutil
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import platform

import numpy as np

# Try to import parallel_search module
try:
    from parallel_search import (
        ParallelSearchEngine,
        ParallelCollection,
        MemoryMappedVectors,
        _compute_distances_vectorized,
        HNSWLIB_AVAILABLE
    )
    PARALLEL_SEARCH_AVAILABLE = True
except ImportError as e:
    PARALLEL_SEARCH_AVAILABLE = False
    HNSWLIB_AVAILABLE = False
    print(f"Warning: parallel_search module not available: {e}")
    print("Some benchmarks will be skipped.")


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

# Benchmark categories
BENCHMARK_CATEGORIES = [
    "naive",       # Naive Python loop baseline (skipped for large presets)
    "vectorized",  # Vectorized NumPy BLAS operations
    "engine",      # ParallelSearchEngine single query
    "batch",       # Batch GEMM matrix-matrix multiply
    "chunked",     # Chunked parallel processing
    "hnsw",        # HNSW approximate search
    "mmap",        # Memory-mapped storage + search
    "hybrid",      # ParallelCollection hybrid search
]

# Skip naive for these presets (too slow)
SKIP_NAIVE_PRESETS = {"medium", "large", "stress", "extreme"}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    num_records: int = 50_000
    dimension: int = 128
    num_queries: int = 100
    k: int = 10
    workers: int = 0          # 0 = auto (CPU count)
    chunk_size: int = 50_000
    hnsw_ef_search: int = 50
    hnsw_ef_construction: int = 200
    hnsw_M: int = 16
    recall_queries: int = 20  # Queries used for recall calculation
    naive_max_records: int = 10_000  # Max records for naive baseline
    naive_max_queries: int = 5       # Max queries for naive baseline

    def __post_init__(self):
        if self.workers == 0:
            self.workers = mp.cpu_count()


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    records: int
    dimension: int
    duration_seconds: float
    operations_per_second: float
    memory_mb: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    recall: Optional[float] = None
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
            lines.append(f"    Memory: {self.memory_mb:.1f} MB")
        if self.p50_latency_ms is not None and self.p95_latency_ms is not None and self.p99_latency_ms is not None:
            lines.append(f"    Latency p50/p95/p99: {self.p50_latency_ms:.3f}/{self.p95_latency_ms:.3f}/{self.p99_latency_ms:.3f} ms")
        elif self.p50_latency_ms is not None:
            lines.append(f"    Latency avg: {self.p50_latency_ms:.3f} ms")
        if self.min_latency_ms is not None:
            lines.append(f"    Latency min/max: {self.min_latency_ms:.3f}/{self.max_latency_ms:.3f} ms")
        if self.recall is not None:
            lines.append(f"    Recall@{self.extra.get('k', 10)}: {self.recall:.1%}")
        for k, v in self.extra.items():
            if k != 'k':
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
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "recall": self.recall,
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
        "hnswlib_available": HNSWLIB_AVAILABLE,
        "parallel_search_available": PARALLEL_SEARCH_AVAILABLE,
    }
    try:
        import psutil
        info["total_memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass
    return info


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
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


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def compute_recall(ground_truth: Set[int], results: Set[int]) -> float:
    """Compute recall@k."""
    if not ground_truth:
        return 0.0
    return len(ground_truth & results) / len(ground_truth)


def get_ground_truth_brute_force(query: np.ndarray, vectors: np.ndarray, k: int) -> Set[int]:
    """Compute ground truth top-k using brute force."""
    # Cosine distance
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_norm = vectors / norms
    similarities = np.dot(vectors_norm, query_norm)
    distances = 1.0 - similarities
    
    top_k_idx = np.argpartition(distances, k)[:k]
    return set(top_k_idx.tolist())


def naive_search(query: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """Naive O(n*d) Python loop - baseline."""
    distances = np.zeros(len(vectors))
    for i, v in enumerate(vectors):
        distances[i] = np.sqrt(np.sum((query - v) ** 2))
    return np.argpartition(distances, k)[:k]


def estimate_benchmark_time(num_records: int, skip_naive: bool) -> str:
    """Estimate total benchmark time based on record count."""
    # Rough estimates
    naive_time = 0 if skip_naive else (num_records / 100) * 0.5  # Very slow
    vectorized_time = num_records / 50000 * 2
    hnsw_build_time = num_records / 100000 * 5
    other_time = num_records / 100000 * 10
    
    total_seconds = naive_time + vectorized_time + hnsw_build_time + other_time
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds / 60)} minutes"
    else:
        return f"~{total_seconds / 3600:.1f} hours"


# =============================================================================
# Benchmarker Class
# =============================================================================

class ParallelBenchmarker:
    """Runs comprehensive parallel processing benchmarks."""

    def __init__(self, config: BenchmarkConfig, verbose: bool = True,
                 only: List[str] = None, skip: List[str] = None,
                 preset: str = "default"):
        self.config = config
        self.verbose = verbose
        self.only = set(only) if only else None
        self.skip = set(skip) if skip else set()
        self.preset = preset
        self.results: List[BenchmarkResult] = []
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="parallel_bench_")
        
        # Will be populated during run
        self.vectors: Optional[np.ndarray] = None
        self.queries: Optional[np.ndarray] = None
        self.ground_truths: Optional[List[Set[int]]] = None
        self.engine: Optional['ParallelSearchEngine'] = None
        self.collection: Optional['ParallelCollection'] = None

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

    def should_skip_naive(self) -> bool:
        """Check if naive baseline should be skipped for this preset."""
        return self.preset in SKIP_NAIVE_PRESETS

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        skip_naive = self.should_skip_naive()
        
        self.log(f"\n{'='*70}")
        self.log(f"  PHASE 3: PARALLEL PROCESSING BENCHMARK SUITE")
        self.log(f"  Records: {self.config.num_records:,} | Dimension: {self.config.dimension}")
        self.log(f"  Workers: {self.config.workers} | Chunk Size: {self.config.chunk_size:,}")
        self.log(f"  Estimated time: {estimate_benchmark_time(self.config.num_records, skip_naive)}")
        self.log(f"{'='*70}\n")

        if not PARALLEL_SEARCH_AVAILABLE:
            self.log("ERROR: parallel_search module not available. Cannot run benchmarks.")
            return self.results

        # Generate test data
        self.log("Generating test data...")
        self.vectors = generate_vectors(self.config.num_records, self.config.dimension, seed=42)
        self.queries = generate_vectors(self.config.num_queries, self.config.dimension, seed=999)
        self.log(f"  Generated {self.config.num_records:,} vectors and {self.config.num_queries} queries\n")

        # Compute ground truth for recall calculation
        self.log("Computing ground truth for recall measurement...")
        self.ground_truths = []
        for q in self.queries[:self.config.recall_queries]:
            gt = get_ground_truth_brute_force(q, self.vectors, self.config.k)
            self.ground_truths.append(gt)
        self.log(f"  Computed ground truth for {len(self.ground_truths)} queries\n")

        # Initialize parallel engine
        self.engine = ParallelSearchEngine(
            n_workers=self.config.workers,
            chunk_size=self.config.chunk_size
        )

        # Run benchmarks
        benchmark_methods = [
            ("naive", self.benchmark_naive),
            ("vectorized", self.benchmark_vectorized),
            ("engine", self.benchmark_engine),
            ("batch", self.benchmark_batch),
            ("chunked", self.benchmark_chunked),
            ("hnsw", self.benchmark_hnsw),
            ("mmap", self.benchmark_mmap),
            ("hybrid", self.benchmark_hybrid),
        ]

        for category, method in benchmark_methods:
            if self.should_run(category):
                # Skip naive for large presets
                if category == "naive" and skip_naive:
                    self.log(f"--- Benchmark: Naive Baseline ---")
                    self.log(f"  Skipped for {self.preset} preset (too slow)\n")
                    continue
                    
                try:
                    method()
                except Exception as e:
                    self.log(f"  ERROR in {category}: {e}\n")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()

        self.print_summary()
        return self.results

    def benchmark_naive(self):
        """Benchmark naive Python loop baseline."""
        self.log("--- Benchmark: Naive Python Loop Baseline ---")

        # Use limited records/queries for naive (it's very slow)
        n_records = min(self.config.num_records, self.config.naive_max_records)
        n_queries = min(self.config.num_queries, self.config.naive_max_queries)
        
        vectors_subset = self.vectors[:n_records]
        queries_subset = self.queries[:n_queries]

        self.log(f"  Using {n_records:,} records, {n_queries} queries (limited for speed)")

        latencies = []
        gc.collect()
        
        start = time.perf_counter()
        for q in queries_subset:
            t0 = time.perf_counter()
            _ = naive_search(q, vectors_subset, self.config.k)
            latencies.append((time.perf_counter() - t0) * 1000)
        
        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        result = BenchmarkResult(
            name="Naive Python Loop",
            records=n_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=n_queries / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={
                "num_queries": n_queries,
                "note": "Baseline for speedup comparison"
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_vectorized(self):
        """Benchmark vectorized NumPy BLAS operations."""
        self.log("--- Benchmark: Vectorized BLAS ---")

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, q in enumerate(self.queries):
            t0 = time.perf_counter()
            distances = _compute_distances_vectorized(q, self.vectors, "cosine")
            top_k_idx = np.argpartition(distances, self.config.k)[:self.config.k]
            latencies.append((time.perf_counter() - t0) * 1000)

            # Compute recall for subset
            if i < len(self.ground_truths):
                recalls.append(compute_recall(self.ground_truths[i], set(top_k_idx.tolist())))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        result = BenchmarkResult(
            name="Vectorized BLAS",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_queries / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            recall=np.mean(recalls) if recalls else None,
            extra={"k": self.config.k, "num_queries": self.config.num_queries}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_engine(self):
        """Benchmark ParallelSearchEngine single query search."""
        self.log("--- Benchmark: ParallelSearchEngine ---")

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, q in enumerate(self.queries):
            t0 = time.perf_counter()
            results = self.engine.search_parallel(q, self.vectors, k=self.config.k, metric="cosine")
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < len(self.ground_truths):
                result_indices = set(r.index for r in results)
                recalls.append(compute_recall(self.ground_truths[i], result_indices))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        result = BenchmarkResult(
            name="ParallelSearchEngine",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_queries / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            recall=np.mean(recalls) if recalls else None,
            extra={
                "k": self.config.k,
                "workers": self.config.workers,
                "num_queries": self.config.num_queries
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_batch(self):
        """Benchmark batch GEMM search."""
        self.log("--- Benchmark: Batch GEMM Search ---")

        # Test different batch sizes
        batch_sizes = [10, 20, 50, 100]
        
        for batch_size in batch_sizes:
            if batch_size > self.config.num_queries:
                continue

            query_batch = self.queries[:batch_size]
            
            gc.collect()
            
            # Warmup
            _ = self.engine.search_batch_parallel(query_batch[:5], self.vectors, k=self.config.k)
            
            # Benchmark
            start = time.perf_counter()
            batch_results = self.engine.search_batch_parallel(
                query_batch, self.vectors, k=self.config.k, metric="cosine"
            )
            duration = time.perf_counter() - start

            per_query_ms = duration * 1000 / batch_size

            # Compute recall
            recalls = []
            for i, results in enumerate(batch_results):
                if i < len(self.ground_truths):
                    result_indices = set(r.index for r in results)
                    recalls.append(compute_recall(self.ground_truths[i], result_indices))

            result = BenchmarkResult(
                name=f"Batch GEMM (batch={batch_size})",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=duration,
                operations_per_second=batch_size / duration,
                p50_latency_ms=per_query_ms,
                recall=np.mean(recalls) if recalls else None,
                extra={
                    "k": self.config.k,
                    "batch_size": batch_size,
                    "total_ms": duration * 1000,
                    "per_query_ms": per_query_ms
                }
            )
            self.results.append(result)
            self.log(str(result))

        self.log("")

    def benchmark_chunked(self):
        """Benchmark chunked parallel processing."""
        self.log("--- Benchmark: Chunked Parallel Search ---")

        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, q in enumerate(self.queries):
            t0 = time.perf_counter()
            results = self.engine.search_chunked_parallel(
                q, self.vectors, k=self.config.k, metric="cosine"
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < len(self.ground_truths):
                result_indices = set(r.index for r in results)
                recalls.append(compute_recall(self.ground_truths[i], result_indices))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        result = BenchmarkResult(
            name="Chunked Parallel",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_queries / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            recall=np.mean(recalls) if recalls else None,
            extra={
                "k": self.config.k,
                "chunk_size": self.config.chunk_size,
                "workers": self.config.workers
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_hnsw(self):
        """Benchmark HNSW approximate search with multiple ef_search values."""
        self.log("--- Benchmark: HNSW Approximate Search ---")

        if not HNSWLIB_AVAILABLE:
            self.log("  Skipped: hnswlib not available\n")
            return

        import hnswlib

        # Build HNSW index
        self.log("  Building HNSW index...")
        index = hnswlib.Index(space='cosine', dim=self.config.dimension)
        index.init_index(
            max_elements=self.config.num_records,
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_M
        )
        index.set_num_threads(self.config.workers)

        build_start = time.perf_counter()
        index.add_items(self.vectors, np.arange(self.config.num_records))
        build_time = time.perf_counter() - build_start
        self.log(f"  Build time: {build_time:.2f}s ({self.config.num_records/build_time:,.0f} vec/sec)")

        # Test multiple ef_search values to show recall/speed tradeoff
        ef_search_values = [50, 100, 200, 400]
        
        for ef_search in ef_search_values:
            index.set_ef(ef_search)

            # Search benchmark
            latencies = []
            recalls = []

            gc.collect()
            start = time.perf_counter()

            for i, q in enumerate(self.queries):
                t0 = time.perf_counter()
                labels, distances = index.knn_query(q.reshape(1, -1), k=self.config.k)
                latencies.append((time.perf_counter() - t0) * 1000)

                if i < len(self.ground_truths):
                    result_indices = set(labels[0].tolist())
                    recalls.append(compute_recall(self.ground_truths[i], result_indices))

            duration = time.perf_counter() - start
            percentiles = calculate_percentiles(latencies)

            result = BenchmarkResult(
                name=f"HNSW (ef={ef_search})",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=duration,
                operations_per_second=self.config.num_queries / duration,
                p50_latency_ms=percentiles.get("p50"),
                p95_latency_ms=percentiles.get("p95"),
                p99_latency_ms=percentiles.get("p99"),
                min_latency_ms=percentiles.get("min"),
                max_latency_ms=percentiles.get("max"),
                recall=np.mean(recalls) if recalls else None,
                extra={
                    "k": self.config.k,
                    "ef_search": ef_search,
                    "ef_construction": self.config.hnsw_ef_construction,
                    "M": self.config.hnsw_M,
                    "build_time_sec": round(build_time, 2),
                    "build_rate": f"{self.config.num_records/build_time:,.0f} vec/sec"
                }
            )
            self.results.append(result)
            self.log(str(result))
        
        self.log("")

    def benchmark_mmap(self):
        """Benchmark memory-mapped storage and search."""
        self.log("--- Benchmark: Memory-Mapped Storage + Search ---")

        mmap_path = os.path.join(self.temp_dir, "mmap_bench")
        
        # Create memory-mapped storage
        self.log("  Creating memory-mapped storage...")
        mmap_store = MemoryMappedVectors(mmap_path, self.config.dimension)
        
        create_start = time.perf_counter()
        mmap_store.create(self.config.num_records, self.config.dimension)
        mmap_store.append_batch(self.vectors)
        create_time = time.perf_counter() - create_start
        self.log(f"  Created in {create_time:.2f}s")

        # Search benchmark
        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, q in enumerate(self.queries):
            t0 = time.perf_counter()
            results = mmap_store.search_parallel(
                q, k=self.config.k, metric="cosine", engine=self.engine
            )
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < len(self.ground_truths):
                result_indices = set(r.index for r in results)
                recalls.append(compute_recall(self.ground_truths[i], result_indices))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        # Get file size
        mmap_file = os.path.join(mmap_path, "vectors.mmap")
        file_size_mb = os.path.getsize(mmap_file) / (1024 * 1024) if os.path.exists(mmap_file) else 0

        mmap_store.close()

        result = BenchmarkResult(
            name="Memory-Mapped Search",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_queries / duration,
            memory_mb=file_size_mb,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            recall=np.mean(recalls) if recalls else None,
            extra={
                "k": self.config.k,
                "file_size_mb": round(file_size_mb, 2),
                "create_time_sec": round(create_time, 2)
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_hybrid(self):
        """Benchmark ParallelCollection hybrid search."""
        self.log("--- Benchmark: ParallelCollection Hybrid Search ---")

        if not HNSWLIB_AVAILABLE:
            self.log("  Skipped: hnswlib not available\n")
            return

        collection_path = os.path.join(self.temp_dir, "hybrid_collection")
        
        # Create ParallelCollection
        self.log("  Creating ParallelCollection with HNSW...")
        collection = ParallelCollection(
            dimensions=self.config.dimension,
            n_workers=self.config.workers,
            max_elements=self.config.num_records * 2,
            use_mmap=True,
            mmap_path=collection_path
        )
        
        # Setup mmap storage
        collection._mmap_store.create(self.config.num_records, self.config.dimension)
        
        # Insert vectors
        insert_start = time.perf_counter()
        collection.insert_batch(self.vectors)
        insert_time = time.perf_counter() - insert_start
        self.log(f"  Inserted {self.config.num_records:,} vectors in {insert_time:.2f}s")

        # Hybrid search benchmark
        latencies = []
        recalls = []

        gc.collect()
        start = time.perf_counter()

        for i, q in enumerate(self.queries):
            t0 = time.perf_counter()
            results = collection.search_hybrid(q, k=self.config.k, hnsw_candidates=100)
            latencies.append((time.perf_counter() - t0) * 1000)

            if i < len(self.ground_truths):
                result_indices = set(r.index for r in results)
                recalls.append(compute_recall(self.ground_truths[i], result_indices))

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        collection._mmap_store.close()

        result = BenchmarkResult(
            name="Hybrid (HNSW + Re-rank)",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_queries / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            recall=np.mean(recalls) if recalls else None,
            extra={
                "k": self.config.k,
                "hnsw_candidates": 100,
                "insert_time_sec": round(insert_time, 2),
                "workers": self.config.workers
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def print_summary(self):
        """Print a summary table of all results."""
        self.log("\n" + "="*70)
        self.log("  BENCHMARK SUMMARY")
        self.log("="*70 + "\n")

        # Group by category
        categories = {
            "Exact Search": ["Naive", "Vectorized", "ParallelSearchEngine", "Chunked"],
            "Batch Search": ["Batch GEMM"],
            "Approximate": ["HNSW"],
            "Hybrid": ["Hybrid", "Memory-Mapped"],
        }

        for category, prefixes in categories.items():
            relevant = [r for r in self.results
                       if any(r.name.startswith(p) for p in prefixes)]
            if relevant:
                self.log(f"  {category}:")
                for r in relevant:
                    qps = f"{r.operations_per_second:,.0f} QPS"
                    latency = f"p50={r.p50_latency_ms:.2f}ms" if r.p50_latency_ms else ""
                    recall = f"recall={r.recall:.1%}" if r.recall is not None else ""
                    self.log(f"    {r.name}: {qps} {latency} {recall}".rstrip())
                self.log("")

        # Key metrics
        vectorized = next((r for r in self.results if r.name == "Vectorized BLAS"), None)
        # Find HNSW results - pick the one with best recall for key metrics
        hnsw_results = [r for r in self.results if r.name.startswith("HNSW")]
        hnsw = max(hnsw_results, key=lambda r: r.recall or 0) if hnsw_results else None
        hnsw_fast = min(hnsw_results, key=lambda r: r.p50_latency_ms or float('inf')) if hnsw_results else None
        hybrid = next((r for r in self.results if "Hybrid" in r.name), None)

        self.log("  KEY METRICS:")
        if vectorized:
            self.log(f"    Vectorized BLAS: {vectorized.operations_per_second:,.0f} QPS, {vectorized.p50_latency_ms:.2f}ms p50")
        if hnsw:
            self.log(f"    HNSW (best recall): {hnsw.operations_per_second:,.0f} QPS, {hnsw.p50_latency_ms:.3f}ms p50, {hnsw.recall:.1%} recall")
        if hnsw_fast and hnsw_fast != hnsw:
            self.log(f"    HNSW (fastest): {hnsw_fast.operations_per_second:,.0f} QPS, {hnsw_fast.p50_latency_ms:.3f}ms p50, {hnsw_fast.recall:.1%} recall")
        if hybrid:
            self.log(f"    Hybrid: {hybrid.operations_per_second:,.0f} QPS, {hybrid.p50_latency_ms:.3f}ms p50, {hybrid.recall:.1%} recall")

        # Speedup comparison
        naive = next((r for r in self.results if "Naive" in r.name), None)
        if naive and vectorized:
            speedup = vectorized.operations_per_second / naive.operations_per_second
            self.log(f"\n  SPEEDUP vs Naive:")
            self.log(f"    Vectorized BLAS: {speedup:.1f}x")
        if naive and hnsw:
            speedup = hnsw.operations_per_second / naive.operations_per_second
            self.log(f"    HNSW: {speedup:.0f}x")

    def export_json(self, path: str):
        """Export results to JSON."""
        data = {
            "benchmark_version": "3.0",
            "benchmark_type": "parallel",
            "timestamp": datetime.now().isoformat(),
            "system_info": get_system_info(),
            "config": {
                "num_records": self.config.num_records,
                "dimension": self.config.dimension,
                "num_queries": self.config.num_queries,
                "k": self.config.k,
                "workers": self.config.workers,
                "chunk_size": self.config.chunk_size,
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
    print(f"  BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"\n  File 1: {file1}")
    print(f"  File 2: {file2}")
    print(f"\n  Config 1: {data1['config']['num_records']:,} records, {data1['config']['workers']} workers")
    print(f"  Config 2: {data2['config']['num_records']:,} records, {data2['config']['workers']} workers")

    results1 = {r['name']: r for r in data1['results']}
    results2 = {r['name']: r for r in data2['results']}

    common = set(results1.keys()) & set(results2.keys())

    print(f"\n  {'Benchmark':<35} {'File1 QPS':>12} {'File2 QPS':>12} {'Change':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    for name in sorted(common):
        r1 = results1[name]
        r2 = results2[name]
        
        ops1 = r1.get('operations_per_second', 0)
        ops2 = r2.get('operations_per_second', 0)
        
        if ops1 > 0 and ops2 > 0:
            change = ((ops2 - ops1) / ops1) * 100
            change_str = f"{change:+.1f}%"
            print(f"  {name:<35} {ops1:>12,.0f} {ops2:>12,.0f} {change_str:>10}")

    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyVectorDB Phase 3: Parallel Processing Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  --quick     10K records (smoke test)
  --medium    100K records (naive skipped)
  --large     500K records (naive skipped)
  --stress    1M records (naive skipped)
  --extreme   2M records (naive skipped)

Benchmark Categories:
  naive       Naive Python loop baseline (skipped for medium+ presets)
  vectorized  Vectorized NumPy BLAS operations
  engine      ParallelSearchEngine single query
  batch       Batch GEMM matrix-matrix multiply
  chunked     Chunked parallel processing
  hnsw        HNSW approximate search
  mmap        Memory-mapped storage + search
  hybrid      ParallelCollection hybrid search

Examples:
  python benchmark_parallel.py --quick
  python benchmark_parallel.py --medium --workers 8
  python benchmark_parallel.py --large --only hnsw hybrid
  python benchmark_parallel.py --output results.json
  python benchmark_parallel.py --compare run1.json run2.json
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

    # Configuration options
    parser.add_argument("--dimension", "-d", type=int, default=128,
                       help="Vector dimension (default: 128)")
    parser.add_argument("--queries", type=int, default=100,
                       help="Number of queries (default: 100)")
    parser.add_argument("--k", type=int, default=10,
                       help="Top-k results (default: 10)")
    parser.add_argument("--workers", "-w", type=int, default=0,
                       help="Number of workers (default: CPU count)")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                       help="Chunk size for parallel processing (default: 50000)")

    # Output options
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

    # Determine preset
    preset = "default"
    if args.quick:
        preset = "quick"
        num_records = PRESETS["quick"]["records"]
    elif args.medium:
        preset = "medium"
        num_records = PRESETS["medium"]["records"]
    elif args.large:
        preset = "large"
        num_records = PRESETS["large"]["records"]
    elif args.stress:
        preset = "stress"
        num_records = PRESETS["stress"]["records"]
    elif args.extreme:
        preset = "extreme"
        num_records = PRESETS["extreme"]["records"]
    else:
        num_records = args.records

    config = BenchmarkConfig(
        num_records=num_records,
        dimension=args.dimension,
        num_queries=args.queries,
        k=args.k,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    benchmarker = ParallelBenchmarker(
        config=config,
        verbose=not args.quiet,
        only=args.only,
        skip=args.skip,
        preset=preset,
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
