#!/usr/bin/env python3
"""
RuVector Python Benchmark Suite (v2)

Comprehensive performance testing for the vector database.
Optimized for large-scale benchmarks (up to 1M+ records).

Usage:
    python benchmark.py                      # Run all benchmarks with defaults (50K)
    python benchmark.py --records 1000000    # Test with 1M records
    python benchmark.py --quick              # Quick benchmark (10K records)
    python benchmark.py --medium             # Medium benchmark (100K records)
    python benchmark.py --large              # Large benchmark (500K records)
    python benchmark.py --stress             # Stress test (1M records)
    python benchmark.py --only insert search # Run only specific benchmarks
    python benchmark.py --skip memory        # Skip specific benchmarks
    python benchmark.py --dimension 384      # Test with 384-dim vectors (e.g., sentence-transformers)
    python benchmark.py --compare file1.json file2.json  # Compare two benchmark results
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
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import platform

import numpy as np

# Import our modules
from vectordb import VectorDB, Filter, Collection


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
    "insert",      # Sequential insertion
    "bulk",        # Bulk/batch insertion
    "search",      # Basic search
    "search_k",    # Search with varying k
    "filtered",    # Filtered search
    "batch",       # Batch search
    "upsert",      # Update operations
    "delete",      # Delete operations
    "persistence", # Save/load
    "memory",      # Memory scaling
]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    num_records: int = 50_000
    dimension: int = 128
    batch_size: int = 5_000          # Batch size for bulk operations
    num_search_queries: int = 1_000   # Number of search queries
    latency_sample_rate: float = 0.1  # Sample 10% of latencies for large runs
    max_latency_samples: int = 10_000 # Cap latency samples
    progress_interval: int = 0        # Auto-calculated if 0
    chunk_size: int = 50_000          # Generate vectors in chunks
    
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
        "p95": latencies_sorted[int(n * 0.95)],
        "p99": latencies_sorted[int(n * 0.99)],
        "min": latencies_sorted[0],
        "max": latencies_sorted[-1],
    }


class LatencySampler:
    """
    Efficient latency sampling for large-scale benchmarks.
    Uses reservoir sampling to maintain a fixed-size sample.
    """
    
    def __init__(self, max_samples: int = 10_000, sample_rate: float = 1.0):
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.samples: List[float] = []
        self.count = 0
        self._rng = np.random.default_rng(42)
    
    def add(self, latency_ms: float):
        """Add a latency measurement (may be sampled)."""
        self.count += 1
        
        # For small datasets, keep everything
        if self.sample_rate >= 1.0 and len(self.samples) < self.max_samples:
            self.samples.append(latency_ms)
            return
        
        # Probabilistic sampling for large datasets
        if self._rng.random() < self.sample_rate:
            if len(self.samples) < self.max_samples:
                self.samples.append(latency_ms)
            else:
                # Reservoir sampling
                idx = self._rng.integers(0, self.count)
                if idx < self.max_samples:
                    self.samples[idx] = latency_ms
    
    def get_percentiles(self) -> Dict[str, float]:
        """Get percentile statistics."""
        return calculate_percentiles(self.samples)


def generate_vectors_chunked(n: int, dim: int, seed: int = 42, chunk_size: int = 50_000):
    """
    Generator that yields normalized vectors in chunks.
    Memory-efficient for large datasets.
    """
    rng = np.random.default_rng(seed)
    
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        count = end - start
        
        vectors = rng.standard_normal((count, dim), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        yield start, vectors


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random normalized vectors (for smaller datasets)."""
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


def estimate_benchmark_time(num_records: int) -> str:
    """Estimate total benchmark time based on record count."""
    # Rough estimates based on typical performance
    insert_time = num_records / 7000  # ~7K inserts/sec
    search_time = 2  # Fixed search benchmarks
    other_time = num_records / 20000  # Other operations
    
    total_seconds = insert_time * 2 + search_time + other_time  # x2 for sequential + bulk
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds / 60)} minutes"
    else:
        return f"~{total_seconds / 3600:.1f} hours"


# =============================================================================
# Benchmarker Class
# =============================================================================

class Benchmarker:
    """Runs comprehensive benchmarks on the vector database."""

    def __init__(self, config: BenchmarkConfig, verbose: bool = True,
                 only: List[str] = None, skip: List[str] = None):
        self.config = config
        self.verbose = verbose
        self.only = set(only) if only else None
        self.skip = set(skip) if skip else set()
        self.results: List[BenchmarkResult] = []
        
        # Create temp directory for benchmark databases
        self.temp_dir = tempfile.mkdtemp(prefix="vectordb_bench_")
        self._db_counter = 0
        
        # Will be populated during run
        self.db: Optional[VectorDB] = None
        self.collection: Optional[Collection] = None
        self.query_vectors: Optional[np.ndarray] = None

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
        self.log(f"  RUVECTOR PYTHON BENCHMARK SUITE v2")
        self.log(f"  Records: {self.config.num_records:,} | Dimension: {self.config.dimension}")
        self.log(f"  Estimated time: {estimate_benchmark_time(self.config.num_records)}")
        self.log(f"{'='*70}\n")

        # Generate query vectors (always needed, small dataset)
        self.log("Generating query vectors...")
        self.query_vectors = generate_vectors(1000, self.config.dimension, seed=999)
        self.log(f"  Generated 1,000 query vectors\n")

        # Run benchmarks based on selection
        benchmark_methods = [
            ("insert", self.benchmark_insertion),
            ("bulk", self.benchmark_bulk_insertion),
            ("search", self.benchmark_search),
            ("search_k", self.benchmark_search_with_k),
            ("filtered", self.benchmark_filtered_search),
            ("batch", self.benchmark_batch_search),
            ("upsert", self.benchmark_upsert),
            ("delete", self.benchmark_delete),
            ("persistence", self.benchmark_persistence),
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

    def benchmark_insertion(self):
        """Benchmark sequential insertions with chunked vector generation."""
        self.log("--- Benchmark: Sequential Insertion ---")

        db = VectorDB(self._get_db_path())
        collection = db.create_collection("bench_insert", self.config.dimension)

        gc.collect()
        mem_before = get_memory_usage_mb()
        
        # Determine sampling rate based on dataset size
        sample_rate = min(1.0, self.config.max_latency_samples / self.config.num_records)
        latency_sampler = LatencySampler(
            max_samples=self.config.max_latency_samples,
            sample_rate=sample_rate
        )

        start = time.perf_counter()
        records_inserted = 0

        # Generate and insert in chunks to save memory
        for chunk_start, vectors in generate_vectors_chunked(
            self.config.num_records, 
            self.config.dimension,
            chunk_size=self.config.chunk_size
        ):
            metadata_batch = generate_metadata_batch(chunk_start, len(vectors))
            
            for i, (vector, meta) in enumerate(zip(vectors, metadata_batch)):
                idx = chunk_start + i
                t0 = time.perf_counter()
                collection.insert(
                    id=f"vec_{idx}",
                    vector=vector.tolist(),
                    metadata=meta
                )
                latency_sampler.add((time.perf_counter() - t0) * 1000)
                records_inserted += 1

                if records_inserted % self.config.progress_interval == 0:
                    elapsed = time.perf_counter() - start
                    rate = records_inserted / elapsed
                    eta = (self.config.num_records - records_inserted) / rate
                    self.log(f"  Inserted {records_inserted:,} records... ({rate:,.0f}/s, ETA: {eta:.0f}s)")

        duration = time.perf_counter() - start
        mem_after = get_memory_usage_mb()

        percentiles = latency_sampler.get_percentiles()
        result = BenchmarkResult(
            name="Sequential Insertion",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_records / duration,
            memory_mb=mem_after - mem_before,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"latency_samples": len(latency_sampler.samples)}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

        # Store for later benchmarks
        self.db = db
        self.collection = collection

    def benchmark_bulk_insertion(self):
        """Benchmark true batch insertions using insert_batch()."""
        self.log("--- Benchmark: Bulk Insertion ---")

        db = VectorDB(self._get_db_path())
        collection = db.create_collection("bench_bulk", self.config.dimension)

        batch_size = self.config.batch_size
        gc.collect()
        
        start = time.perf_counter()
        records_inserted = 0
        batch_times = []

        # Generate and insert in batches
        for chunk_start, vectors in generate_vectors_chunked(
            self.config.num_records,
            self.config.dimension,
            chunk_size=batch_size
        ):
            metadata_batch = generate_metadata_batch(chunk_start, len(vectors))
            ids = [f"vec_{chunk_start + i}" for i in range(len(vectors))]
            
            t0 = time.perf_counter()
            # Use the actual batch insert method
            collection.insert_batch(
                vectors=vectors,
                ids=ids,
                metadata_list=metadata_batch
            )
            batch_times.append((time.perf_counter() - t0) * 1000)
            
            records_inserted += len(vectors)
            
            if records_inserted % self.config.progress_interval == 0:
                elapsed = time.perf_counter() - start
                rate = records_inserted / elapsed
                self.log(f"  Bulk inserted {records_inserted:,} records... ({rate:,.0f}/s)")

        duration = time.perf_counter() - start

        # Calculate batch-level latencies
        percentiles = calculate_percentiles(batch_times)
        
        result = BenchmarkResult(
            name=f"Bulk Insertion (batch={batch_size:,})",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=self.config.num_records / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            extra={
                "batch_size": batch_size,
                "num_batches": len(batch_times),
                "avg_batch_ms": sum(batch_times) / len(batch_times) if batch_times else 0
            }
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_search(self):
        """Benchmark single-query search performance."""
        self.log("--- Benchmark: Single Query Search (k=10) ---")

        if self.collection is None:
            self.log("  Skipped: No collection available (run 'insert' first)\n")
            return

        num_queries = self.config.num_search_queries
        latencies = []

        # Warm up
        for i in range(min(10, num_queries)):
            self.collection.search(self.query_vectors[i].tolist(), k=10)

        gc.collect()
        start = time.perf_counter()

        for i in range(num_queries):
            t0 = time.perf_counter()
            self.collection.search(
                self.query_vectors[i % len(self.query_vectors)].tolist(), 
                k=10
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        duration = time.perf_counter() - start

        percentiles = calculate_percentiles(latencies)
        result = BenchmarkResult(
            name="Single Query Search (k=10)",
            records=self.config.num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=num_queries / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
            min_latency_ms=percentiles.get("min"),
            max_latency_ms=percentiles.get("max"),
            extra={"num_queries": num_queries}
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_search_with_k(self):
        """Benchmark search with different k values."""
        self.log("--- Benchmark: Search with Varying k ---")

        if self.collection is None:
            self.log("  Skipped: No collection available\n")
            return

        # Scale k values based on dataset size
        k_values = [1, 10, 50, 100]
        if self.config.num_records >= 100_000:
            k_values.extend([500, 1000])
        if self.config.num_records >= 500_000:
            k_values.append(5000)
        
        num_queries = 100

        for k in k_values:
            if k > self.config.num_records:
                continue

            latencies = []
            start = time.perf_counter()

            for i in range(num_queries):
                t0 = time.perf_counter()
                self.collection.search(
                    self.query_vectors[i % len(self.query_vectors)].tolist(), 
                    k=k
                )
                latencies.append((time.perf_counter() - t0) * 1000)

            duration = time.perf_counter() - start
            percentiles = calculate_percentiles(latencies)

            result = BenchmarkResult(
                name=f"Search k={k}",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=duration,
                operations_per_second=num_queries / duration,
                p50_latency_ms=percentiles.get("p50"),
                p95_latency_ms=percentiles.get("p95"),
                p99_latency_ms=percentiles.get("p99"),
            )
            self.results.append(result)
            self.log(str(result))

        self.log("")

    def benchmark_filtered_search(self):
        """Benchmark search with metadata filtering."""
        self.log("--- Benchmark: Filtered Search ---")

        if self.collection is None:
            self.log("  Skipped: No collection available\n")
            return

        num_queries = 100

        # Test different filter selectivities
        filters = [
            ("category='electronics' (~17%)", Filter.eq("category", "electronics")),
            ("price < 100 (~9%)", Filter.lt("price", 100)),
            ("rating > 4.0 (~20%)", Filter.gt("rating", 4.0)),
            ("in_stock = true (~70%)", Filter.eq("in_stock", True)),
            ("category='books' AND price<500 (~8%)",
             Filter.and_([Filter.eq("category", "books"), Filter.lt("price", 500)])),
            ("rating BETWEEN 3-4 (~25%)",
             Filter.and_([Filter.gte("rating", 3.0), Filter.lte("rating", 4.0)])),
        ]

        for filter_name, filter_obj in filters:
            latencies = []
            start = time.perf_counter()

            for i in range(num_queries):
                t0 = time.perf_counter()
                self.collection.search(
                    self.query_vectors[i % len(self.query_vectors)].tolist(),
                    k=10,
                    filter=filter_obj
                )
                latencies.append((time.perf_counter() - t0) * 1000)

            duration = time.perf_counter() - start
            percentiles = calculate_percentiles(latencies)

            result = BenchmarkResult(
                name=f"Filtered: {filter_name}",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=duration,
                operations_per_second=num_queries / duration,
                p50_latency_ms=percentiles.get("p50"),
                p95_latency_ms=percentiles.get("p95"),
                p99_latency_ms=percentiles.get("p99"),
            )
            self.results.append(result)
            self.log(str(result))

        self.log("")

    def benchmark_batch_search(self):
        """Benchmark batch search performance."""
        self.log("--- Benchmark: Batch Search ---")

        if self.collection is None:
            self.log("  Skipped: No collection available\n")
            return

        batch_sizes = [10, 50, 100, 500]

        for batch_size in batch_sizes:
            num_batches = 10
            total_queries = num_batches * batch_size
            
            # Prepare query batches
            query_batches = []
            for b in range(num_batches):
                batch = self.query_vectors[b * batch_size % len(self.query_vectors):
                                          (b * batch_size + batch_size) % len(self.query_vectors) + batch_size]
                if len(batch) < batch_size:
                    batch = self.query_vectors[:batch_size]
                query_batches.append(batch)

            start = time.perf_counter()

            for batch in query_batches:
                # Use search_batch if available, otherwise sequential
                if hasattr(self.collection, 'search_batch'):
                    self.collection.search_batch(batch, k=10)
                else:
                    for query in batch:
                        self.collection.search(query.tolist(), k=10)

            duration = time.perf_counter() - start

            result = BenchmarkResult(
                name=f"Batch Search (batch={batch_size})",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=duration,
                operations_per_second=total_queries / duration,
                extra={"total_queries": total_queries, "num_batches": num_batches}
            )
            self.results.append(result)
            self.log(str(result))

        self.log("")

    def benchmark_upsert(self):
        """Benchmark upsert operations."""
        self.log("--- Benchmark: Upsert Operations ---")

        if self.collection is None:
            self.log("  Skipped: No collection available\n")
            return

        # Scale upsert count with dataset size
        num_upserts = min(5000, self.config.num_records // 10)
        new_vectors = generate_vectors(num_upserts, self.config.dimension, seed=888)

        latencies = []
        start = time.perf_counter()

        for i in range(num_upserts):
            t0 = time.perf_counter()
            self.collection.upsert(
                vector=new_vectors[i].tolist(),
                id=f"vec_{i}",
                metadata={"category": "updated", "price": 999.99, "version": 2}
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        result = BenchmarkResult(
            name="Upsert Operations",
            records=num_upserts,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=num_upserts / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_delete(self):
        """Benchmark delete operations."""
        self.log("--- Benchmark: Delete Operations ---")

        # Create a fresh collection for delete test
        db = VectorDB(self._get_db_path())
        collection = db.create_collection("bench_delete", self.config.dimension)

        # Scale delete test size
        num_records = min(50_000, self.config.num_records)
        
        self.log(f"  Preparing {num_records:,} records for deletion test...")
        
        # Insert records
        for chunk_start, vectors in generate_vectors_chunked(
            num_records,
            self.config.dimension,
            chunk_size=min(10_000, num_records)
        ):
            metadata_batch = generate_metadata_batch(chunk_start, len(vectors))
            ids = [f"del_{chunk_start + i}" for i in range(len(vectors))]
            collection.insert_batch(vectors=vectors, ids=ids, metadata_list=metadata_batch)

        latencies = []
        start = time.perf_counter()

        for i in range(num_records):
            t0 = time.perf_counter()
            collection.delete(f"del_{i}")
            latencies.append((time.perf_counter() - t0) * 1000)

        duration = time.perf_counter() - start
        percentiles = calculate_percentiles(latencies)

        result = BenchmarkResult(
            name="Delete Operations",
            records=num_records,
            dimension=self.config.dimension,
            duration_seconds=duration,
            operations_per_second=num_records / duration,
            p50_latency_ms=percentiles.get("p50"),
            p95_latency_ms=percentiles.get("p95"),
            p99_latency_ms=percentiles.get("p99"),
        )
        self.results.append(result)
        self.log(str(result) + "\n")

    def benchmark_persistence(self):
        """Benchmark save/load operations."""
        self.log("--- Benchmark: Persistence (Save/Load) ---")

        if self.db is None:
            self.log("  Skipped: No database available\n")
            return

        db_path = self.db.path

        try:
            # Save benchmark
            start = time.perf_counter()
            self.db.save()
            save_duration = time.perf_counter() - start

            # Get file size
            total_size = 0
            for root, dirs, files in os.walk(db_path):
                for f in files:
                    total_size += os.path.getsize(os.path.join(root, f))
            size_mb = total_size / (1024 * 1024)

            result = BenchmarkResult(
                name="Save to Disk",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=save_duration,
                operations_per_second=self.config.num_records / save_duration,
                extra={
                    "file_size_mb": f"{size_mb:.2f}",
                    "mb_per_second": f"{size_mb / save_duration:.1f}"
                }
            )
            self.results.append(result)
            self.log(str(result))

            # Load benchmark
            gc.collect()
            start = time.perf_counter()
            loaded_db = VectorDB(db_path)  # This auto-loads
            load_duration = time.perf_counter() - start

            # Verify load
            loaded_collection = loaded_db.get_collection("bench_insert")
            loaded_count = len(loaded_collection)

            result = BenchmarkResult(
                name="Load from Disk",
                records=self.config.num_records,
                dimension=self.config.dimension,
                duration_seconds=load_duration,
                operations_per_second=self.config.num_records / load_duration,
                extra={
                    "verified_count": loaded_count,
                    "mb_per_second": f"{size_mb / load_duration:.1f}"
                }
            )
            self.results.append(result)
            self.log(str(result) + "\n")

        except Exception as e:
            self.log(f"  Persistence benchmark failed: {e}\n")

    def benchmark_memory_scaling(self):
        """Benchmark memory usage at different scales."""
        self.log("--- Benchmark: Memory Scaling ---")

        # Dynamic scales based on target record count
        base_scales = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
        scales = [s for s in base_scales if s <= self.config.num_records]
        
        # Ensure we test at least a few scales
        if len(scales) < 3 and self.config.num_records >= 1000:
            scales = [s for s in [1000, self.config.num_records // 2, self.config.num_records] if s > 0]

        for scale in scales:
            gc.collect()
            time.sleep(0.1)  # Let GC settle
            mem_before = get_memory_usage_mb()

            db = VectorDB(self._get_db_path())
            collection = db.create_collection(f"mem_scale_{scale}", self.config.dimension)

            # Use batch insert for speed
            for chunk_start, vectors in generate_vectors_chunked(
                scale, 
                self.config.dimension, 
                chunk_size=min(10_000, scale)
            ):
                metadata_batch = generate_metadata_batch(chunk_start, len(vectors))
                ids = [f"v_{chunk_start + i}" for i in range(len(vectors))]
                collection.insert_batch(vectors=vectors, ids=ids, metadata_list=metadata_batch)

            gc.collect()
            time.sleep(0.1)
            mem_after = get_memory_usage_mb()
            mem_used = max(0, mem_after - mem_before)

            bytes_per_record = (mem_used * 1024 * 1024) / scale if mem_used > 0 else 0

            result = BenchmarkResult(
                name=f"Memory at {scale:,} records",
                records=scale,
                dimension=self.config.dimension,
                duration_seconds=0,
                operations_per_second=0,
                memory_mb=mem_used,
                extra={"bytes_per_record": f"{bytes_per_record:.1f}"}
            )
            self.results.append(result)
            self.log(str(result))

            # Cleanup to free memory for next iteration
            del db, collection
            gc.collect()

        self.log("")

    def print_summary(self):
        """Print a summary table of all results."""
        self.log("\n" + "="*70)
        self.log("  BENCHMARK SUMMARY")
        self.log("="*70 + "\n")

        # Group by category
        categories = {
            "Insertion": ["Sequential Insertion", "Bulk Insertion"],
            "Search": ["Single Query Search", "Search k="],
            "Filtered": ["Filtered:"],
            "Batch": ["Batch Search"],
            "Mutations": ["Upsert", "Delete"],
            "Persistence": ["Save", "Load"],
            "Memory": ["Memory at"],
        }

        for category, prefixes in categories.items():
            relevant = [r for r in self.results
                       if any(r.name.startswith(p) for p in prefixes)]
            if relevant:
                self.log(f"  {category}:")
                for r in relevant:
                    throughput = f"{r.operations_per_second:,.0f} ops/s" if r.operations_per_second > 0 else "N/A"
                    latency = f"p50={r.p50_latency_ms:.2f}ms" if r.p50_latency_ms else ""
                    memory = f"mem={r.memory_mb:.1f}MB" if r.memory_mb and r.memory_mb > 0 else ""
                    self.log(f"    {r.name}: {throughput} {latency} {memory}".rstrip())
                self.log("")

        # Key metrics summary
        insert_result = next((r for r in self.results if r.name == "Sequential Insertion"), None)
        bulk_result = next((r for r in self.results if "Bulk Insertion" in r.name), None)
        search_result = next((r for r in self.results if r.name == "Single Query Search (k=10)"), None)

        self.log("  KEY METRICS:")
        if insert_result:
            self.log(f"    Sequential insert: {insert_result.operations_per_second:,.0f} vectors/sec")
        if bulk_result:
            self.log(f"    Bulk insert: {bulk_result.operations_per_second:,.0f} vectors/sec")
        if search_result:
            self.log(f"    Search throughput: {search_result.operations_per_second:,.0f} queries/sec")
            self.log(f"    Search latency p50: {search_result.p50_latency_ms:.3f} ms")
            self.log(f"    Search latency p99: {search_result.p99_latency_ms:.3f} ms")
        if insert_result and insert_result.memory_mb:
            self.log(f"    Memory usage: {insert_result.memory_mb:.1f} MB for {self.config.num_records:,} vectors")

    def export_json(self, path: str):
        """Export results to JSON with full metadata."""
        data = {
            "benchmark_version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "system_info": get_system_info(),
            "config": {
                "num_records": self.config.num_records,
                "dimension": self.config.dimension,
                "batch_size": self.config.batch_size,
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
    print(f"\n  Config 1: {data1['config']['num_records']:,} records, dim={data1['config']['dimension']}")
    print(f"  Config 2: {data2['config']['num_records']:,} records, dim={data2['config']['dimension']}")

    # Build result maps
    results1 = {r['name']: r for r in data1['results']}
    results2 = {r['name']: r for r in data2['results']}

    # Find common benchmarks
    common = set(results1.keys()) & set(results2.keys())

    print(f"\n  {'Benchmark':<40} {'File1':>12} {'File2':>12} {'Change':>10}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*10}")

    for name in sorted(common):
        r1 = results1[name]
        r2 = results2[name]
        
        ops1 = r1.get('operations_per_second', 0)
        ops2 = r2.get('operations_per_second', 0)
        
        if ops1 > 0 and ops2 > 0:
            change = ((ops2 - ops1) / ops1) * 100
            change_str = f"{change:+.1f}%"
            print(f"  {name:<40} {ops1:>12,.0f} {ops2:>12,.0f} {change_str:>10}")

    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RuVector Python Benchmark Suite v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  --quick     10K records (smoke test)
  --medium    100K records
  --large     500K records  
  --stress    1M records
  --extreme   2M records

Examples:
  python benchmark.py --quick                    # Quick test
  python benchmark.py --records 250000           # Custom size
  python benchmark.py --large --only insert search  # Large, specific benchmarks
  python benchmark.py --stress --skip memory     # Stress test, skip memory scaling
  python benchmark.py --compare run1.json run2.json  # Compare results
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
    parser.add_argument("--batch-size", "-b", type=int, default=5_000,
                       help="Batch size for bulk operations (default: 5000)")
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
        batch_size=args.batch_size,
    )

    benchmarker = Benchmarker(
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
