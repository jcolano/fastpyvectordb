#!/usr/bin/env python3
"""
Benchmark Comparison: Original vs Optimized VectorDB

Compares performance of the original implementation vs Phase 1 optimizations.
"""

import numpy as np
import time
import tempfile
import shutil
import sys

# Import both implementations
from vectordb import VectorDB as OriginalVectorDB, Filter as OriginalFilter
from vectordb_optimized import VectorDB as OptimizedVectorDB, Filter as OptimizedFilter


def benchmark_implementation(db_class, filter_class, name: str,
                              n_vectors: int, dimensions: int,
                              n_queries: int, k: int) -> dict:
    """Run benchmark on a specific implementation."""
    results = {"name": name}

    # Create temp database
    temp_dir = tempfile.mkdtemp()
    db = db_class(temp_dir)
    collection = db.create_collection("benchmark", dimensions=dimensions)

    # Generate data
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    ids = [f"vec_{i}" for i in range(n_vectors)]
    metadata_list = [
        {"category": np.random.choice(["A", "B", "C", "D"]),
         "value": float(np.random.rand())}
        for _ in range(n_vectors)
    ]

    # Benchmark: Batch Insert
    start = time.perf_counter()
    collection.insert_batch(vectors, ids, metadata_list)
    insert_time = time.perf_counter() - start
    results["insert_time"] = insert_time
    results["insert_rate"] = n_vectors / insert_time

    # Generate queries
    queries = np.random.randn(n_queries, dimensions).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Benchmark: Single Search
    search_times = []
    for q in queries:
        start = time.perf_counter()
        results_list = collection.search(q, k=k)
        search_times.append(time.perf_counter() - start)

    results["search_avg_ms"] = np.mean(search_times) * 1000
    results["search_p99_ms"] = np.percentile(search_times, 99) * 1000
    results["search_qps"] = 1000 / results["search_avg_ms"]

    # Benchmark: Batch Search
    start = time.perf_counter()
    all_results = collection.search_batch(queries, k=k)
    batch_time = time.perf_counter() - start
    results["batch_total_ms"] = batch_time * 1000
    results["batch_per_query_ms"] = batch_time * 1000 / n_queries

    # Benchmark: Filtered Search
    filter_obj = filter_class.eq("category", "A")
    filter_times = []
    for q in queries[:20]:
        start = time.perf_counter()
        results_list = collection.search(q, k=k, filter=filter_obj)
        filter_times.append(time.perf_counter() - start)

    results["filtered_avg_ms"] = np.mean(filter_times) * 1000

    # Cleanup
    shutil.rmtree(temp_dir)

    return results


def print_comparison(original: dict, optimized: dict):
    """Print side-by-side comparison."""

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPARISON: Original vs Optimized")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Original':>15} {'Optimized':>15} {'Speedup':>12}")
    print("-" * 80)

    metrics = [
        ("Insert Rate (vec/sec)", "insert_rate", "{:,.0f}"),
        ("Search Avg (ms)", "search_avg_ms", "{:.3f}"),
        ("Search P99 (ms)", "search_p99_ms", "{:.3f}"),
        ("Search QPS", "search_qps", "{:,.0f}"),
        ("Batch Total (ms)", "batch_total_ms", "{:.2f}"),
        ("Batch Per Query (ms)", "batch_per_query_ms", "{:.3f}"),
        ("Filtered Search (ms)", "filtered_avg_ms", "{:.3f}"),
    ]

    for label, key, fmt in metrics:
        orig_val = original[key]
        opt_val = optimized[key]

        # Calculate speedup (higher is better for rate/qps, lower is better for time)
        if "rate" in key.lower() or "qps" in key.lower():
            speedup = opt_val / orig_val if orig_val > 0 else 0
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = orig_val / opt_val if opt_val > 0 else 0
            speedup_str = f"{speedup:.2f}x"

        print(f"{label:<30} {fmt.format(orig_val):>15} {fmt.format(opt_val):>15} {speedup_str:>12}")

    print("-" * 80)

    # Calculate overall improvement
    search_speedup = original["search_avg_ms"] / optimized["search_avg_ms"]
    batch_speedup = original["batch_total_ms"] / optimized["batch_total_ms"]

    print(f"\n{'OVERALL SEARCH SPEEDUP:':<30} {' ':>15} {' ':>15} {search_speedup:>10.2f}x")
    print(f"{'BATCH SEARCH SPEEDUP:':<30} {' ':>15} {' ':>15} {batch_speedup:>10.2f}x")


def main():
    # Test parameters
    n_vectors = 10000  # Reduced for faster comparison
    dimensions = 128
    n_queries = 50
    k = 10

    print("=" * 80)
    print("  VectorDB Performance Comparison")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Dimensions: {dimensions}")
    print(f"  Queries: {n_queries}")
    print(f"  Top-K: {k}")

    # Run benchmarks
    print("\n" + "-" * 40)
    print("Running Original Implementation...")
    print("-" * 40)
    original_results = benchmark_implementation(
        OriginalVectorDB, OriginalFilter, "Original",
        n_vectors, dimensions, n_queries, k
    )
    print(f"  Insert: {original_results['insert_rate']:,.0f} vec/sec")
    print(f"  Search: {original_results['search_avg_ms']:.3f} ms/query")

    print("\n" + "-" * 40)
    print("Running Optimized Implementation...")
    print("-" * 40)
    optimized_results = benchmark_implementation(
        OptimizedVectorDB, OptimizedFilter, "Optimized",
        n_vectors, dimensions, n_queries, k
    )
    print(f"  Insert: {optimized_results['insert_rate']:,.0f} vec/sec")
    print(f"  Search: {optimized_results['search_avg_ms']:.3f} ms/query")

    # Print comparison
    print_comparison(original_results, optimized_results)

    print("\n" + "=" * 80)
    print("  Phase 1 Optimizations Applied:")
    print("=" * 80)
    print("""
  1. Vector Matrix Caching - O(1) access to all vectors
  2. Vectorized Distance Calculation - NumPy BLAS operations
  3. O(n) Top-K Selection - np.argpartition vs O(n log n) sort
  4. Native HNSW Batch Queries - Single call for multiple queries
  5. Pre-allocated Arrays - Reduces memory allocation overhead
  6. Contiguous Memory Layout - Cache-friendly np.ascontiguousarray
    """)


if __name__ == "__main__":
    main()
