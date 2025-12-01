#!/usr/bin/env python3
"""
RuVector Python Demo - Interactive showcase of all features

Run: python demo.py
"""

import numpy as np
import time
import json
from typing import List

# Import our modules
from vectordb import VectorDB, Filter
from graph import GraphDB, NodeBuilder, EdgeBuilder
from embeddings import MockEmbedder, EmbeddingCollection
from realtime import EventBus, ObservableCollection, EventType


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_section(title: str):
    print(f"\n--- {title} ---\n")


# =============================================================================
# DEMO 1: Vector Search Basics
# =============================================================================
def demo_vector_search():
    print_header("DEMO 1: Vector Search with HNSW")

    import tempfile
    temp_dir = tempfile.mkdtemp()
    db = VectorDB(temp_dir)

    # Create a collection for product embeddings
    collection = db.create_collection(
        name="products",
        dimensions=128,
        metric="cosine"
    )

    # Simulate product embeddings (in real use, these come from an embedding model)
    products = [
        {"id": "p1", "name": "Red Running Shoes", "category": "footwear", "price": 89.99},
        {"id": "p2", "name": "Blue Running Shoes", "category": "footwear", "price": 79.99},
        {"id": "p3", "name": "Black Leather Boots", "category": "footwear", "price": 149.99},
        {"id": "p4", "name": "White Sneakers", "category": "footwear", "price": 69.99},
        {"id": "p5", "name": "Red T-Shirt", "category": "clothing", "price": 29.99},
        {"id": "p6", "name": "Blue Jeans", "category": "clothing", "price": 59.99},
        {"id": "p7", "name": "Black Jacket", "category": "clothing", "price": 129.99},
        {"id": "p8", "name": "Running Shorts", "category": "clothing", "price": 34.99},
    ]

    print_section("Adding products to vector database")
    np.random.seed(42)  # For reproducibility

    # Create embeddings that cluster similar items
    base_embeddings = {
        "running": np.random.randn(128) * 0.1,
        "footwear": np.random.randn(128) * 0.1,
        "clothing": np.random.randn(128) * 0.1,
        "red": np.random.randn(128) * 0.1,
        "blue": np.random.randn(128) * 0.1,
    }

    for product in products:
        # Create a composite embedding based on product attributes
        embedding = np.zeros(128)
        name_lower = product["name"].lower()

        if "running" in name_lower:
            embedding += base_embeddings["running"]
        if product["category"] == "footwear":
            embedding += base_embeddings["footwear"]
        if product["category"] == "clothing":
            embedding += base_embeddings["clothing"]
        if "red" in name_lower:
            embedding += base_embeddings["red"]
        if "blue" in name_lower:
            embedding += base_embeddings["blue"]

        # Add some noise
        embedding += np.random.randn(128) * 0.05
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        collection.insert(
            vector=embedding.tolist(),
            id=product["id"],
            metadata=product
        )
        print(f"  Added: {product['name']}")

    print_section("Semantic Search: 'running gear'")
    # Query for running-related items
    query_embedding = base_embeddings["running"] + base_embeddings["footwear"]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    results = collection.search(query_embedding.tolist(), k=5)
    print("Top 5 results for 'running gear':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.metadata['name']} (score: {r.score:.4f})")

    print_section("Filtered Search: Footwear under $100")
    filter_obj = Filter.and_([
        Filter.eq("category", "footwear"),
        Filter.lt("price", 100)
    ])

    results = collection.search(query_embedding.tolist(), k=5, filter=filter_obj)
    print("Footwear under $100:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.metadata['name']} - ${r.metadata['price']}")

    return db


# =============================================================================
# DEMO 2: Graph Database
# =============================================================================
def demo_graph_database():
    print_header("DEMO 2: Graph Database with Traversal")

    graph = GraphDB()

    print_section("Building a social network graph")

    # Create users
    users = [
        ("alice", {"name": "Alice", "age": 28, "city": "NYC"}),
        ("bob", {"name": "Bob", "age": 32, "city": "LA"}),
        ("charlie", {"name": "Charlie", "age": 25, "city": "NYC"}),
        ("diana", {"name": "Diana", "age": 30, "city": "Chicago"}),
        ("eve", {"name": "Eve", "age": 27, "city": "NYC"}),
    ]

    for user_id, props in users:
        graph.create_node(
            NodeBuilder(user_id)
            .label("Person")
            .properties(props)
            .build()
        )
        print(f"  Added user: {props['name']}")

    # Create interests
    interests = ["Python", "Rust", "AI", "Music", "Sports"]
    for interest in interests:
        graph.create_node(
            NodeBuilder(interest.lower())
            .label("Interest")
            .property("name", interest)
            .build()
        )

    print_section("Creating relationships")

    # Friendships
    friendships = [
        ("alice", "bob"), ("alice", "charlie"), ("bob", "diana"),
        ("charlie", "eve"), ("diana", "eve")
    ]
    for u1, u2 in friendships:
        graph.create_edge(
            EdgeBuilder(u1, u2, "FRIENDS_WITH")
            .property("since", 2023)
            .build()
        )
        print(f"  {u1.title()} <-> {u2.title()} (friends)")

    # Interests
    user_interests = [
        ("alice", ["python", "ai"]),
        ("bob", ["rust", "ai"]),
        ("charlie", ["python", "music"]),
        ("diana", ["sports", "music"]),
        ("eve", ["python", "rust", "ai"]),
    ]
    for user, ints in user_interests:
        for interest in ints:
            graph.create_edge(
                EdgeBuilder(user, interest, "INTERESTED_IN")
                .build()
            )

    print_section("Graph Queries")

    # Find all NYC users
    print("1. People in NYC:")
    results = graph.query("MATCH (p:Person) WHERE p.city = 'NYC' RETURN p")
    for r in results:
        print(f"   - {r['p']['properties']['name']}")

    # Find Alice's friends
    print("\n2. Alice's friends:")
    alice_friends = graph.neighbors("alice", edge_type="FRIENDS_WITH")
    for friend in alice_friends:
        print(f"   - {friend.properties['name']}")

    # Find people interested in AI
    print("\n3. People interested in AI:")
    ai_fans = graph.neighbors("ai", direction="in", edge_type="INTERESTED_IN")
    for fan in ai_fans:
        print(f"   - {fan.properties['name']}")

    # Shortest path
    print("\n4. Shortest path from Alice to Diana:")
    path = graph.shortest_path("alice", "diana")
    if path:
        names = [node.properties.get('name', node.id) for node in path]
        print(f"   Path: {' -> '.join(names)}")

    # Traversal (returns paths as list of Nodes)
    print("\n5. Reachable from Alice (depth 2):")
    paths = graph.traverse("alice", max_depth=2)
    # Collect unique Person nodes from all paths
    seen = set()
    for path in paths:
        for node in path:
            if node.id not in seen and "Person" in node.labels:
                seen.add(node.id)
                print(f"   - {node.properties['name']}")

    return graph


# =============================================================================
# DEMO 3: Hybrid Search (Vector + Keyword)
# =============================================================================
def demo_hybrid_search():
    print_header("DEMO 3: Hybrid Search (Vector + BM25)")

    # Use BM25Index directly for keyword search
    from hybrid_search import BM25Index

    bm25 = BM25Index()
    vectors = {}  # id -> vector
    metadata = {}  # id -> metadata

    # Sample documents
    documents = [
        {"id": "doc1", "title": "Introduction to Machine Learning",
         "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
        {"id": "doc2", "title": "Deep Learning Fundamentals",
         "text": "Deep learning uses neural networks with multiple layers to process complex patterns in data."},
        {"id": "doc3", "title": "Natural Language Processing",
         "text": "NLP combines linguistics and machine learning to help computers understand human language."},
        {"id": "doc4", "title": "Computer Vision Applications",
         "text": "Computer vision enables machines to interpret and understand visual information from the world."},
        {"id": "doc5", "title": "Reinforcement Learning",
         "text": "Reinforcement learning trains agents to make decisions by rewarding desired behaviors."},
    ]

    print_section("Indexing documents")
    np.random.seed(123)

    # Create mock embeddings based on content
    for doc in documents:
        # Simple embedding: random but seeded by content hash
        np.random.seed(hash(doc["text"]) % 2**32)
        embedding = np.random.randn(64)
        embedding = embedding / np.linalg.norm(embedding)

        # Store vector and metadata
        vectors[doc["id"]] = embedding
        metadata[doc["id"]] = {"title": doc["title"], "text": doc["text"]}

        # Index text in BM25
        bm25.add_document(doc["id"], doc["text"])
        print(f"  Indexed: {doc['title']}")

    print_section("Search Comparisons")

    # Create a query embedding (similar to doc1 about ML)
    np.random.seed(hash(documents[0]["text"]) % 2**32)
    query_vec = np.random.randn(64)
    query_vec = query_vec / np.linalg.norm(query_vec)
    query_text = "machine learning artificial intelligence"

    print(f"Query: '{query_text}'\n")

    def vector_search(query_vec, k=3):
        """Simple cosine similarity search."""
        scores = []
        for doc_id, vec in vectors.items():
            # Cosine similarity
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            scores.append((doc_id, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def hybrid_search(query_vec, query_text, k=3, alpha=0.5):
        """Combine vector and keyword search."""
        # Get vector scores
        vec_results = vector_search(query_vec, k=len(vectors))
        vec_scores = {doc_id: score for doc_id, score in vec_results}

        # Get BM25 scores
        bm25_results = bm25.search(query_text, k=len(vectors))
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}

        # Normalize scores to [0, 1]
        if vec_scores:
            max_vec = max(vec_scores.values())
            min_vec = min(vec_scores.values())
            range_vec = max_vec - min_vec if max_vec != min_vec else 1
            vec_scores = {k: (v - min_vec) / range_vec for k, v in vec_scores.items()}

        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}

        # Combine scores
        combined = {}
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        for doc_id in all_ids:
            v_score = vec_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            combined[doc_id] = alpha * v_score + (1 - alpha) * b_score

        # Sort and return top k
        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(doc_id, score, metadata[doc_id]) for doc_id, score in results]

    # Vector-only search (alpha=1.0)
    print("1. Vector Search Only (alpha=1.0):")
    results = hybrid_search(query_vec, query_text, k=3, alpha=1.0)
    for i, (doc_id, score, meta) in enumerate(results, 1):
        print(f"   {i}. {meta['title']} (score: {score:.4f})")

    # Keyword-only search (alpha=0.0)
    print("\n2. Keyword Search Only (alpha=0.0):")
    results = hybrid_search(query_vec, query_text, k=3, alpha=0.0)
    for i, (doc_id, score, meta) in enumerate(results, 1):
        print(f"   {i}. {meta['title']} (score: {score:.4f})")

    # Hybrid search (alpha=0.5)
    print("\n3. Hybrid Search (alpha=0.5):")
    results = hybrid_search(query_vec, query_text, k=3, alpha=0.5)
    for i, (doc_id, score, meta) in enumerate(results, 1):
        print(f"   {i}. {meta['title']} (score: {score:.4f})")

    return bm25


# =============================================================================
# DEMO 4: Embeddings Integration
# =============================================================================
def demo_embeddings():
    print_header("DEMO 4: Automatic Embeddings")

    import tempfile

    # Use mock embedder for demo (no API keys needed)
    embedder = MockEmbedder(dimensions=128)

    # Create VectorDB and collection
    temp_dir = tempfile.mkdtemp()
    db = VectorDB(temp_dir)
    base_collection = db.create_collection("articles", dimensions=embedder.dimensions)

    # Wrap with EmbeddingCollection for auto-embedding
    collection = EmbeddingCollection(base_collection, embedder)

    print_section("Adding documents with auto-embedding")

    articles = [
        {"id": "a1", "text": "Python is a versatile programming language used for web development and data science."},
        {"id": "a2", "text": "Rust provides memory safety without garbage collection through its ownership system."},
        {"id": "a3", "text": "JavaScript is essential for building interactive web applications and frontend development."},
        {"id": "a4", "text": "Machine learning algorithms can identify patterns in large datasets automatically."},
        {"id": "a5", "text": "Cloud computing enables scalable infrastructure and serverless architectures."},
    ]

    for article in articles:
        collection.add(
            id=article["id"],
            text=article["text"],
            metadata={"text": article["text"][:50] + "..."}
        )
        print(f"  Embedded & indexed: {article['id']}")

    print_section("Semantic text search")

    queries = [
        "programming languages for beginners",
        "memory management in systems programming",
        "AI and data analysis",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = collection.search(query, k=2)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.metadata['text']}")

    return collection


# =============================================================================
# DEMO 5: Real-time Events
# =============================================================================
def demo_realtime():
    print_header("DEMO 5: Real-time Event System")

    import tempfile
    temp_dir = tempfile.mkdtemp()
    db = VectorDB(temp_dir)
    event_bus = EventBus()

    # Create an observable collection
    collection = ObservableCollection(
        db.create_collection("realtime_demo", dimensions=32),
        event_bus=event_bus
    )

    print_section("Setting up event listeners")

    # Register event handler (handles all event types)
    def on_event(event):
        event_type = event.type.value if hasattr(event.type, 'value') else event.type
        print(f"  [EVENT] {event_type.upper()}: id={event.data.get('id', 'N/A')}")

    event_bus.register(on_event)
    event_bus.start()  # Start processing events in background

    print("  Registered handler for all events")

    print_section("Performing operations (watch for events)")

    # Insert (vector, id, metadata)
    print("\n1. Inserting item 'item1'...")
    collection.insert(np.random.randn(32).tolist(), "item1", {"name": "First Item"})

    # Insert another
    print("\n2. Inserting item 'item2'...")
    collection.insert(np.random.randn(32).tolist(), "item2", {"name": "Second Item"})

    # Upsert (update)
    print("\n3. Updating item 'item1'...")
    collection.upsert(np.random.randn(32).tolist(), "item1", {"name": "Updated First Item"})

    # Delete
    print("\n4. Deleting item 'item2'...")
    collection.delete("item2")

    # Give background thread time to process events
    time.sleep(0.3)
    event_bus.stop()

    print("\n  All events were captured in real-time!")

    return event_bus


# =============================================================================
# DEMO 6: Persistence
# =============================================================================
def demo_persistence():
    print_header("DEMO 6: Persistence & Recovery")

    import os
    import tempfile
    import shutil

    # Create a temp directory for this demo
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "demo_db")

    print_section("Creating and saving database")

    # Create database at specific path
    db = VectorDB(db_path)
    collection = db.create_collection("persistent", dimensions=64)

    # Add some data (vector, id, metadata)
    for i in range(5):
        collection.insert(
            vector=np.random.randn(64).tolist(),
            id=f"record_{i}",
            metadata={"index": i, "name": f"Record {i}"}
        )
        print(f"  Added: record_{i}")

    # Save to disk
    db.save()
    print(f"\n  Database saved to: {db_path}")

    print_section("Loading from disk")

    # Load by creating new VectorDB with same path
    db2 = VectorDB(db_path)
    collection2 = db2.get_collection("persistent")

    print(f"  Loaded collection 'persistent' with {collection2.count()} records")

    # Verify data
    print("\n  Verifying data integrity:")
    for i in range(5):
        record = collection2.get(f"record_{i}")
        if record:
            print(f"    record_{i}: {record['metadata']['name']} ✓")

    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\n  Cleaned up temp files")

    return db2


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n" + "="*60)
    print("   RUVECTOR PYTHON DEMO")
    print("   Showcasing Vector DB, Graph, Embeddings & Real-time")
    print("="*60)

    demos = [
        ("Vector Search", demo_vector_search),
        ("Graph Database", demo_graph_database),
        ("Hybrid Search", demo_hybrid_search),
        ("Embeddings", demo_embeddings),
        ("Real-time Events", demo_realtime),
        ("Persistence", demo_persistence),
    ]

    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos)+1}. Run ALL demos")
    print("  0. Exit")

    while True:
        try:
            choice = input("\nSelect demo (0-7): ").strip()
            if choice == "0":
                print("\nGoodbye!")
                break
            elif choice == str(len(demos) + 1):
                for name, func in demos:
                    func()
                    input("\nPress Enter to continue to next demo...")
                print("\n✓ All demos completed!")
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                name, func = demos[int(choice) - 1]
                func()
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
