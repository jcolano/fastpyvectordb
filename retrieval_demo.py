#!/usr/bin/env python3
"""
Retrieval Quality Demo for VectorDB

Tests the database's ability to find semantically similar documents.
Uses real semantic embeddings (SentenceTransformers) when available,
falls back to a bag-of-words approach that still captures word overlap.

Run: python retrieval_demo.py
"""

import sys
import os
import numpy as np
from collections import Counter
import re
import math

# Add project to path
sys.path.insert(0, '/mnt/project')

from vectordb import VectorDB, Filter
from hybrid_search import HybridCollection, CollectionConfig


# =============================================================================
# Embedder Options
# =============================================================================

class BagOfWordsEmbedder:
    """
    Simple BoW embedder that captures word overlap.
    Not as good as transformer models, but demonstrates retrieval concepts.
    Uses TF-IDF-like weighting with a fixed vocabulary.
    """
    
    def __init__(self, dimensions: int = 512):
        self._dimensions = dimensions
        self._vocab = {}  # word -> index
        self._idf = {}    # word -> idf score
        self._documents = []
        
    @property
    def dimensions(self):
        return self._dimensions
    
    @property    
    def model_name(self):
        return "bag-of-words"
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alpha."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        # Remove very common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                     'very', 'just', 'also', 'now', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'each', 'every', 'both',
                     'few', 'more', 'most', 'other', 'some', 'such', 'no',
                     'any', 'this', 'that', 'these', 'those', 'it', 'its'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def build_vocab(self, documents: list[str]):
        """Build vocabulary from document corpus."""
        self._documents = documents
        word_doc_count = Counter()
        
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] += 1
        
        # Keep top N words by document frequency
        top_words = word_doc_count.most_common(self._dimensions)
        self._vocab = {word: idx for idx, (word, _) in enumerate(top_words)}
        
        # Calculate IDF
        n_docs = len(documents)
        for word, count in word_doc_count.items():
            if word in self._vocab:
                self._idf[word] = math.log(n_docs / (1 + count))
    
    def embed(self, text: str) -> np.ndarray:
        """Create embedding vector for text."""
        tokens = self._tokenize(text)
        word_counts = Counter(tokens)
        
        vector = np.zeros(self._dimensions, dtype=np.float32)
        
        for word, count in word_counts.items():
            if word in self._vocab:
                idx = self._vocab[word]
                # TF-IDF: term frequency * inverse document frequency
                tf = 1 + math.log(count) if count > 0 else 0
                idf = self._idf.get(word, 1.0)
                vector[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


def get_embedder():
    """Try to get the best available embedder."""
    # Try SentenceTransformers first (best quality)
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Using SentenceTransformers (all-MiniLM-L6-v2) - real semantic embeddings!")
        
        class STEmbedder:
            def __init__(self):
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self._dimensions = 384
            
            @property
            def dimensions(self):
                return self._dimensions
            
            @property
            def model_name(self):
                return "all-MiniLM-L6-v2"
            
            def embed(self, text: str) -> np.ndarray:
                return self._model.encode(text, convert_to_numpy=True).astype(np.float32)
            
            def embed_batch(self, texts: list[str]) -> np.ndarray:
                return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
            
            def build_vocab(self, documents):
                pass  # Not needed for transformer models
        
        return STEmbedder()
    
    except ImportError:
        print("⚠ SentenceTransformers not available, using Bag-of-Words embedder")
        print("  (Install with: pip install sentence-transformers)")
        return BagOfWordsEmbedder(dimensions=512)


# =============================================================================
# Document Corpus
# =============================================================================

DOCUMENTS = [
    # Cluster 1: Machine Learning / AI
    {
        "id": "ml_1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions.",
        "category": "AI"
    },
    {
        "id": "ml_2", 
        "title": "Deep Learning Neural Networks",
        "content": "Deep learning uses artificial neural networks with multiple layers to model complex patterns. These networks are inspired by the human brain and excel at image recognition and natural language processing.",
        "category": "AI"
    },
    {
        "id": "ml_3",
        "title": "Supervised vs Unsupervised Learning",
        "content": "Supervised learning trains models on labeled data to predict outcomes. Unsupervised learning finds hidden patterns in unlabeled data through clustering and dimensionality reduction techniques.",
        "category": "AI"
    },
    
    # Cluster 2: Databases / Data Storage
    {
        "id": "db_1",
        "title": "Introduction to Vector Databases",
        "content": "Vector databases store high-dimensional embeddings and enable fast similarity search. They use algorithms like HNSW to find nearest neighbors efficiently, powering semantic search and recommendation systems.",
        "category": "Database"
    },
    {
        "id": "db_2",
        "title": "SQL Database Fundamentals",
        "content": "Relational databases organize data into tables with rows and columns. SQL queries retrieve and manipulate data using structured query language with joins, filters, and aggregations.",
        "category": "Database"
    },
    {
        "id": "db_3",
        "title": "NoSQL Document Stores",
        "content": "Document databases like MongoDB store data as JSON documents without rigid schemas. They offer flexibility and horizontal scaling for applications with evolving data structures.",
        "category": "Database"
    },
    
    # Cluster 3: Web Development
    {
        "id": "web_1",
        "title": "Building REST APIs",
        "content": "REST APIs use HTTP methods to expose web services. Endpoints return JSON data, supporting CRUD operations. Authentication tokens and rate limiting protect API resources.",
        "category": "Web"
    },
    {
        "id": "web_2",
        "title": "Frontend JavaScript Frameworks",
        "content": "React, Vue, and Angular are popular JavaScript frameworks for building interactive user interfaces. They use component-based architecture and virtual DOM for efficient rendering.",
        "category": "Web"
    },
    {
        "id": "web_3",
        "title": "HTML and CSS Fundamentals",
        "content": "HTML structures web content with semantic elements. CSS styles the presentation with selectors, properties, and responsive layouts using flexbox and grid systems.",
        "category": "Web"
    },
    
    # Cluster 4: Python Programming
    {
        "id": "py_1",
        "title": "Python for Data Science",
        "content": "Python is the leading language for data science with libraries like NumPy, Pandas, and Matplotlib. It enables data manipulation, analysis, and visualization with clean syntax.",
        "category": "Python"
    },
    {
        "id": "py_2",
        "title": "Python Object-Oriented Programming",
        "content": "Python supports object-oriented programming with classes, inheritance, and encapsulation. Decorators and context managers provide elegant patterns for code organization.",
        "category": "Python"
    },
    
    # Cluster 5: Cooking (completely different domain)
    {
        "id": "cook_1",
        "title": "Italian Pasta Recipes",
        "content": "Traditional Italian pasta dishes like carbonara and bolognese use fresh ingredients. Al dente pasta, quality olive oil, and aged parmesan cheese are essential for authentic flavors.",
        "category": "Cooking"
    },
    {
        "id": "cook_2",
        "title": "Baking Bread at Home",
        "content": "Homemade bread requires flour, water, yeast, and salt. Kneading develops gluten structure while proofing allows the dough to rise. A hot oven creates crispy crust.",
        "category": "Cooking"
    },
    
    # Edge cases: Cross-domain documents
    {
        "id": "cross_1",
        "title": "AI in Recipe Generation",
        "content": "Machine learning models can generate creative recipes by learning from culinary databases. Neural networks understand flavor pairings and ingredient substitutions.",
        "category": "AI+Cooking"
    },
    {
        "id": "cross_2",
        "title": "Database-Backed Web Applications",
        "content": "Modern web apps use databases for persistent storage. SQL and NoSQL databases connect to backend APIs, serving data to frontend JavaScript applications.",
        "category": "Web+Database"
    },
]


# =============================================================================
# Test Queries
# =============================================================================

TEST_QUERIES = [
    {
        "query": "How do neural networks learn from data?",
        "expected_top": ["ml_1", "ml_2", "ml_3"],
        "description": "Should find ML/AI documents"
    },
    {
        "query": "What is the best way to store embeddings for similarity search?",
        "expected_top": ["db_1"],
        "description": "Should find vector database document"
    },
    {
        "query": "How to build a website with JavaScript",
        "expected_top": ["web_1", "web_2", "web_3"],
        "description": "Should find web development documents"
    },
    {
        "query": "Python data analysis and visualization",
        "expected_top": ["py_1"],
        "description": "Should find Python data science document"
    },
    {
        "query": "How to make homemade Italian food",
        "expected_top": ["cook_1", "cook_2"],
        "description": "Should find cooking documents"
    },
    {
        "query": "Using AI to create new recipes",
        "expected_top": ["cross_1", "ml_1", "ml_2"],
        "description": "Should find AI+Cooking crossover and AI docs"
    },
    {
        "query": "Connecting databases to web applications",
        "expected_top": ["cross_2", "db_1", "db_2", "web_1"],
        "description": "Should find cross-domain and related docs"
    },
    {
        "query": "Tables, rows, columns, and SQL queries",
        "expected_top": ["db_2"],
        "description": "Should find SQL database document"
    },
]


# =============================================================================
# Main Demo
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_retrieval_demo():
    print("=" * 70)
    print("  VECTOR DATABASE RETRIEVAL QUALITY DEMO")
    print("=" * 70)
    
    # Get embedder
    print("\n📦 Initializing embedder...")
    embedder = get_embedder()
    print(f"   Model: {embedder.model_name}")
    print(f"   Dimensions: {embedder.dimensions}")
    
    # Build vocabulary for BoW embedder
    if hasattr(embedder, 'build_vocab') and embedder.model_name == "bag-of-words":
        all_text = [f"{d['title']} {d['content']}" for d in DOCUMENTS]
        embedder.build_vocab(all_text)
        print(f"   Vocabulary size: {len(embedder._vocab)}")
    
    # Create database and collection
    print("\n📁 Creating vector database...")
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        db = VectorDB(temp_dir)
        collection = db.create_collection(
            name="documents",
            dimensions=embedder.dimensions,
            metric="cosine"
        )
        
        # Insert documents
        print(f"\n📄 Inserting {len(DOCUMENTS)} documents...")
        for doc in DOCUMENTS:
            text = f"{doc['title']}. {doc['content']}"
            vector = embedder.embed(text)
            collection.insert(
                vector=vector,
                id=doc['id'],
                metadata={
                    "title": doc['title'],
                    "content": doc['content'][:100] + "...",
                    "category": doc['category']
                }
            )
        print(f"   Collection size: {collection.count()}")
        
        # Show document similarity matrix (sample)
        print("\n" + "=" * 70)
        print("  DOCUMENT SIMILARITY ANALYSIS")
        print("=" * 70)
        
        print("\n📊 Sample similarities between document pairs:")
        pairs_to_check = [
            ("ml_1", "ml_2", "Both ML docs - should be HIGH"),
            ("ml_1", "cook_1", "ML vs Cooking - should be LOW"),
            ("db_1", "db_2", "Both DB docs - should be MEDIUM-HIGH"),
            ("cross_1", "ml_1", "AI+Cooking vs ML - should be MEDIUM"),
            ("cross_1", "cook_1", "AI+Cooking vs Cooking - should be MEDIUM"),
            ("web_1", "web_2", "Both Web docs - should be HIGH"),
        ]
        
        doc_vectors = {}
        for doc in DOCUMENTS:
            text = f"{doc['title']}. {doc['content']}"
            doc_vectors[doc['id']] = embedder.embed(text)
        
        print(f"\n   {'Doc A':<10} {'Doc B':<10} {'Similarity':>10}   Expected")
        print("   " + "-" * 55)
        for id_a, id_b, expected in pairs_to_check:
            sim = cosine_similarity(doc_vectors[id_a], doc_vectors[id_b])
            print(f"   {id_a:<10} {id_b:<10} {sim:>10.4f}   {expected}")
        
        # Run retrieval tests
        print("\n" + "=" * 70)
        print("  RETRIEVAL QUALITY TESTS")
        print("=" * 70)
        
        total_tests = len(TEST_QUERIES)
        passed_tests = 0
        
        for i, test in enumerate(TEST_QUERIES, 1):
            print(f"\n{'─' * 70}")
            print(f"Query {i}/{total_tests}: \"{test['query']}\"")
            print(f"Expected: {test['description']}")
            
            # Embed query
            query_vector = embedder.embed(test['query'])
            
            # Search
            results = collection.search(query_vector, k=5)
            
            print(f"\nTop 5 Results:")
            print(f"   {'Rank':<5} {'ID':<12} {'Score':>8} {'Category':<15} Title")
            print("   " + "-" * 65)
            
            retrieved_ids = []
            for rank, r in enumerate(results, 1):
                retrieved_ids.append(r.id)
                # Find full title
                doc = next(d for d in DOCUMENTS if d['id'] == r.id)
                score_display = f"{1 - r.score:.4f}" if r.score < 1 else f"{r.score:.4f}"
                
                # Check if this was expected
                is_expected = r.id in test['expected_top']
                marker = "✓" if is_expected else " "
                
                print(f" {marker} {rank:<5} {r.id:<12} {score_display:>8} {r.metadata['category']:<15} {doc['title'][:30]}")
            
            # Check if any expected doc is in top 3
            top_3_ids = retrieved_ids[:3]
            hits = [eid for eid in test['expected_top'] if eid in top_3_ids]
            
            if hits:
                print(f"\n   ✅ PASS - Found expected document(s) in top 3: {hits}")
                passed_tests += 1
            else:
                print(f"\n   ❌ FAIL - Expected {test['expected_top'][:3]} in top 3")
        
        # Summary
        print("\n" + "=" * 70)
        print("  RETRIEVAL QUALITY SUMMARY")
        print("=" * 70)
        
        accuracy = (passed_tests / total_tests) * 100
        print(f"\n   Tests Passed: {passed_tests}/{total_tests} ({accuracy:.1f}%)")
        print(f"   Embedder: {embedder.model_name}")
        
        if accuracy >= 80:
            print("\n   🎉 Excellent retrieval quality!")
        elif accuracy >= 60:
            print("\n   👍 Good retrieval quality")
        elif accuracy >= 40:
            print("\n   ⚠️  Moderate retrieval quality - consider better embeddings")
        else:
            print("\n   ❌ Poor retrieval quality - upgrade to SentenceTransformers")
        
        # Test filtered search
        print("\n" + "=" * 70)
        print("  FILTERED RETRIEVAL TEST")
        print("=" * 70)
        
        query = "How does learning work?"
        query_vector = embedder.embed(query)
        
        print(f"\nQuery: \"{query}\"")
        print("\nWithout filter (top 3):")
        results = collection.search(query_vector, k=3)
        for r in results:
            doc = next(d for d in DOCUMENTS if d['id'] == r.id)
            print(f"   {r.id}: {doc['title']} [{r.metadata['category']}]")
        
        print("\nWith filter: category='AI' (top 3):")
        results = collection.search(query_vector, k=3, filter={"category": "AI"})
        for r in results:
            doc = next(d for d in DOCUMENTS if d['id'] == r.id)
            print(f"   {r.id}: {doc['title']} [{r.metadata['category']}]")
        
        print("\nWith filter: category='Cooking' (top 3):")
        results = collection.search(query_vector, k=3, filter={"category": "Cooking"})
        for r in results:
            doc = next(d for d in DOCUMENTS if d['id'] == r.id)
            print(f"   {r.id}: {doc['title']} [{r.metadata['category']}]")
        
        # Interactive mode
        print("\n" + "=" * 70)
        print("  INTERACTIVE SEARCH")
        print("=" * 70)
        print("\nEnter your own queries (or 'quit' to exit):\n")
        
        while True:
            try:
                user_query = input("🔍 Query: ").strip()
                if user_query.lower() in ('quit', 'exit', 'q', ''):
                    break
                
                query_vector = embedder.embed(user_query)
                results = collection.search(query_vector, k=5)
                
                print(f"\nResults for: \"{user_query}\"")
                for rank, r in enumerate(results, 1):
                    doc = next(d for d in DOCUMENTS if d['id'] == r.id)
                    sim = 1 - r.score if r.score < 1 else r.score
                    print(f"   {rank}. [{sim:.3f}] {doc['title']}")
                    print(f"      {doc['content'][:80]}...")
                print()
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\n👋 Demo complete!")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    run_retrieval_demo()
