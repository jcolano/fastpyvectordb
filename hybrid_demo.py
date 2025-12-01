#!/usr/bin/env python3
"""
Hybrid Search Demo - Vector + BM25 Keyword Search

Compares pure vector search, pure keyword (BM25) search, 
and hybrid combinations to show when each approach excels.
"""

import sys
sys.path.insert(0, '/mnt/project')

import numpy as np
import tempfile
import shutil
from pathlib import Path

from vectordb import VectorDB, CollectionConfig
from hybrid_search import HybridCollection, BM25Index


# Simple BoW embedder (same as retrieval demo)
import re
import math
from collections import Counter

class BagOfWordsEmbedder:
    def __init__(self, dimensions: int = 256):
        self._dimensions = dimensions
        self._vocab = {}
        self._idf = {}
        
    @property
    def dimensions(self):
        return self._dimensions
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'but', 'or', 'so', 'if', 'then', 'this', 'that', 'it'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def build_vocab(self, documents: list[str]):
        word_doc_count = Counter()
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] += 1
        
        top_words = word_doc_count.most_common(self._dimensions)
        self._vocab = {word: idx for idx, (word, _) in enumerate(top_words)}
        
        n_docs = len(documents)
        for word, count in word_doc_count.items():
            if word in self._vocab:
                self._idf[word] = math.log(n_docs / (1 + count))
    
    def embed(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        word_counts = Counter(tokens)
        
        vector = np.zeros(self._dimensions, dtype=np.float32)
        for word, count in word_counts.items():
            if word in self._vocab:
                idx = self._vocab[word]
                tf = 1 + math.log(count) if count > 0 else 0
                idf = self._idf.get(word, 1.0)
                vector[idx] = tf * idf
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


# Document corpus - designed to show hybrid search benefits
DOCUMENTS = [
    # Tech documents with specific terminology
    {
        "id": "hnsw_1",
        "text": "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layer graph structure for efficient vector similarity queries.",
        "category": "algorithm"
    },
    {
        "id": "hnsw_2", 
        "text": "Vector databases use HNSW indexing to find similar embeddings quickly. The algorithm provides logarithmic search complexity with high recall rates.",
        "category": "database"
    },
    {
        "id": "faiss_1",
        "text": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search. It supports GPU acceleration and various index types including IVF and PQ.",
        "category": "library"
    },
    {
        "id": "embed_1",
        "text": "Word embeddings represent text as dense vectors where semantic similarity corresponds to vector proximity. Models like Word2Vec and BERT create these representations.",
        "category": "ml"
    },
    {
        "id": "embed_2",
        "text": "Sentence transformers generate embeddings for entire sentences, capturing contextual meaning. They are widely used for semantic search applications.",
        "category": "ml"
    },
    
    # Documents with exact keyword matches
    {
        "id": "python_1",
        "text": "Python list comprehensions provide a concise way to create lists. The syntax [x for x in iterable] filters and transforms elements efficiently.",
        "category": "python"
    },
    {
        "id": "python_2",
        "text": "NumPy arrays enable fast numerical computations in Python. Broadcasting and vectorized operations eliminate slow Python loops.",
        "category": "python"
    },
    
    # Documents with semantic similarity but different words
    {
        "id": "cooking_1",
        "text": "Making fresh pasta requires flour, eggs, and patience. Knead the dough until smooth, then roll thin sheets before cutting into shapes.",
        "category": "cooking"
    },
    {
        "id": "cooking_2",
        "text": "Homemade noodles taste better than store-bought. The key is proper gluten development through kneading and adequate resting time.",
        "category": "cooking"
    },
    
    # Edge case: specific acronym
    {
        "id": "rag_1",
        "text": "RAG (Retrieval Augmented Generation) combines vector search with language models. It retrieves relevant documents to ground LLM responses in factual content.",
        "category": "ai"
    },
]


def run_hybrid_demo():
    print("=" * 70)
    print("  HYBRID SEARCH COMPARISON DEMO")
    print("  Vector Search vs Keyword Search vs Hybrid")
    print("=" * 70)
    
    # Setup embedder
    embedder = BagOfWordsEmbedder(dimensions=256)
    all_text = [d['text'] for d in DOCUMENTS]
    embedder.build_vocab(all_text)
    
    # Create hybrid collection
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = CollectionConfig(
            name="hybrid_test",
            dimensions=embedder.dimensions,
            max_elements=1000
        )
        
        collection = HybridCollection(
            config=config,
            base_path=Path(temp_dir),
            text_fields=["text"]
        )
        
        # Insert documents
        print("\n📄 Inserting documents...")
        for doc in DOCUMENTS:
            vector = embedder.embed(doc['text'])
            collection.insert(
                vector=vector,
                id=doc['id'],
                metadata={"text": doc['text'], "category": doc['category']}
            )
        print(f"   Inserted {len(DOCUMENTS)} documents")
        
        # Test queries designed to show different search strengths
        test_cases = [
            {
                "query": "HNSW algorithm for nearest neighbor",
                "query_text": "HNSW algorithm for nearest neighbor",
                "note": "Exact term 'HNSW' - keyword search should excel"
            },
            {
                "query": "how to make fresh pasta at home",
                "query_text": "how to make fresh pasta at home",
                "note": "Semantic match - 'noodles' doc is related but uses different words"
            },
            {
                "query": "RAG retrieval augmented generation",
                "query_text": "RAG retrieval augmented generation",
                "note": "Acronym search - exact keyword match critical"
            },
            {
                "query": "convert text to vectors for similarity",
                "query_text": "convert text to vectors for similarity",
                "note": "Semantic concept - embeddings docs should match"
            },
            {
                "query": "Python fast numerical array operations",
                "query_text": "Python fast numerical array operations",
                "note": "Mix of exact (Python) and semantic (numerical/array)"
            },
        ]
        
        print("\n" + "=" * 70)
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'─' * 70}")
            print(f"Test {i}: \"{test['query']}\"")
            print(f"Note: {test['note']}")
            
            query_vector = embedder.embed(test['query'])
            
            # Pure vector search (alpha=1.0)
            print("\n  📊 VECTOR ONLY (alpha=1.0):")
            results = collection.hybrid_search(
                vector=query_vector,
                query_text=test['query_text'],
                k=3,
                alpha=1.0
            )
            for r in results:
                print(f"     {r.id}: v={r.vector_score:.3f} k={r.keyword_score:.3f} -> {r.score:.3f}")
            
            # Pure keyword search (alpha=0.0)
            print("\n  📝 KEYWORD ONLY (alpha=0.0):")
            results = collection.hybrid_search(
                vector=query_vector,
                query_text=test['query_text'],
                k=3,
                alpha=0.0
            )
            for r in results:
                print(f"     {r.id}: v={r.vector_score:.3f} k={r.keyword_score:.3f} -> {r.score:.3f}")
            
            # Hybrid (alpha=0.5)
            print("\n  🔀 HYBRID (alpha=0.5):")
            results = collection.hybrid_search(
                vector=query_vector,
                query_text=test['query_text'],
                k=3,
                alpha=0.5
            )
            for r in results:
                print(f"     {r.id}: v={r.vector_score:.3f} k={r.keyword_score:.3f} -> {r.score:.3f}")
            
            # Hybrid favoring keywords (alpha=0.3)
            print("\n  🔀 HYBRID (alpha=0.3 - favor keywords):")
            results = collection.hybrid_search(
                vector=query_vector,
                query_text=test['query_text'],
                k=3,
                alpha=0.3
            )
            for r in results:
                print(f"     {r.id}: v={r.vector_score:.3f} k={r.keyword_score:.3f} -> {r.score:.3f}")
        
        # Summary
        print("\n" + "=" * 70)
        print("  KEY INSIGHTS")
        print("=" * 70)
        print("""
  • VECTOR SEARCH excels at:
    - Finding semantically similar content (pasta ↔ noodles)
    - Matching concepts even with different vocabulary
    - Handling paraphrased queries
    
  • KEYWORD (BM25) SEARCH excels at:
    - Exact term matching (HNSW, RAG, specific names)
    - Acronym and technical jargon lookups
    - When users know the exact terminology
    
  • HYBRID SEARCH provides:
    - Best of both worlds when tuned properly
    - alpha=0.5: Balanced for general use
    - alpha=0.3: Better for technical/jargon queries
    - alpha=0.7: Better for conversational queries
        """)
        
        # Interactive
        print("=" * 70)
        print("  TRY YOUR OWN QUERIES")
        print("=" * 70)
        print("\nFormat: Enter query, or 'alpha=X query' to set alpha")
        print("Example: 'alpha=0.3 FAISS GPU acceleration'")
        print("Enter 'quit' to exit\n")
        
        current_alpha = 0.5
        
        while True:
            try:
                user_input = input(f"🔍 [alpha={current_alpha}] Query: ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q', ''):
                    break
                
                # Check for alpha override
                if user_input.startswith('alpha='):
                    parts = user_input.split(' ', 1)
                    try:
                        current_alpha = float(parts[0].split('=')[1])
                        current_alpha = max(0, min(1, current_alpha))
                    except:
                        pass
                    user_input = parts[1] if len(parts) > 1 else ''
                    if not user_input:
                        print(f"   Alpha set to {current_alpha}")
                        continue
                
                query_vector = embedder.embed(user_input)
                
                results = collection.hybrid_search(
                    vector=query_vector,
                    query_text=user_input,
                    k=5,
                    alpha=current_alpha
                )
                
                print(f"\n   Results (alpha={current_alpha}):")
                for rank, r in enumerate(results, 1):
                    doc = next(d for d in DOCUMENTS if d['id'] == r.id)
                    print(f"   {rank}. [{r.score:.3f}] {r.id}")
                    print(f"      v={r.vector_score:.3f} k={r.keyword_score:.3f}")
                    print(f"      {doc['text'][:70]}...")
                print()
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\n👋 Demo complete!")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    run_hybrid_demo()
