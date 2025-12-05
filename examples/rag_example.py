#!/usr/bin/env python3
"""
Complete RAG (Retrieval-Augmented Generation) Example

This script demonstrates how to build a simple RAG system using VectorDB.
It can work with or without an LLM - without an LLM it just returns the
retrieved documents.

Usage:
    python rag_example.py                    # Interactive mode
    python rag_example.py --index            # Index sample documents
    python rag_example.py --query "question" # Single query mode
    
Requirements:
    pip install numpy hnswlib sentence-transformers  # For embeddings
    pip install anthropic  # Optional: for LLM responses
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np

# Add project to path
sys.path.insert(0, '/mnt/project')

from vectordb import VectorDB, Filter


# =============================================================================
# Configuration
# =============================================================================

class RAGConfig:
    """Configuration for the RAG system."""
    db_path: str = "./rag_demo_db"
    collection_name: str = "knowledge_base"
    embedding_model: str = "all-MiniLM-L6-v2"  # 384 dimensions
    dimensions: int = 384
    top_k: int = 3
    use_llm: bool = False  # Set True if you have ANTHROPIC_API_KEY


# =============================================================================
# Embedder (tries SentenceTransformers, falls back to simple BoW)
# =============================================================================

class Embedder:
    """Wrapper for embedding generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimensions = None
        self._use_st = False
        
        # Try SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            self._use_st = True
            print(f"✓ Using SentenceTransformers: {model_name} ({self._dimensions}d)")
        except ImportError:
            print("⚠ SentenceTransformers not available, using simple embedder")
            print("  Install with: pip install sentence-transformers")
            self._dimensions = 384
            self._init_simple_embedder()
    
    def _init_simple_embedder(self):
        """Initialize a simple bag-of-words embedder."""
        import hashlib
        import re
        from collections import Counter
        
        self._vocab = {}
        self._tokenize = lambda text: [t for t in re.findall(r'\b[a-z]+\b', text.lower()) 
                                       if len(t) > 2]
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self._use_st:
            return self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            # Simple hash-based embedding for demo
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            rng = np.random.RandomState(seed)
            vector = rng.randn(self._dimensions).astype(np.float32)
            return vector / np.linalg.norm(vector)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if self._use_st:
            return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        else:
            return np.array([self.embed(t) for t in texts])


# =============================================================================
# RAG System
# =============================================================================

class RAGSystem:
    """
    Retrieval-Augmented Generation System.
    
    This class handles:
    1. Document indexing (embedding + storage)
    2. Retrieval (semantic search)
    3. Response generation (optional LLM integration)
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embedder = Embedder(self.config.embedding_model)
        
        # Update dimensions based on embedder
        self.config.dimensions = self.embedder.dimensions
        
        # Initialize database
        self.db = VectorDB(self.config.db_path)
        
        # Get or create collection
        if self.config.collection_name in self.db.list_collections():
            self.collection = self.db.get_collection(self.config.collection_name)
            print(f"✓ Loaded existing collection: {self.config.collection_name} ({self.collection.count()} docs)")
        else:
            self.collection = self.db.create_collection(
                name=self.config.collection_name,
                dimensions=self.config.dimensions,
                metric="cosine"
            )
            print(f"✓ Created new collection: {self.config.collection_name}")
        
        # Optional: LLM client
        self.llm_client = None
        if self.config.use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client."""
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.llm_client = anthropic.Anthropic()
                print("✓ LLM client initialized (Claude)")
            else:
                print("⚠ ANTHROPIC_API_KEY not set, LLM disabled")
        except ImportError:
            print("⚠ anthropic package not installed, LLM disabled")
    
    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------
    
    def index_document(self, doc_id: str, content: str, metadata: dict = None) -> str:
        """
        Index a single document.
        
        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Additional metadata (title, author, etc.)
        
        Returns:
            The document ID
        """
        # Generate embedding
        vector = self.embedder.embed(content)
        
        # Prepare metadata
        meta = metadata or {}
        meta["content"] = content  # Store original content for retrieval
        
        # Insert or update
        self.collection.upsert(vector=vector, id=doc_id, metadata=meta)
        return doc_id
    
    def index_documents(self, documents: list[dict]) -> list[str]:
        """
        Index multiple documents.
        
        Args:
            documents: List of dicts with keys: id, content, and optional metadata
        
        Returns:
            List of document IDs
        """
        ids = []
        texts = []
        metadata_list = []
        
        for doc in documents:
            ids.append(doc["id"])
            texts.append(doc["content"])
            meta = doc.get("metadata", {})
            meta["content"] = doc["content"]
            if "title" in doc:
                meta["title"] = doc["title"]
            metadata_list.append(meta)
        
        # Batch embed
        vectors = self.embedder.embed_batch(texts)
        
        # Batch insert
        self.collection.insert_batch(vectors=vectors, ids=ids, metadata_list=metadata_list)
        
        return ids
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        return self.collection.delete(doc_id)
    
    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    
    def retrieve(self, query: str, k: int = None, filter: dict = None) -> list[dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question or search query
            k: Number of documents to retrieve
            filter: Optional metadata filter
        
        Returns:
            List of relevant documents with scores
        """
        k = k or self.config.top_k
        
        # Embed query
        query_vector = self.embedder.embed(query)
        
        # Search
        results = self.collection.search(query_vector, k=k, filter=filter)
        
        # Format results
        retrieved = []
        for r in results:
            retrieved.append({
                "id": r.id,
                "score": float(r.score),
                "content": r.metadata.get("content", ""),
                "title": r.metadata.get("title", r.id),
                "metadata": {k: v for k, v in r.metadata.items() if k not in ["content"]}
            })
        
        return retrieved
    
    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    
    def generate_response(self, query: str, context_docs: list[dict]) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User question
            context_docs: Retrieved documents
        
        Returns:
            Generated response
        """
        if not self.llm_client:
            # Without LLM, just format the retrieved documents
            response = f"Found {len(context_docs)} relevant documents:\n\n"
            for i, doc in enumerate(context_docs, 1):
                response += f"[{i}] {doc['title']} (relevance: {1-doc['score']:.2%})\n"
                response += f"    {doc['content'][:200]}...\n\n"
            return response
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[Document {i}: {doc['title']}]\n{doc['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""
        
        # Call LLM
        response = self.llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    # -------------------------------------------------------------------------
    # Main Query Interface
    # -------------------------------------------------------------------------
    
    def query(self, question: str, k: int = None, filter: dict = None, 
              show_sources: bool = True) -> dict:
        """
        Main RAG query interface.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            filter: Optional metadata filter
            show_sources: Include source documents in response
        
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        # Retrieve
        docs = self.retrieve(question, k=k, filter=filter)
        
        # Generate
        answer = self.generate_response(question, docs)
        
        result = {"answer": answer}
        if show_sources:
            result["sources"] = docs
        
        return result
    
    def save(self):
        """Save the database to disk."""
        self.db.save()


# =============================================================================
# Sample Documents
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "python_basics",
        "title": "Python Programming Basics",
        "content": """Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum and first released in 1991. Python supports 
multiple programming paradigms including procedural, object-oriented, and functional programming. 
Its simple syntax makes it excellent for beginners while its powerful libraries make it suitable 
for complex applications.""",
        "metadata": {"category": "programming", "language": "python"}
    },
    {
        "id": "python_data",
        "title": "Python for Data Science",
        "content": """Python has become the leading language for data science due to powerful libraries 
like NumPy for numerical computing, Pandas for data manipulation, and Matplotlib for visualization. 
Scikit-learn provides machine learning algorithms, while TensorFlow and PyTorch enable deep learning. 
Jupyter notebooks provide an interactive environment for data exploration and analysis.""",
        "metadata": {"category": "data-science", "language": "python"}
    },
    {
        "id": "ml_intro",
        "title": "Introduction to Machine Learning",
        "content": """Machine learning is a subset of artificial intelligence that enables computers to 
learn from data without being explicitly programmed. There are three main types: supervised learning 
(learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and 
reinforcement learning (learning through trial and error with rewards). Common algorithms include 
linear regression, decision trees, neural networks, and support vector machines.""",
        "metadata": {"category": "AI", "topic": "machine-learning"}
    },
    {
        "id": "neural_networks",
        "title": "Neural Networks and Deep Learning",
        "content": """Neural networks are computing systems inspired by biological neural networks in 
the human brain. Deep learning uses neural networks with multiple layers (deep neural networks) to 
learn hierarchical representations of data. Convolutional Neural Networks (CNNs) excel at image 
processing, while Recurrent Neural Networks (RNNs) and Transformers are used for sequential data 
like text and time series.""",
        "metadata": {"category": "AI", "topic": "deep-learning"}
    },
    {
        "id": "vector_db",
        "title": "Vector Databases Explained",
        "content": """Vector databases are specialized databases designed to store and search 
high-dimensional vectors (embeddings). They use algorithms like HNSW (Hierarchical Navigable Small 
World) for fast approximate nearest neighbor search. Vector databases power semantic search, 
recommendation systems, and RAG (Retrieval-Augmented Generation) applications. Popular options 
include Pinecone, Milvus, Qdrant, and Chroma.""",
        "metadata": {"category": "database", "topic": "vector-search"}
    },
    {
        "id": "rag_systems",
        "title": "RAG Systems Architecture",
        "content": """Retrieval-Augmented Generation (RAG) combines retrieval systems with large 
language models. The process involves: 1) Embedding documents and storing in a vector database, 
2) When a query arrives, embed it and search for similar documents, 3) Include retrieved documents 
as context in the LLM prompt, 4) Generate a grounded response. RAG reduces hallucinations and 
allows LLMs to access up-to-date information without retraining.""",
        "metadata": {"category": "AI", "topic": "RAG"}
    },
    {
        "id": "web_apis",
        "title": "Building REST APIs",
        "content": """REST (Representational State Transfer) APIs use HTTP methods to expose web 
services. GET retrieves resources, POST creates new resources, PUT updates existing resources, 
and DELETE removes resources. APIs typically return JSON data and use status codes to indicate 
success (200), creation (201), bad request (400), not found (404), and server error (500). 
Authentication is commonly handled via API keys, OAuth, or JWT tokens.""",
        "metadata": {"category": "web-development", "topic": "APIs"}
    },
    {
        "id": "sql_basics",
        "title": "SQL Database Fundamentals",
        "content": """SQL (Structured Query Language) is used to manage relational databases. 
Tables store data in rows and columns. SELECT queries retrieve data, INSERT adds new rows, 
UPDATE modifies existing rows, and DELETE removes rows. JOIN operations combine data from 
multiple tables. Indexes speed up queries but slow down writes. ACID properties (Atomicity, 
Consistency, Isolation, Durability) ensure reliable transactions.""",
        "metadata": {"category": "database", "topic": "SQL"}
    },
]


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument("--index", action="store_true", help="Index sample documents")
    parser.add_argument("--query", "-q", type=str, help="Query to run")
    parser.add_argument("--k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--filter", type=str, help="JSON filter for metadata")
    parser.add_argument("--llm", action="store_true", help="Use LLM for generation")
    parser.add_argument("--db-path", type=str, default="./rag_demo_db", help="Database path")
    parser.add_argument("--clear", action="store_true", help="Clear database before indexing")
    
    args = parser.parse_args()
    
    # Configure
    config = RAGConfig()
    config.db_path = args.db_path
    config.top_k = args.k
    config.use_llm = args.llm
    
    print("=" * 60)
    print("  RAG SYSTEM DEMO")
    print("=" * 60)
    
    # Clear if requested
    if args.clear:
        import shutil
        if Path(args.db_path).exists():
            shutil.rmtree(args.db_path)
            print(f"✓ Cleared database at {args.db_path}")
    
    # Initialize RAG system
    rag = RAGSystem(config)
    
    # Index sample documents
    if args.index or rag.collection.count() == 0:
        print("\n📄 Indexing sample documents...")
        ids = rag.index_documents(SAMPLE_DOCUMENTS)
        print(f"✓ Indexed {len(ids)} documents")
        rag.save()
    
    # Parse filter
    filter_dict = None
    if args.filter:
        try:
            filter_dict = json.loads(args.filter)
        except json.JSONDecodeError:
            print(f"⚠ Invalid filter JSON: {args.filter}")
    
    # Single query mode
    if args.query:
        print(f"\n🔍 Query: {args.query}")
        if filter_dict:
            print(f"   Filter: {filter_dict}")
        
        result = rag.query(args.query, k=args.k, filter=filter_dict)
        
        print("\n" + "─" * 60)
        print("ANSWER:")
        print(result["answer"])
        
        if "sources" in result:
            print("\n" + "─" * 60)
            print("SOURCES:")
            for src in result["sources"]:
                relevance = (1 - src["score"]) * 100
                print(f"  • {src['title']} ({relevance:.1f}% relevant)")
        
        return
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("=" * 60)
    print("\nCommands:")
    print("  /index    - Re-index sample documents")
    print("  /count    - Show document count")
    print("  /list     - List document IDs")
    print("  /filter X - Set filter (JSON), /filter clear to remove")
    print("  /k N      - Set number of results")
    print("  /quit     - Exit")
    print("\nOr just type your question!\n")
    
    current_filter = None
    current_k = args.k
    
    while True:
        try:
            user_input = input("🔍 > ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                arg = user_input[len(cmd):].strip()
                
                if cmd == "/quit" or cmd == "/exit":
                    print("👋 Goodbye!")
                    break
                    
                elif cmd == "/index":
                    print("📄 Re-indexing documents...")
                    ids = rag.index_documents(SAMPLE_DOCUMENTS)
                    print(f"✓ Indexed {len(ids)} documents")
                    rag.save()
                    
                elif cmd == "/count":
                    print(f"📊 Collection has {rag.collection.count()} documents")
                    
                elif cmd == "/list":
                    ids = rag.collection.list_ids(limit=20)
                    print(f"📋 Documents: {', '.join(ids)}")
                    
                elif cmd == "/filter":
                    if arg.lower() == "clear":
                        current_filter = None
                        print("✓ Filter cleared")
                    else:
                        try:
                            current_filter = json.loads(arg)
                            print(f"✓ Filter set: {current_filter}")
                        except json.JSONDecodeError:
                            print(f"⚠ Invalid JSON: {arg}")
                            
                elif cmd == "/k":
                    try:
                        current_k = int(arg)
                        print(f"✓ k set to {current_k}")
                    except ValueError:
                        print("⚠ Invalid number")
                        
                else:
                    print(f"⚠ Unknown command: {cmd}")
                    
                continue
            
            # Query
            result = rag.query(user_input, k=current_k, filter=current_filter)
            
            print("\n" + result["answer"])
            
            if "sources" in result and result["sources"]:
                print("\n📚 Sources:")
                for src in result["sources"][:3]:
                    relevance = (1 - src["score"]) * 100
                    print(f"   • {src['title']} ({relevance:.1f}%)")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"⚠ Error: {e}")
    
    rag.save()


if __name__ == "__main__":
    main()
