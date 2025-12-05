"""
Tests for the FastPyVectorDB Client API.
"""

import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fastpyvectordb
from fastpyvectordb import Client, Collection


class TestClient:
    """Test the Client class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def client(self, temp_dir):
        """Create a test client with mock embedder."""
        return Client(
            path=temp_dir,
            embedding_provider="mock"
        )

    def test_create_collection(self, client):
        """Test creating a collection."""
        collection = client.create_collection("test_docs")
        assert collection.name == "test_docs"
        assert collection.count == 0

    def test_get_collection(self, client):
        """Test getting an existing collection."""
        client.create_collection("test_docs")
        collection = client.get_collection("test_docs")
        assert collection.name == "test_docs"

    def test_get_or_create_collection(self, client):
        """Test get_or_create_collection."""
        # First call creates
        coll1 = client.get_or_create_collection("docs")
        assert coll1.name == "docs"

        # Second call gets
        coll2 = client.get_or_create_collection("docs")
        assert coll2.name == "docs"

    def test_list_collections(self, client):
        """Test listing collections."""
        client.create_collection("docs1")
        client.create_collection("docs2")
        collections = client.list_collections()
        assert "docs1" in collections
        assert "docs2" in collections

    def test_delete_collection(self, client):
        """Test deleting a collection."""
        client.create_collection("to_delete")
        assert "to_delete" in client.list_collections()

        client.delete_collection("to_delete")
        assert "to_delete" not in client.list_collections()


class TestCollection:
    """Test the Collection class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def collection(self, temp_dir):
        """Create a test collection with mock embedder."""
        client = Client(path=temp_dir, embedding_provider="mock")
        return client.create_collection("test")

    def test_add_documents(self, collection):
        """Test adding documents."""
        collection.add(
            documents=["Hello world", "Goodbye world"],
            ids=["doc1", "doc2"]
        )
        assert collection.count == 2

    def test_add_with_metadata(self, collection):
        """Test adding documents with metadata."""
        collection.add(
            documents=["Test document"],
            ids=["doc1"],
            metadatas=[{"category": "test", "score": 0.9}]
        )

        result = collection.get(ids=["doc1"])
        assert result.metadatas[0]["category"] == "test"
        assert result.metadatas[0]["score"] == 0.9

    def test_add_with_embeddings(self, collection):
        """Test adding pre-computed embeddings."""
        embeddings = np.random.randn(2, 384).tolist()
        collection.add(
            embeddings=embeddings,
            ids=["emb1", "emb2"]
        )
        assert collection.count == 2

    def test_query(self, collection):
        """Test querying documents."""
        collection.add(
            documents=["Machine learning", "Deep learning", "Natural language"],
            ids=["ml", "dl", "nlp"]
        )

        results = collection.query(
            query_texts="artificial intelligence",
            n_results=2
        )

        assert len(results.ids[0]) <= 2
        assert len(results.documents[0]) <= 2
        assert len(results.distances[0]) <= 2

    def test_query_with_filter(self, collection):
        """Test querying with filter."""
        collection.add(
            documents=["Doc A", "Doc B", "Doc C"],
            ids=["a", "b", "c"],
            metadatas=[
                {"category": "tech"},
                {"category": "science"},
                {"category": "tech"}
            ]
        )

        results = collection.query(
            query_texts="document",
            n_results=10,
            where={"category": "tech"}
        )

        # Should only return tech documents
        for meta in results.metadatas[0]:
            assert meta.get("category") == "tech"

    def test_get_by_id(self, collection):
        """Test getting documents by ID."""
        collection.add(
            documents=["Hello", "World"],
            ids=["doc1", "doc2"]
        )

        result = collection.get(ids=["doc1"])
        assert len(result.ids) == 1
        assert result.ids[0] == "doc1"
        assert result.documents[0] == "Hello"

    def test_get_with_filter(self, collection):
        """Test getting documents with filter."""
        collection.add(
            documents=["A", "B", "C"],
            ids=["a", "b", "c"],
            metadatas=[{"val": 1}, {"val": 2}, {"val": 3}]
        )

        result = collection.get(where={"val": 2})
        assert len(result.ids) == 1
        assert result.ids[0] == "b"

    def test_update(self, collection):
        """Test updating documents."""
        collection.add(
            documents=["Original"],
            ids=["doc1"],
            metadatas=[{"version": 1}]
        )

        collection.update(
            ids=["doc1"],
            documents=["Updated"],
            metadatas=[{"version": 2}]
        )

        result = collection.get(ids=["doc1"])
        assert result.documents[0] == "Updated"
        assert result.metadatas[0]["version"] == 2

    def test_upsert(self, collection):
        """Test upserting documents."""
        # Add initial document
        collection.add(
            documents=["Original"],
            ids=["doc1"]
        )

        # Upsert same ID and new ID
        collection.upsert(
            documents=["Updated", "New"],
            ids=["doc1", "doc2"]
        )

        assert collection.count == 2
        result = collection.get(ids=["doc1"])
        assert result.documents[0] == "Updated"

    def test_delete_by_id(self, collection):
        """Test deleting by ID."""
        collection.add(
            documents=["A", "B", "C"],
            ids=["a", "b", "c"]
        )

        deleted = collection.delete(ids=["b"])
        assert deleted == 1
        assert collection.count == 2

    def test_delete_by_filter(self, collection):
        """Test deleting by filter."""
        collection.add(
            documents=["A", "B", "C"],
            ids=["a", "b", "c"],
            metadatas=[
                {"keep": True},
                {"keep": False},
                {"keep": False}
            ]
        )

        deleted = collection.delete(where={"keep": False})
        assert deleted == 2
        assert collection.count == 1

    def test_peek(self, collection):
        """Test peeking at documents."""
        collection.add(
            documents=[f"Doc {i}" for i in range(10)],
            ids=[f"doc{i}" for i in range(10)]
        )

        sample = collection.peek(limit=3)
        assert len(sample.ids) == 3


class TestPersistence:
    """Test data persistence."""

    def test_persist_and_reload(self):
        """Test that data persists across client instances."""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create and populate
            client1 = Client(path=temp_dir, embedding_provider="mock")
            coll1 = client1.create_collection("persist_test")
            coll1.add(
                documents=["Persistent data"],
                ids=["doc1"],
                metadatas=[{"test": True}]
            )
            client1.persist()

            # Reload
            client2 = Client(path=temp_dir, embedding_provider="mock")
            coll2 = client2.get_collection("persist_test")

            assert coll2.count == 1
            result = coll2.get(ids=["doc1"])
            assert result.documents[0] == "Persistent data"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
