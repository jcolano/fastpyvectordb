"""
PyVectorDB Python Client

Usage:
    from client import VectorDBClient

    client = VectorDBClient("http://localhost:8000")

    # Create collection
    client.create_collection("docs", dimensions=384)

    # Insert
    client.insert("docs", vector=[0.1, 0.2, ...], metadata={"title": "Hello"})

    # Search
    results = client.search("docs", vector=[0.1, 0.2, ...], k=10)
"""

import httpx
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict
    vector: Optional[list[float]] = None


class VectorDBClient:
    """HTTP client for PyVectorDB server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health(self) -> dict:
        """Check server health."""
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    # -------------------------------------------------------------------------
    # Collections
    # -------------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """List all collections."""
        r = self.client.get(f"{self.base_url}/collections")
        r.raise_for_status()
        return r.json()

    def create_collection(
        self,
        name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs
    ) -> dict:
        """Create a new collection."""
        data = {
            "name": name,
            "dimensions": dimensions,
            "metric": metric,
            **kwargs
        }
        r = self.client.post(f"{self.base_url}/collections", json=data)
        r.raise_for_status()
        return r.json()

    def get_collection(self, name: str) -> dict:
        """Get collection info."""
        r = self.client.get(f"{self.base_url}/collections/{name}")
        r.raise_for_status()
        return r.json()

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        r = self.client.delete(f"{self.base_url}/collections/{name}")
        r.raise_for_status()
        return r.json()["deleted"]

    # -------------------------------------------------------------------------
    # Vectors
    # -------------------------------------------------------------------------

    def insert(
        self,
        collection: str,
        vector: list[float],
        id: str = None,
        metadata: dict = None
    ) -> str:
        """Insert a vector."""
        data = {"vector": vector}
        if id:
            data["id"] = id
        if metadata:
            data["metadata"] = metadata

        r = self.client.post(
            f"{self.base_url}/collections/{collection}/vectors",
            json=data
        )
        r.raise_for_status()
        return r.json()["id"]

    def insert_batch(
        self,
        collection: str,
        vectors: list[list[float]],
        ids: list[str] = None,
        metadata: list[dict] = None
    ) -> list[str]:
        """Insert multiple vectors."""
        data = {"vectors": vectors}
        if ids:
            data["ids"] = ids
        if metadata:
            data["metadata"] = metadata

        r = self.client.post(
            f"{self.base_url}/collections/{collection}/vectors/batch",
            json=data
        )
        r.raise_for_status()
        return r.json()["ids"]

    def upsert(
        self,
        collection: str,
        id: str,
        vector: list[float],
        metadata: dict = None
    ) -> str:
        """Upsert a vector."""
        data = {"id": id, "vector": vector}
        if metadata:
            data["metadata"] = metadata

        r = self.client.put(
            f"{self.base_url}/collections/{collection}/vectors",
            json=data
        )
        r.raise_for_status()
        return r.json()["id"]

    def get(
        self,
        collection: str,
        id: str,
        include_vector: bool = False
    ) -> Optional[dict]:
        """Get a vector by ID."""
        params = {"include_vector": include_vector}
        r = self.client.get(
            f"{self.base_url}/collections/{collection}/vectors/{id}",
            params=params
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def delete(self, collection: str, id: str) -> bool:
        """Delete a vector."""
        r = self.client.delete(
            f"{self.base_url}/collections/{collection}/vectors/{id}"
        )
        r.raise_for_status()
        return r.json()["deleted"]

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        collection: str,
        vector: list[float],
        k: int = 10,
        filter: dict = None,
        include_vectors: bool = False,
        ef_search: int = None
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        data = {
            "vector": vector,
            "k": k,
            "include_vectors": include_vectors
        }
        if filter:
            data["filter"] = filter
        if ef_search:
            data["ef_search"] = ef_search

        r = self.client.post(
            f"{self.base_url}/collections/{collection}/search",
            json=data
        )
        r.raise_for_status()
        response = r.json()

        return [
            SearchResult(
                id=item["id"],
                score=item["score"],
                metadata=item.get("metadata", {}),
                vector=item.get("vector")
            )
            for item in response["results"]
        ]

    def search_batch(
        self,
        collection: str,
        vectors: list[list[float]],
        k: int = 10,
        filter: dict = None
    ) -> list[list[SearchResult]]:
        """Search for multiple queries."""
        data = {"vectors": vectors, "k": k}
        if filter:
            data["filter"] = filter

        r = self.client.post(
            f"{self.base_url}/collections/{collection}/search/batch",
            json=data
        )
        r.raise_for_status()
        response = r.json()

        return [
            [
                SearchResult(
                    id=item["id"],
                    score=item["score"],
                    metadata=item.get("metadata", {})
                )
                for item in results
            ]
            for results in response["results"]
        ]

    # -------------------------------------------------------------------------
    # Admin
    # -------------------------------------------------------------------------

    def save(self):
        """Force save to disk."""
        r = self.client.post(f"{self.base_url}/admin/save")
        r.raise_for_status()

    def list_ids(
        self,
        collection: str,
        limit: int = 100,
        offset: int = 0
    ) -> dict:
        """List vector IDs."""
        r = self.client.get(
            f"{self.base_url}/collections/{collection}/ids",
            params={"limit": limit, "offset": offset}
        )
        r.raise_for_status()
        return r.json()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("PyVectorDB Client Demo")
    print("=" * 50)
    print("Make sure server is running: python server.py")
    print("=" * 50)

    try:
        with VectorDBClient("http://localhost:8000") as client:
            # Health check
            health = client.health()
            print(f"\nServer status: {health['status']}")

            # Create collection
            collection_name = "demo_collection"
            dimensions = 128

            if collection_name not in client.list_collections():
                print(f"\nCreating collection '{collection_name}'...")
                client.create_collection(collection_name, dimensions)
            else:
                print(f"\nUsing existing collection '{collection_name}'")

            # Insert vectors
            print("\nInserting vectors...")
            np.random.seed(42)

            vectors = np.random.randn(100, dimensions).astype(np.float32)
            vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).tolist()

            ids = [f"doc_{i}" for i in range(100)]
            metadata = [
                {"category": np.random.choice(["A", "B", "C"]), "value": float(i)}
                for i in range(100)
            ]

            inserted_ids = client.insert_batch(collection_name, vectors, ids, metadata)
            print(f"Inserted {len(inserted_ids)} vectors")

            # Search
            query = np.random.randn(dimensions).astype(np.float32)
            query = (query / np.linalg.norm(query)).tolist()

            print("\nSearching (no filter)...")
            results = client.search(collection_name, query, k=5)
            for r in results:
                print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}")

            print("\nSearching (filter: category='A')...")
            results = client.search(collection_name, query, k=5, filter={"category": "A"})
            for r in results:
                print(f"  {r.id}: score={r.score:.4f}, category={r.metadata.get('category')}")

            # Get info
            info = client.get_collection(collection_name)
            print(f"\nCollection info: {info}")

            print("\nDemo complete!")

    except httpx.ConnectError:
        print("\nError: Could not connect to server.")
        print("Start the server first: python server.py")
