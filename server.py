"""
PyVectorDB REST API Server

Dependencies:
    pip install fastapi uvicorn pydantic numpy hnswlib

Run:
    uvicorn server:app --reload --port 8000

Or:
    python server.py
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Any
import numpy as np
import uvicorn
import time
from contextlib import asynccontextmanager

from vectordb import VectorDB, Collection, Filter, DistanceMetric, SearchResult


# =============================================================================
# Pydantic Models (API Schema)
# =============================================================================

class CollectionCreate(BaseModel):
    """Request to create a collection."""
    name: str = Field(..., description="Collection name")
    dimensions: int = Field(..., gt=0, description="Vector dimensions")
    metric: str = Field("cosine", description="Distance metric: cosine, l2, ip")
    M: int = Field(16, description="HNSW M parameter")
    ef_construction: int = Field(200, description="HNSW ef_construction")
    max_elements: int = Field(1_000_000, description="Max vectors")


class CollectionInfo(BaseModel):
    """Collection information."""
    name: str
    dimensions: int
    metric: str
    count: int


class VectorInsert(BaseModel):
    """Request to insert a vector."""
    id: Optional[str] = Field(None, description="Vector ID (auto-generated if not provided)")
    vector: list[float] = Field(..., description="Vector data")
    metadata: Optional[dict] = Field(None, description="Metadata")


class VectorBatchInsert(BaseModel):
    """Request to insert multiple vectors."""
    vectors: list[list[float]] = Field(..., description="List of vectors")
    ids: Optional[list[str]] = Field(None, description="List of IDs")
    metadata: Optional[list[dict]] = Field(None, description="List of metadata")


class VectorUpsert(BaseModel):
    """Request to upsert a vector."""
    id: str = Field(..., description="Vector ID")
    vector: list[float] = Field(..., description="Vector data")
    metadata: Optional[dict] = Field(None, description="Metadata")


class SearchRequest(BaseModel):
    """Search request."""
    vector: list[float] = Field(..., description="Query vector")
    k: int = Field(10, gt=0, le=1000, description="Number of results")
    filter: Optional[dict] = Field(None, description="Metadata filter")
    include_vectors: bool = Field(False, description="Include vectors in response")
    ef_search: Optional[int] = Field(None, description="Override ef_search")


class SearchBatchRequest(BaseModel):
    """Batch search request."""
    vectors: list[list[float]] = Field(..., description="Query vectors")
    k: int = Field(10, gt=0, le=1000, description="Number of results per query")
    filter: Optional[dict] = Field(None, description="Metadata filter")


class SearchResultItem(BaseModel):
    """Single search result."""
    id: str
    score: float
    metadata: dict = {}
    vector: Optional[list[float]] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: list[SearchResultItem]
    took_ms: float


class VectorResponse(BaseModel):
    """Vector retrieval response."""
    id: str
    metadata: dict = {}
    vector: Optional[list[float]] = None


class InsertResponse(BaseModel):
    """Insert response."""
    id: str
    success: bool = True


class BatchInsertResponse(BaseModel):
    """Batch insert response."""
    ids: list[str]
    count: int
    success: bool = True


class DeleteResponse(BaseModel):
    """Delete response."""
    deleted: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    collections: int
    uptime_seconds: float


# =============================================================================
# Application State
# =============================================================================

class AppState:
    db: VectorDB = None
    start_time: float = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    state.db = VectorDB("./vectordb_data")
    state.start_time = time.time()
    print(f"Loaded {len(state.db.list_collections())} collections")
    yield
    # Shutdown
    state.db.save()
    print("Database saved")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="PyVectorDB",
    description="High-performance vector database with HNSW indexing",
    version="0.2.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        collections=len(state.db.list_collections()),
        uptime_seconds=time.time() - state.start_time
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "name": "PyVectorDB",
        "version": "0.2.0",
        "docs": "/docs"
    }


# =============================================================================
# Collection Endpoints
# =============================================================================

@app.get("/collections", response_model=list[str], tags=["Collections"])
async def list_collections():
    """List all collections."""
    return state.db.list_collections()


@app.post("/collections", response_model=CollectionInfo, tags=["Collections"])
async def create_collection(request: CollectionCreate):
    """Create a new collection."""
    try:
        collection = state.db.create_collection(
            name=request.name,
            dimensions=request.dimensions,
            metric=request.metric,
            M=request.M,
            ef_construction=request.ef_construction,
            max_elements=request.max_elements
        )
        return CollectionInfo(
            name=collection.config.name,
            dimensions=collection.config.dimensions,
            metric=collection.config.metric.value,
            count=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{name}", response_model=CollectionInfo, tags=["Collections"])
async def get_collection_info(name: str):
    """Get collection information."""
    try:
        collection = state.db.get_collection(name)
        return CollectionInfo(
            name=collection.config.name,
            dimensions=collection.config.dimensions,
            metric=collection.config.metric.value,
            count=len(collection)
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")


@app.delete("/collections/{name}", response_model=DeleteResponse, tags=["Collections"])
async def delete_collection(name: str):
    """Delete a collection."""
    deleted = state.db.delete_collection(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return DeleteResponse(deleted=True)


# =============================================================================
# Vector Endpoints
# =============================================================================

@app.post("/collections/{name}/vectors", response_model=InsertResponse, tags=["Vectors"])
async def insert_vector(name: str, request: VectorInsert):
    """Insert a single vector."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        vector = np.array(request.vector, dtype=np.float32)
        id = collection.insert(vector, request.id, request.metadata)
        return InsertResponse(id=id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{name}/vectors/batch", response_model=BatchInsertResponse, tags=["Vectors"])
async def insert_vectors_batch(name: str, request: VectorBatchInsert):
    """Insert multiple vectors."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        vectors = np.array(request.vectors, dtype=np.float32)
        ids = collection.insert_batch(vectors, request.ids, request.metadata)
        return BatchInsertResponse(ids=ids, count=len(ids))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/collections/{name}/vectors", response_model=InsertResponse, tags=["Vectors"])
async def upsert_vector(name: str, request: VectorUpsert):
    """Upsert (insert or update) a vector."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        vector = np.array(request.vector, dtype=np.float32)
        id = collection.upsert(vector, request.id, request.metadata)
        return InsertResponse(id=id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{name}/vectors/{vector_id}", response_model=VectorResponse, tags=["Vectors"])
async def get_vector(
    name: str,
    vector_id: str,
    include_vector: bool = Query(False, description="Include vector data")
):
    """Get a vector by ID."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    result = collection.get(vector_id, include_vector=include_vector)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Vector '{vector_id}' not found")

    return VectorResponse(
        id=result["id"],
        metadata=result.get("metadata", {}),
        vector=result.get("vector", np.array([])).tolist() if include_vector else None
    )


@app.delete("/collections/{name}/vectors/{vector_id}", response_model=DeleteResponse, tags=["Vectors"])
async def delete_vector(name: str, vector_id: str):
    """Delete a vector by ID."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    deleted = collection.delete(vector_id)
    return DeleteResponse(deleted=deleted)


# =============================================================================
# Search Endpoints
# =============================================================================

@app.post("/collections/{name}/search", response_model=SearchResponse, tags=["Search"])
async def search_vectors(name: str, request: SearchRequest):
    """
    Search for similar vectors.

    Supports metadata filtering with operators:
    - Simple equality: {"category": "tech"}
    - Multiple conditions (AND): {"category": "tech", "year": 2023}
    """
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        query = np.array(request.vector, dtype=np.float32)
        start = time.time()

        results = collection.search(
            query,
            k=request.k,
            filter=request.filter,
            include_vectors=request.include_vectors,
            ef_search=request.ef_search
        )

        took_ms = (time.time() - start) * 1000

        return SearchResponse(
            results=[
                SearchResultItem(
                    id=r.id,
                    score=r.score,
                    metadata=r.metadata,
                    vector=r.vector.tolist() if r.vector is not None else None
                )
                for r in results
            ],
            took_ms=took_ms
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{name}/search/batch", tags=["Search"])
async def search_vectors_batch(name: str, request: SearchBatchRequest):
    """Search for multiple query vectors."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        queries = np.array(request.vectors, dtype=np.float32)
        start = time.time()

        all_results = collection.search_batch(queries, k=request.k, filter=request.filter)

        took_ms = (time.time() - start) * 1000

        return {
            "results": [
                [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in results]
                for results in all_results
            ],
            "took_ms": took_ms
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.post("/admin/save", tags=["Admin"])
async def save_database():
    """Force save all collections to disk."""
    state.db.save()
    return {"status": "saved"}


@app.get("/collections/{name}/ids", tags=["Admin"])
async def list_vector_ids(
    name: str,
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0)
):
    """List vector IDs in a collection."""
    try:
        collection = state.db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    ids = collection.list_ids(limit=limit, offset=offset)
    return {
        "ids": ids,
        "count": len(ids),
        "total": len(collection)
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
