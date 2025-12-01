"""
PyVectorDB Full-Featured REST API Server

Includes:
- Vector database CRUD and search
- Graph database with Cypher queries
- Embeddings integration (auto-embed text)
- WebSocket real-time updates

Dependencies:
    pip install fastapi uvicorn pydantic numpy hnswlib websockets

Optional:
    pip install openai sentence-transformers  # For embeddings

Run:
    uvicorn server_full:app --reload --port 8000

WebSocket:
    Connect to ws://localhost:8000/ws or ws://localhost:8000/ws/{collection}
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Any
import numpy as np
import uvicorn
import time
import json
import asyncio
from contextlib import asynccontextmanager

from vectordb import VectorDB, Collection, Filter
from graph import GraphDB, NodeBuilder, EdgeBuilder, HyperedgeBuilder, Node, Edge
from embeddings import get_embedder, Embedder, EmbeddingCollection
from realtime import AsyncConnectionManager, Event, EventType, Subscription


# =============================================================================
# Pydantic Models
# =============================================================================

# --- Collections ---
class CollectionCreate(BaseModel):
    name: str
    dimensions: int
    metric: str = "cosine"


class CollectionInfo(BaseModel):
    name: str
    dimensions: int
    metric: str
    count: int


# --- Vectors ---
class VectorInsert(BaseModel):
    id: Optional[str] = None
    vector: list[float]
    metadata: Optional[dict] = None


class TextInsert(BaseModel):
    """Insert with automatic embedding."""
    id: Optional[str] = None
    text: str
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    vector: Optional[list[float]] = None
    text: Optional[str] = None  # For semantic search
    k: int = 10
    filter: Optional[dict] = None
    include_vectors: bool = False


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: dict = {}
    vector: Optional[list[float]] = None


# --- Graph ---
class NodeCreate(BaseModel):
    id: Optional[str] = None
    labels: list[str] = []
    properties: dict = {}


class EdgeCreate(BaseModel):
    id: Optional[str] = None
    from_node: str
    to_node: str
    type: str
    properties: dict = {}


class HyperedgeCreate(BaseModel):
    id: Optional[str] = None
    nodes: list[str]
    type: str
    properties: dict = {}


class CypherQuery(BaseModel):
    query: str


class TraversalRequest(BaseModel):
    start_node: str
    edge_type: Optional[str] = None
    max_depth: int = 3
    direction: str = "out"


class ShortestPathRequest(BaseModel):
    from_node: str
    to_node: str
    edge_type: Optional[str] = None
    max_depth: int = 10


# --- WebSocket ---
class SubscriptionUpdate(BaseModel):
    collection: Optional[str] = "*"
    filter: Optional[dict] = None
    event_types: Optional[list[str]] = None


# =============================================================================
# Application State
# =============================================================================

class AppState:
    vector_db: VectorDB = None
    graph_db: GraphDB = None
    embedder: Embedder = None
    ws_manager: AsyncConnectionManager = None
    start_time: float = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    # Startup
    state.vector_db = VectorDB("./vectordb_data")
    state.graph_db = GraphDB("./graphdb_data")
    state.ws_manager = AsyncConnectionManager()
    state.start_time = time.time()

    # Try to initialize embedder
    try:
        state.embedder = get_embedder("auto")
        print(f"Embedder initialized: {state.embedder.model_name} ({state.embedder.dimensions}d)")
    except Exception as e:
        print(f"Embedder not available: {e}")
        state.embedder = None

    print(f"Loaded {len(state.vector_db.list_collections())} vector collections")
    print(f"Graph: {state.graph_db.node_count()} nodes, {state.graph_db.edge_count()} edges")

    yield

    # Shutdown
    state.vector_db.save()
    state.graph_db.save()
    print("Databases saved")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="PyVectorDB",
    description="Vector + Graph database with real-time updates",
    version="0.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "version": "0.3.0",
        "uptime_seconds": time.time() - state.start_time,
        "vector_collections": len(state.vector_db.list_collections()),
        "graph_nodes": state.graph_db.node_count(),
        "graph_edges": state.graph_db.edge_count(),
        "websocket_connections": state.ws_manager.connection_count,
        "embedder": state.embedder.model_name if state.embedder else None
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "PyVectorDB",
        "version": "0.3.0",
        "docs": "/docs",
        "features": ["vectors", "graph", "embeddings", "realtime"]
    }


# =============================================================================
# Vector Collection Endpoints
# =============================================================================

@app.get("/collections", response_model=list[str], tags=["Collections"])
async def list_collections():
    return state.vector_db.list_collections()


@app.post("/collections", response_model=CollectionInfo, tags=["Collections"])
async def create_collection(request: CollectionCreate):
    try:
        collection = state.vector_db.create_collection(
            name=request.name,
            dimensions=request.dimensions,
            metric=request.metric
        )

        await state.ws_manager.broadcast(Event(
            type=EventType.COLLECTION_CREATED,
            collection=request.name,
            data={"dimensions": request.dimensions, "metric": request.metric}
        ))

        return CollectionInfo(
            name=request.name,
            dimensions=request.dimensions,
            metric=request.metric,
            count=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{name}", response_model=CollectionInfo, tags=["Collections"])
async def get_collection_info(name: str):
    try:
        collection = state.vector_db.get_collection(name)
        return CollectionInfo(
            name=collection.config.name,
            dimensions=collection.config.dimensions,
            metric=collection.config.metric.value,
            count=len(collection)
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")


@app.delete("/collections/{name}", tags=["Collections"])
async def delete_collection(name: str):
    if not state.vector_db.delete_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    await state.ws_manager.broadcast(Event(
        type=EventType.COLLECTION_DELETED,
        collection=name,
        data={}
    ))

    return {"deleted": True}


# =============================================================================
# Vector Endpoints
# =============================================================================

@app.post("/collections/{name}/vectors", tags=["Vectors"])
async def insert_vector(name: str, request: VectorInsert):
    try:
        collection = state.vector_db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        vector = np.array(request.vector, dtype=np.float32)
        id = collection.insert(vector, request.id, request.metadata)

        await state.ws_manager.broadcast(Event(
            type=EventType.INSERT,
            collection=name,
            data={"id": id, "metadata": request.metadata or {}}
        ))

        return {"id": id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{name}/texts", tags=["Vectors"])
async def insert_text(name: str, request: TextInsert):
    """Insert text with automatic embedding."""
    if not state.embedder:
        raise HTTPException(status_code=503, detail="Embedder not available")

    try:
        collection = state.vector_db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    if collection.config.dimensions != state.embedder.dimensions:
        raise HTTPException(
            status_code=400,
            detail=f"Collection dimensions ({collection.config.dimensions}) "
                   f"don't match embedder ({state.embedder.dimensions})"
        )

    try:
        vector = state.embedder.embed(request.text)
        metadata = request.metadata or {}
        metadata["_text"] = request.text

        id = collection.insert(vector, request.id, metadata)

        await state.ws_manager.broadcast(Event(
            type=EventType.INSERT,
            collection=name,
            data={"id": id, "metadata": metadata}
        ))

        return {"id": id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/{name}/search", tags=["Vectors"])
async def search_vectors(name: str, request: SearchRequest):
    try:
        collection = state.vector_db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        # Get query vector
        if request.vector:
            query = np.array(request.vector, dtype=np.float32)
        elif request.text and state.embedder:
            query = state.embedder.embed(request.text)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'vector' or 'text' (with embedder) required"
            )

        start = time.time()
        results = collection.search(
            query,
            k=request.k,
            filter=request.filter,
            include_vectors=request.include_vectors
        )
        took_ms = (time.time() - start) * 1000

        return {
            "results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "metadata": r.metadata,
                    "vector": r.vector.tolist() if r.vector is not None else None
                }
                for r in results
            ],
            "took_ms": took_ms
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{name}/vectors/{vector_id}", tags=["Vectors"])
async def get_vector(name: str, vector_id: str, include_vector: bool = False):
    try:
        collection = state.vector_db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    result = collection.get(vector_id, include_vector=include_vector)
    if not result:
        raise HTTPException(status_code=404, detail=f"Vector '{vector_id}' not found")

    return result


@app.delete("/collections/{name}/vectors/{vector_id}", tags=["Vectors"])
async def delete_vector(name: str, vector_id: str):
    try:
        collection = state.vector_db.get_collection(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    deleted = collection.delete(vector_id)

    if deleted:
        await state.ws_manager.broadcast(Event(
            type=EventType.DELETE,
            collection=name,
            data={"id": vector_id}
        ))

    return {"deleted": deleted}


# =============================================================================
# Graph Endpoints
# =============================================================================

@app.get("/graph/stats", tags=["Graph"])
async def graph_stats():
    return state.graph_db.stats()


# --- Nodes ---
@app.post("/graph/nodes", tags=["Graph"])
async def create_node(request: NodeCreate):
    try:
        node = NodeBuilder(request.id)
        for label in request.labels:
            node.label(label)
        node.properties(request.properties)

        node_id = state.graph_db.create_node(node.build())

        await state.ws_manager.broadcast(Event(
            type=EventType.INSERT,
            collection="_graph_nodes",
            data={"id": node_id, "labels": request.labels, "properties": request.properties}
        ))

        return {"id": node_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/graph/nodes/{node_id}", tags=["Graph"])
async def get_node(node_id: str):
    node = state.graph_db.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return node.to_dict()


@app.get("/graph/nodes", tags=["Graph"])
async def find_nodes(label: str = None, limit: int = 100):
    if label:
        nodes = state.graph_db.get_nodes_by_label(label)
    else:
        nodes = list(state.graph_db._nodes.values())
    return [n.to_dict() for n in nodes[:limit]]


@app.delete("/graph/nodes/{node_id}", tags=["Graph"])
async def delete_node(node_id: str, cascade: bool = True):
    try:
        deleted = state.graph_db.delete_node(node_id, cascade=cascade)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

        await state.ws_manager.broadcast(Event(
            type=EventType.DELETE,
            collection="_graph_nodes",
            data={"id": node_id}
        ))

        return {"deleted": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Edges ---
@app.post("/graph/edges", tags=["Graph"])
async def create_edge(request: EdgeCreate):
    try:
        edge = EdgeBuilder(request.from_node, request.to_node, request.type)
        if request.id:
            edge.id(request.id)
        edge.properties(request.properties)

        edge_id = state.graph_db.create_edge(edge.build())

        await state.ws_manager.broadcast(Event(
            type=EventType.INSERT,
            collection="_graph_edges",
            data={
                "id": edge_id,
                "from": request.from_node,
                "to": request.to_node,
                "type": request.type
            }
        ))

        return {"id": edge_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/graph/edges/{edge_id}", tags=["Graph"])
async def get_edge(edge_id: str):
    edge = state.graph_db.get_edge(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail=f"Edge '{edge_id}' not found")
    return edge.to_dict()


@app.get("/graph/edges", tags=["Graph"])
async def find_edges(type: str = None, limit: int = 100):
    if type:
        edges = state.graph_db.get_edges_by_type(type)
    else:
        edges = list(state.graph_db._edges.values())
    return [e.to_dict() for e in edges[:limit]]


@app.delete("/graph/edges/{edge_id}", tags=["Graph"])
async def delete_edge(edge_id: str):
    deleted = state.graph_db.delete_edge(edge_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Edge '{edge_id}' not found")

    await state.ws_manager.broadcast(Event(
        type=EventType.DELETE,
        collection="_graph_edges",
        data={"id": edge_id}
    ))

    return {"deleted": True}


# --- Hyperedges ---
@app.post("/graph/hyperedges", tags=["Graph"])
async def create_hyperedge(request: HyperedgeCreate):
    try:
        he = HyperedgeBuilder(request.nodes, request.type)
        if request.id:
            he.id(request.id)
        he.property("properties", request.properties)

        he_id = state.graph_db.create_hyperedge(he.build())
        return {"id": he_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Queries ---
@app.post("/graph/query", tags=["Graph"])
async def execute_query(request: CypherQuery):
    try:
        results = state.graph_db.query(request.query)
        return {"columns": results.columns, "rows": results.rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/graph/traverse", tags=["Graph"])
async def traverse(request: TraversalRequest):
    paths = state.graph_db.traverse(
        request.start_node,
        request.edge_type,
        request.max_depth,
        request.direction
    )
    return {
        "paths": [
            [n.to_dict() for n in path]
            for path in paths
        ]
    }


@app.post("/graph/shortest-path", tags=["Graph"])
async def shortest_path(request: ShortestPathRequest):
    path = state.graph_db.shortest_path(
        request.from_node,
        request.to_node,
        request.edge_type,
        request.max_depth
    )
    if path is None:
        return {"path": None}
    return {"path": [n.to_dict() for n in path]}


@app.get("/graph/neighbors/{node_id}", tags=["Graph"])
async def get_neighbors(
    node_id: str,
    direction: str = "out",
    edge_type: str = None
):
    node = state.graph_db.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    neighbors = state.graph_db.neighbors(node_id, direction, edge_type)
    return [n.to_dict() for n in neighbors]


# =============================================================================
# Embeddings Endpoints
# =============================================================================

@app.get("/embeddings/info", tags=["Embeddings"])
async def embeddings_info():
    if not state.embedder:
        return {"available": False}
    return {
        "available": True,
        "model": state.embedder.model_name,
        "dimensions": state.embedder.dimensions
    }


@app.post("/embeddings/embed", tags=["Embeddings"])
async def embed_text(text: str):
    if not state.embedder:
        raise HTTPException(status_code=503, detail="Embedder not available")

    try:
        vector = state.embedder.embed(text)
        return {"vector": vector.tolist(), "dimensions": len(vector)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/embed-batch", tags=["Embeddings"])
async def embed_texts(texts: list[str]):
    if not state.embedder:
        raise HTTPException(status_code=503, detail="Embedder not available")

    try:
        vectors = state.embedder.embed_batch(texts)
        return {
            "vectors": vectors.tolist(),
            "count": len(vectors),
            "dimensions": vectors.shape[1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws")
async def websocket_all(websocket: WebSocket):
    """Subscribe to all events."""
    await state.ws_manager.connect(websocket, "*")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "subscribe":
                    sub = Subscription(
                        collection=msg.get("collection", "*"),
                        filter=msg.get("filter"),
                        event_types=[EventType(t) for t in msg.get("event_types", [])]
                        if msg.get("event_types") else None
                    )
                    await state.ws_manager.update_subscription(websocket, sub)
                    await websocket.send_text(json.dumps({"status": "subscribed", "subscription": msg}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await state.ws_manager.disconnect(websocket, "*")


@app.websocket("/ws/{collection}")
async def websocket_collection(websocket: WebSocket, collection: str):
    """Subscribe to a specific collection."""
    await state.ws_manager.connect(websocket, collection)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "filter":
                    sub = Subscription(
                        collection=collection,
                        filter=msg.get("filter"),
                        event_types=[EventType(t) for t in msg.get("event_types", [])]
                        if msg.get("event_types") else None
                    )
                    await state.ws_manager.update_subscription(websocket, sub)
                    await websocket.send_text(json.dumps({"status": "filtered", "filter": msg.get("filter")}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await state.ws_manager.disconnect(websocket, collection)


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.post("/admin/save", tags=["Admin"])
async def save_all():
    state.vector_db.save()
    state.graph_db.save()
    return {"status": "saved"}


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("server_full:app", host="0.0.0.0", port=8000, reload=True)
