"""
Real-time Updates Module

Provides WebSocket-based real-time subscriptions for vector database changes.

Features:
- Subscribe to collection changes (insert, update, delete)
- Filter subscriptions by metadata
- Broadcast to multiple clients
- Event history for late joiners

Usage:
    # Server side (with FastAPI)
    from realtime import RealtimeManager, Event

    manager = RealtimeManager()

    @app.websocket("/ws/{collection}")
    async def websocket_endpoint(websocket: WebSocket, collection: str):
        await manager.connect(websocket, collection)
        try:
            while True:
                data = await websocket.receive_text()
                # Handle client messages (e.g., filter updates)
        except WebSocketDisconnect:
            manager.disconnect(websocket, collection)

    # Emit events when data changes
    manager.emit(Event(
        type="insert",
        collection="documents",
        data={"id": "doc1", "metadata": {...}}
    ))

    # Client side (JavaScript)
    const ws = new WebSocket("ws://localhost:8000/ws/documents");
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("Event:", data);
    };
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from enum import Enum
from collections import defaultdict
import threading
import queue


# =============================================================================
# Event Types
# =============================================================================

class EventType(str, Enum):
    """Types of events that can be emitted."""
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    BATCH_INSERT = "batch_insert"
    COLLECTION_CREATED = "collection_created"
    COLLECTION_DELETED = "collection_deleted"


@dataclass
class Event:
    """An event to broadcast to subscribers."""
    type: EventType
    collection: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"{time.time_ns()}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "collection": self.collection,
            "data": self.data,
            "timestamp": self.timestamp
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class Subscription:
    """A client subscription to events."""
    collection: str
    filter: Optional[dict] = None  # Metadata filter
    event_types: Optional[list[EventType]] = None  # Filter by event type
    created_at: float = field(default_factory=time.time)

    def matches(self, event: Event) -> bool:
        """Check if this subscription matches an event."""
        # Check collection
        if self.collection != "*" and self.collection != event.collection:
            return False

        # Check event type
        if self.event_types:
            event_type = event.type if isinstance(event.type, EventType) else EventType(event.type)
            if event_type not in self.event_types:
                return False

        # Check metadata filter
        if self.filter:
            event_metadata = event.data.get("metadata", {})
            for key, value in self.filter.items():
                if event_metadata.get(key) != value:
                    return False

        return True


# =============================================================================
# Connection Manager (Async)
# =============================================================================

class AsyncConnectionManager:
    """
    Manages WebSocket connections and broadcasts events.

    For use with FastAPI/Starlette WebSocket.
    """

    def __init__(self, history_size: int = 100):
        self._connections: dict[str, list] = defaultdict(list)  # collection -> websockets
        self._subscriptions: dict = {}  # websocket -> Subscription
        self._history: list[Event] = []
        self._history_size = history_size
        self._lock = asyncio.Lock()

    async def connect(self, websocket, collection: str = "*",
                      subscription: Subscription = None):
        """
        Connect a WebSocket client.

        Args:
            websocket: FastAPI/Starlette WebSocket
            collection: Collection to subscribe to ("*" for all)
            subscription: Optional subscription with filters
        """
        await websocket.accept()
        async with self._lock:
            self._connections[collection].append(websocket)
            self._subscriptions[websocket] = subscription or Subscription(collection=collection)

            # Send recent history
            for event in self._history[-10:]:
                if self._subscriptions[websocket].matches(event):
                    try:
                        await websocket.send_text(event.to_json())
                    except Exception:
                        pass

    async def disconnect(self, websocket, collection: str = "*"):
        """Disconnect a WebSocket client."""
        async with self._lock:
            if collection in self._connections:
                if websocket in self._connections[collection]:
                    self._connections[collection].remove(websocket)
            if websocket in self._subscriptions:
                del self._subscriptions[websocket]

    async def update_subscription(self, websocket, subscription: Subscription):
        """Update a client's subscription."""
        async with self._lock:
            if websocket in self._subscriptions:
                # Update collection list
                old_sub = self._subscriptions[websocket]
                if old_sub.collection in self._connections:
                    if websocket in self._connections[old_sub.collection]:
                        self._connections[old_sub.collection].remove(websocket)

                self._connections[subscription.collection].append(websocket)
                self._subscriptions[websocket] = subscription

    async def broadcast(self, event: Event):
        """Broadcast an event to all matching subscribers."""
        # Add to history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        # Get all potential recipients
        collections_to_check = [event.collection, "*"]
        websockets_to_notify = set()

        for collection in collections_to_check:
            if collection in self._connections:
                for ws in self._connections[collection]:
                    websockets_to_notify.add(ws)

        # Send to matching subscriptions
        message = event.to_json()
        disconnected = []

        for websocket in websockets_to_notify:
            subscription = self._subscriptions.get(websocket)
            if subscription and subscription.matches(event):
                try:
                    await websocket.send_text(message)
                except Exception:
                    disconnected.append((websocket, subscription.collection))

        # Clean up disconnected clients
        for ws, collection in disconnected:
            await self.disconnect(ws, collection)

    def emit(self, event: Event):
        """
        Emit an event (sync wrapper for broadcast).

        Creates a new event loop if needed.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.broadcast(event))
            else:
                loop.run_until_complete(self.broadcast(event))
        except RuntimeError:
            asyncio.run(self.broadcast(event))

    @property
    def connection_count(self) -> int:
        """Total number of active connections."""
        return sum(len(conns) for conns in self._connections.values())


# =============================================================================
# Sync Event Bus (Thread-safe)
# =============================================================================

class EventBus:
    """
    Thread-safe event bus for synchronous code.

    Useful when you need to emit events from sync code
    and have async handlers process them.
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.Queue[Event] = queue.Queue(maxsize=maxsize)
        self._handlers: list[Callable[[Event], None]] = []
        self._async_handlers: list[Callable[[Event], Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register(self, handler: Callable[[Event], None]):
        """Register a synchronous event handler."""
        self._handlers.append(handler)

    def register_async(self, handler: Callable[[Event], Any]):
        """Register an async event handler."""
        self._async_handlers.append(handler)

    def emit(self, event: Event):
        """Emit an event to the queue."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest event
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                pass

    def start(self):
        """Start processing events in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop processing events."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                self._dispatch(event)
            except queue.Empty:
                continue

    def _dispatch(self, event: Event):
        """Dispatch event to all handlers."""
        # Sync handlers
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Event handler error: {e}")

        # Async handlers
        if self._async_handlers:
            try:
                loop = asyncio.new_event_loop()
                for handler in self._async_handlers:
                    loop.run_until_complete(handler(event))
                loop.close()
            except Exception as e:
                print(f"Async event handler error: {e}")


# =============================================================================
# Observable Collection Wrapper
# =============================================================================

class ObservableCollection:
    """
    Wrapper that emits events on collection changes.

    Usage:
        from vectordb import VectorDB
        from realtime import ObservableCollection, AsyncConnectionManager

        manager = AsyncConnectionManager()
        db = VectorDB("./data")
        collection = db.get_collection("docs")

        # Wrap with observable
        observable = ObservableCollection(collection, manager)

        # All operations now emit events
        observable.insert(vector, id="doc1", metadata={...})
    """

    def __init__(self, collection, broadcaster: AsyncConnectionManager = None,
                 event_bus: EventBus = None):
        self._collection = collection
        self._broadcaster = broadcaster
        self._event_bus = event_bus

    def _emit(self, event: Event):
        """Emit event to broadcaster and/or event bus."""
        if self._broadcaster:
            self._broadcaster.emit(event)
        if self._event_bus:
            self._event_bus.emit(event)

    def insert(self, vector, id: str = None, metadata: dict = None) -> str:
        """Insert with event emission."""
        result_id = self._collection.insert(vector, id, metadata)

        self._emit(Event(
            type=EventType.INSERT,
            collection=self._collection.config.name,
            data={
                "id": result_id,
                "metadata": metadata or {}
            }
        ))

        return result_id

    def insert_batch(self, vectors, ids: list[str] = None,
                     metadata_list: list[dict] = None) -> list[str]:
        """Batch insert with event emission."""
        result_ids = self._collection.insert_batch(vectors, ids, metadata_list)

        self._emit(Event(
            type=EventType.BATCH_INSERT,
            collection=self._collection.config.name,
            data={
                "ids": result_ids,
                "count": len(result_ids)
            }
        ))

        return result_ids

    def upsert(self, vector, id: str, metadata: dict = None) -> str:
        """Upsert with event emission."""
        result_id = self._collection.upsert(vector, id, metadata)

        self._emit(Event(
            type=EventType.UPDATE,
            collection=self._collection.config.name,
            data={
                "id": result_id,
                "metadata": metadata or {}
            }
        ))

        return result_id

    def delete(self, id: str) -> bool:
        """Delete with event emission."""
        result = self._collection.delete(id)

        if result:
            self._emit(Event(
                type=EventType.DELETE,
                collection=self._collection.config.name,
                data={"id": id}
            ))

        return result

    def search(self, query, k: int = 10, **kwargs):
        """Search with optional event emission."""
        results = self._collection.search(query, k, **kwargs)

        # Optionally emit search events (useful for analytics)
        # Uncomment if needed:
        # self._emit(Event(
        #     type=EventType.SEARCH,
        #     collection=self._collection.config.name,
        #     data={"k": k, "result_count": len(results)}
        # ))

        return results

    # Delegate other methods
    def get(self, id: str, **kwargs):
        return self._collection.get(id, **kwargs)

    def count(self) -> int:
        return self._collection.count()

    def __len__(self) -> int:
        return len(self._collection)

    @property
    def config(self):
        return self._collection.config


# =============================================================================
# FastAPI Integration
# =============================================================================

def create_websocket_routes(app, manager: AsyncConnectionManager):
    """
    Add WebSocket routes to a FastAPI app.

    Usage:
        from fastapi import FastAPI
        from realtime import AsyncConnectionManager, create_websocket_routes

        app = FastAPI()
        manager = AsyncConnectionManager()
        create_websocket_routes(app, manager)
    """
    try:
        from fastapi import WebSocket, WebSocketDisconnect
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi")

    @app.websocket("/ws")
    async def websocket_all(websocket: WebSocket):
        """Subscribe to all collections."""
        await manager.connect(websocket, "*")
        try:
            while True:
                data = await websocket.receive_text()
                # Handle subscription updates
                try:
                    msg = json.loads(data)
                    if msg.get("action") == "subscribe":
                        sub = Subscription(
                            collection=msg.get("collection", "*"),
                            filter=msg.get("filter"),
                            event_types=[EventType(t) for t in msg.get("event_types", [])] if msg.get("event_types") else None
                        )
                        await manager.update_subscription(websocket, sub)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            await manager.disconnect(websocket, "*")

    @app.websocket("/ws/{collection}")
    async def websocket_collection(websocket: WebSocket, collection: str):
        """Subscribe to a specific collection."""
        await manager.connect(websocket, collection)
        try:
            while True:
                data = await websocket.receive_text()
                # Handle filter updates
                try:
                    msg = json.loads(data)
                    if msg.get("action") == "filter":
                        sub = Subscription(
                            collection=collection,
                            filter=msg.get("filter"),
                            event_types=[EventType(t) for t in msg.get("event_types", [])] if msg.get("event_types") else None
                        )
                        await manager.update_subscription(websocket, sub)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            await manager.disconnect(websocket, collection)

    return app


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Real-time Updates Demo")
    print("=" * 60)

    # Create event bus
    event_bus = EventBus()

    # Register a handler
    def log_event(event: Event):
        print(f"[EVENT] {event.type}: {event.collection} - {event.data}")

    event_bus.register(log_event)
    event_bus.start()

    # Simulate events
    print("\nEmitting events...")

    event_bus.emit(Event(
        type=EventType.INSERT,
        collection="documents",
        data={"id": "doc1", "metadata": {"title": "Hello World"}}
    ))

    event_bus.emit(Event(
        type=EventType.INSERT,
        collection="documents",
        data={"id": "doc2", "metadata": {"title": "Goodbye World"}}
    ))

    event_bus.emit(Event(
        type=EventType.DELETE,
        collection="documents",
        data={"id": "doc1"}
    ))

    # Wait for processing
    import time
    time.sleep(0.5)

    # Test subscription matching
    print("\n--- Subscription Matching ---")

    sub_all = Subscription(collection="*")
    sub_docs = Subscription(collection="documents")
    sub_filtered = Subscription(
        collection="documents",
        filter={"category": "tech"}
    )
    sub_inserts = Subscription(
        collection="documents",
        event_types=[EventType.INSERT]
    )

    test_event = Event(
        type=EventType.INSERT,
        collection="documents",
        data={"id": "doc3", "metadata": {"category": "tech"}}
    )

    print(f"Event: {test_event.to_dict()}")
    print(f"  Matches all (*): {sub_all.matches(test_event)}")
    print(f"  Matches 'documents': {sub_docs.matches(test_event)}")
    print(f"  Matches filter(category=tech): {sub_filtered.matches(test_event)}")
    print(f"  Matches inserts only: {sub_inserts.matches(test_event)}")

    # Test with VectorDB
    print("\n--- Observable Collection ---")
    try:
        from vectordb import VectorDB

        db = VectorDB("./realtime_demo")

        if "events_test" not in db.list_collections():
            collection = db.create_collection("events_test", dimensions=128)
        else:
            collection = db.get_collection("events_test")

        # Wrap with observable
        observable = ObservableCollection(collection, event_bus=event_bus)

        # Operations will emit events
        vec = np.random.randn(128).astype(np.float32)
        observable.insert(vec, id="test1", metadata={"type": "demo"})

        time.sleep(0.5)
        db.save()

    except ImportError:
        print("VectorDB not available, skipping integration test")

    event_bus.stop()

    print("\n" + "=" * 60)
    print("WebSocket Server Example:")
    print("""
    from fastapi import FastAPI
    from realtime import AsyncConnectionManager, create_websocket_routes

    app = FastAPI()
    manager = AsyncConnectionManager()
    create_websocket_routes(app, manager)

    # In your insert endpoint:
    @app.post("/collections/{name}/vectors")
    async def insert_vector(name: str, ...):
        id = collection.insert(...)
        await manager.broadcast(Event(
            type=EventType.INSERT,
            collection=name,
            data={"id": id, ...}
        ))
        return {"id": id}

    # JavaScript client:
    const ws = new WebSocket("ws://localhost:8000/ws/documents");
    ws.onmessage = (e) => console.log(JSON.parse(e.data));
    """)
