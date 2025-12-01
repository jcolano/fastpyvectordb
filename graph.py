"""
Graph Database Module

A property graph database with nodes, edges, and Cypher-like queries.
Integrates with vector search for hybrid graph+vector queries.

Usage:
    from graph import GraphDB, NodeBuilder, EdgeBuilder

    db = GraphDB()

    # Create nodes
    alice = db.create_node(
        NodeBuilder()
        .label("Person")
        .property("name", "Alice")
        .property("age", 30)
        .build()
    )

    bob = db.create_node(
        NodeBuilder()
        .label("Person")
        .property("name", "Bob")
        .build()
    )

    # Create edge
    db.create_edge(
        EdgeBuilder(alice, bob, "KNOWS")
        .property("since", 2020)
        .build()
    )

    # Query
    results = db.query("MATCH (p:Person)-[:KNOWS]->(friend) RETURN friend.name")
"""

import uuid
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator
from collections import defaultdict
from pathlib import Path
from enum import Enum
import threading


# =============================================================================
# Core Types
# =============================================================================

PropertyValue = str | int | float | bool | list | dict


@dataclass
class Node:
    """A node in the graph."""
    id: str
    labels: set[str] = field(default_factory=set)
    properties: dict[str, PropertyValue] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "labels": list(self.labels),
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        return cls(
            id=data["id"],
            labels=set(data.get("labels", [])),
            properties=data.get("properties", {})
        )


@dataclass
class Edge:
    """An edge (relationship) between two nodes."""
    id: str
    from_node: str  # Node ID
    to_node: str    # Node ID
    type: str       # Relationship type (e.g., "KNOWS", "LIKES")
    properties: dict[str, PropertyValue] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.id == other.id
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from": self.from_node,
            "to": self.to_node,
            "type": self.type,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Edge':
        return cls(
            id=data["id"],
            from_node=data["from"],
            to_node=data["to"],
            type=data["type"],
            properties=data.get("properties", {})
        )


@dataclass
class Hyperedge:
    """A hyperedge connecting multiple nodes."""
    id: str
    nodes: list[str]  # List of node IDs
    type: str
    properties: dict[str, PropertyValue] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "nodes": self.nodes,
            "type": self.type,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Hyperedge':
        return cls(
            id=data["id"],
            nodes=data["nodes"],
            type=data["type"],
            properties=data.get("properties", {})
        )


# =============================================================================
# Builders (Fluent API)
# =============================================================================

class NodeBuilder:
    """Fluent builder for creating nodes."""

    def __init__(self, node_id: str = None):
        self._id = node_id or str(uuid.uuid4())
        self._labels: set[str] = set()
        self._properties: dict[str, PropertyValue] = {}

    def id(self, node_id: str) -> 'NodeBuilder':
        self._id = node_id
        return self

    def label(self, label: str) -> 'NodeBuilder':
        self._labels.add(label)
        return self

    def labels(self, *labels: str) -> 'NodeBuilder':
        self._labels.update(labels)
        return self

    def property(self, key: str, value: PropertyValue) -> 'NodeBuilder':
        self._properties[key] = value
        return self

    def properties(self, props: dict) -> 'NodeBuilder':
        self._properties.update(props)
        return self

    def build(self) -> Node:
        return Node(
            id=self._id,
            labels=self._labels,
            properties=self._properties
        )


class EdgeBuilder:
    """Fluent builder for creating edges."""

    def __init__(self, from_node: str | Node, to_node: str | Node, edge_type: str):
        self._id = str(uuid.uuid4())
        self._from = from_node.id if isinstance(from_node, Node) else from_node
        self._to = to_node.id if isinstance(to_node, Node) else to_node
        self._type = edge_type
        self._properties: dict[str, PropertyValue] = {}

    def id(self, edge_id: str) -> 'EdgeBuilder':
        self._id = edge_id
        return self

    def property(self, key: str, value: PropertyValue) -> 'EdgeBuilder':
        self._properties[key] = value
        return self

    def properties(self, props: dict) -> 'EdgeBuilder':
        self._properties.update(props)
        return self

    def build(self) -> Edge:
        return Edge(
            id=self._id,
            from_node=self._from,
            to_node=self._to,
            type=self._type,
            properties=self._properties
        )


class HyperedgeBuilder:
    """Fluent builder for creating hyperedges."""

    def __init__(self, nodes: list[str | Node], edge_type: str):
        self._id = str(uuid.uuid4())
        self._nodes = [n.id if isinstance(n, Node) else n for n in nodes]
        self._type = edge_type
        self._properties: dict[str, PropertyValue] = {}

    def id(self, edge_id: str) -> 'HyperedgeBuilder':
        self._id = edge_id
        return self

    def property(self, key: str, value: PropertyValue) -> 'HyperedgeBuilder':
        self._properties[key] = value
        return self

    def build(self) -> Hyperedge:
        return Hyperedge(
            id=self._id,
            nodes=self._nodes,
            type=self._type,
            properties=self._properties
        )


# =============================================================================
# Indexes
# =============================================================================

class LabelIndex:
    """Index for fast label-based lookups."""

    def __init__(self):
        self._label_to_nodes: dict[str, set[str]] = defaultdict(set)

    def add(self, node: Node):
        for label in node.labels:
            self._label_to_nodes[label].add(node.id)

    def remove(self, node: Node):
        for label in node.labels:
            self._label_to_nodes[label].discard(node.id)

    def get_by_label(self, label: str) -> set[str]:
        return self._label_to_nodes.get(label, set()).copy()

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in self._label_to_nodes.items()}

    @classmethod
    def from_dict(cls, data: dict) -> 'LabelIndex':
        idx = cls()
        for label, node_ids in data.items():
            idx._label_to_nodes[label] = set(node_ids)
        return idx


class AdjacencyIndex:
    """Index for fast neighbor lookups."""

    def __init__(self):
        self._outgoing: dict[str, set[str]] = defaultdict(set)  # node_id -> edge_ids
        self._incoming: dict[str, set[str]] = defaultdict(set)  # node_id -> edge_ids

    def add(self, edge: Edge):
        self._outgoing[edge.from_node].add(edge.id)
        self._incoming[edge.to_node].add(edge.id)

    def remove(self, edge: Edge):
        self._outgoing[edge.from_node].discard(edge.id)
        self._incoming[edge.to_node].discard(edge.id)

    def get_outgoing(self, node_id: str) -> set[str]:
        return self._outgoing.get(node_id, set()).copy()

    def get_incoming(self, node_id: str) -> set[str]:
        return self._incoming.get(node_id, set()).copy()

    def get_all(self, node_id: str) -> set[str]:
        return self.get_outgoing(node_id) | self.get_incoming(node_id)

    def to_dict(self) -> dict:
        return {
            "outgoing": {k: list(v) for k, v in self._outgoing.items()},
            "incoming": {k: list(v) for k, v in self._incoming.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AdjacencyIndex':
        idx = cls()
        for node_id, edge_ids in data.get("outgoing", {}).items():
            idx._outgoing[node_id] = set(edge_ids)
        for node_id, edge_ids in data.get("incoming", {}).items():
            idx._incoming[node_id] = set(edge_ids)
        return idx


class EdgeTypeIndex:
    """Index for fast edge type lookups."""

    def __init__(self):
        self._type_to_edges: dict[str, set[str]] = defaultdict(set)

    def add(self, edge: Edge):
        self._type_to_edges[edge.type].add(edge.id)

    def remove(self, edge: Edge):
        self._type_to_edges[edge.type].discard(edge.id)

    def get_by_type(self, edge_type: str) -> set[str]:
        return self._type_to_edges.get(edge_type, set()).copy()

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in self._type_to_edges.items()}

    @classmethod
    def from_dict(cls, data: dict) -> 'EdgeTypeIndex':
        idx = cls()
        for edge_type, edge_ids in data.items():
            idx._type_to_edges[edge_type] = set(edge_ids)
        return idx


# =============================================================================
# Graph Database
# =============================================================================

class GraphDB:
    """
    In-memory property graph database with persistence.

    Features:
    - Nodes with labels and properties
    - Directed edges (relationships) with types and properties
    - Hyperedges (connecting 3+ nodes)
    - Label, adjacency, and edge type indexes
    - Simple Cypher-like query support
    - JSON persistence
    """

    def __init__(self, path: str = None):
        self._path = Path(path) if path else None
        if self._path:
            self._path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Storage
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._hyperedges: dict[str, Hyperedge] = {}

        # Indexes
        self._label_index = LabelIndex()
        self._adjacency_index = AdjacencyIndex()
        self._edge_type_index = EdgeTypeIndex()

        # Load from disk
        if self._path:
            self._load()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load(self):
        """Load graph from disk."""
        data_path = self._path / "graph.json"
        if not data_path.exists():
            return

        with open(data_path, "r") as f:
            data = json.load(f)

        # Load nodes
        for node_data in data.get("nodes", []):
            node = Node.from_dict(node_data)
            self._nodes[node.id] = node
            self._label_index.add(node)

        # Load edges
        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            self._edges[edge.id] = edge
            self._adjacency_index.add(edge)
            self._edge_type_index.add(edge)

        # Load hyperedges
        for he_data in data.get("hyperedges", []):
            he = Hyperedge.from_dict(he_data)
            self._hyperedges[he.id] = he

    def save(self):
        """Save graph to disk."""
        if not self._path:
            return

        with self._lock:
            data = {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges.values()],
                "hyperedges": [h.to_dict() for h in self._hyperedges.values()]
            }

            with open(self._path / "graph.json", "w") as f:
                json.dump(data, f, indent=2)

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def create_node(self, node: Node) -> str:
        """Create a node."""
        with self._lock:
            if node.id in self._nodes:
                raise ValueError(f"Node '{node.id}' already exists")

            self._nodes[node.id] = node
            self._label_index.add(node)
            return node.id

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, properties: dict = None,
                    labels: set[str] = None) -> bool:
        """Update a node's properties and/or labels."""
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            if labels is not None:
                # Update label index
                self._label_index.remove(node)
                node.labels = labels
                self._label_index.add(node)

            if properties is not None:
                node.properties.update(properties)

            return True

    def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node ID to delete
            cascade: If True, delete connected edges. If False, fail if edges exist.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return False

            # Check for connected edges
            connected_edges = self._adjacency_index.get_all(node_id)
            if connected_edges and not cascade:
                raise ValueError(f"Node '{node_id}' has connected edges. Use cascade=True to delete.")

            # Delete connected edges
            for edge_id in connected_edges:
                self.delete_edge(edge_id)

            # Remove from indexes
            self._label_index.remove(node)

            # Delete node
            del self._nodes[node_id]
            return True

    def get_nodes_by_label(self, label: str) -> list[Node]:
        """Get all nodes with a specific label."""
        node_ids = self._label_index.get_by_label(label)
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def find_nodes(self, label: str = None, properties: dict = None) -> list[Node]:
        """Find nodes by label and/or properties."""
        if label:
            candidates = self.get_nodes_by_label(label)
        else:
            candidates = list(self._nodes.values())

        if properties:
            candidates = [
                n for n in candidates
                if all(n.properties.get(k) == v for k, v in properties.items())
            ]

        return candidates

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def create_edge(self, edge: Edge) -> str:
        """Create an edge."""
        with self._lock:
            if edge.id in self._edges:
                raise ValueError(f"Edge '{edge.id}' already exists")

            if edge.from_node not in self._nodes:
                raise ValueError(f"Source node '{edge.from_node}' not found")

            if edge.to_node not in self._nodes:
                raise ValueError(f"Target node '{edge.to_node}' not found")

            self._edges[edge.id] = edge
            self._adjacency_index.add(edge)
            self._edge_type_index.add(edge)
            return edge.id

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        with self._lock:
            edge = self._edges.get(edge_id)
            if not edge:
                return False

            self._adjacency_index.remove(edge)
            self._edge_type_index.remove(edge)
            del self._edges[edge_id]
            return True

    def get_edges_by_type(self, edge_type: str) -> list[Edge]:
        """Get all edges of a specific type."""
        edge_ids = self._edge_type_index.get_by_type(edge_type)
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_outgoing_edges(self, node_id: str, edge_type: str = None) -> list[Edge]:
        """Get outgoing edges from a node."""
        edge_ids = self._adjacency_index.get_outgoing(node_id)
        edges = [self._edges[eid] for eid in edge_ids if eid in self._edges]
        if edge_type:
            edges = [e for e in edges if e.type == edge_type]
        return edges

    def get_incoming_edges(self, node_id: str, edge_type: str = None) -> list[Edge]:
        """Get incoming edges to a node."""
        edge_ids = self._adjacency_index.get_incoming(node_id)
        edges = [self._edges[eid] for eid in edge_ids if eid in self._edges]
        if edge_type:
            edges = [e for e in edges if e.type == edge_type]
        return edges

    # -------------------------------------------------------------------------
    # Hyperedge Operations
    # -------------------------------------------------------------------------

    def create_hyperedge(self, hyperedge: Hyperedge) -> str:
        """Create a hyperedge."""
        with self._lock:
            if hyperedge.id in self._hyperedges:
                raise ValueError(f"Hyperedge '{hyperedge.id}' already exists")

            for node_id in hyperedge.nodes:
                if node_id not in self._nodes:
                    raise ValueError(f"Node '{node_id}' not found")

            self._hyperedges[hyperedge.id] = hyperedge
            return hyperedge.id

    def get_hyperedge(self, hyperedge_id: str) -> Optional[Hyperedge]:
        """Get a hyperedge by ID."""
        return self._hyperedges.get(hyperedge_id)

    def delete_hyperedge(self, hyperedge_id: str) -> bool:
        """Delete a hyperedge."""
        with self._lock:
            if hyperedge_id in self._hyperedges:
                del self._hyperedges[hyperedge_id]
                return True
            return False

    def get_hyperedges_by_node(self, node_id: str) -> list[Hyperedge]:
        """Get all hyperedges containing a node."""
        return [h for h in self._hyperedges.values() if node_id in h.nodes]

    # -------------------------------------------------------------------------
    # Traversal
    # -------------------------------------------------------------------------

    def neighbors(self, node_id: str, direction: str = "out",
                  edge_type: str = None) -> list[Node]:
        """
        Get neighbor nodes.

        Args:
            node_id: Starting node
            direction: "out" (outgoing), "in" (incoming), or "both"
            edge_type: Filter by edge type
        """
        neighbors = []

        if direction in ("out", "both"):
            for edge in self.get_outgoing_edges(node_id, edge_type):
                node = self.get_node(edge.to_node)
                if node:
                    neighbors.append(node)

        if direction in ("in", "both"):
            for edge in self.get_incoming_edges(node_id, edge_type):
                node = self.get_node(edge.from_node)
                if node:
                    neighbors.append(node)

        return neighbors

    def traverse(self, start_node: str, edge_type: str = None,
                 max_depth: int = 3, direction: str = "out") -> list[list[Node]]:
        """
        Traverse the graph from a starting node.

        Returns all paths up to max_depth.
        """
        paths = []
        start = self.get_node(start_node)
        if not start:
            return paths

        def dfs(node: Node, path: list[Node], depth: int):
            if depth > max_depth:
                return

            current_path = path + [node]
            if len(current_path) > 1:
                paths.append(current_path)

            for neighbor in self.neighbors(node.id, direction, edge_type):
                if neighbor not in path:  # Avoid cycles
                    dfs(neighbor, current_path, depth + 1)

        dfs(start, [], 0)
        return paths

    def shortest_path(self, from_node: str, to_node: str,
                      edge_type: str = None, max_depth: int = 10) -> Optional[list[Node]]:
        """Find shortest path between two nodes using BFS."""
        if from_node == to_node:
            node = self.get_node(from_node)
            return [node] if node else None

        start = self.get_node(from_node)
        end = self.get_node(to_node)
        if not start or not end:
            return None

        from collections import deque

        visited = {from_node}
        queue = deque([(start, [start])])

        while queue:
            node, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for neighbor in self.neighbors(node.id, "out", edge_type):
                if neighbor.id == to_node:
                    return path + [neighbor]

                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor, path + [neighbor]))

        return None

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def hyperedge_count(self) -> int:
        return len(self._hyperedges)

    def stats(self) -> dict:
        """Get database statistics."""
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "hyperedges": self.hyperedge_count(),
            "labels": list(self._label_index._label_to_nodes.keys()),
            "edge_types": list(self._edge_type_index._type_to_edges.keys())
        }


# =============================================================================
# Simple Query Language (Cypher-like subset)
# =============================================================================

class QueryResult:
    """Result of a graph query."""

    def __init__(self, columns: list[str], rows: list[dict]):
        self.columns = columns
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def to_list(self) -> list[dict]:
        return self.rows


class SimpleQueryParser:
    """
    Simple Cypher-like query parser.

    Supports:
    - MATCH (n:Label) RETURN n
    - MATCH (n:Label {prop: value}) RETURN n
    - MATCH (a)-[:TYPE]->(b) RETURN a, b
    - MATCH (a)-[:TYPE*1..3]->(b) RETURN b  (variable length paths)
    """

    def __init__(self, db: GraphDB):
        self.db = db

    def execute(self, query: str) -> QueryResult:
        """Execute a simple query."""
        query = query.strip()

        # Parse MATCH clause
        match_pattern = re.search(r'MATCH\s+(.+?)\s+(?:WHERE|RETURN)', query, re.IGNORECASE)
        if not match_pattern:
            raise ValueError("Invalid query: MATCH clause required")

        pattern = match_pattern.group(1)

        # Parse WHERE clause (optional)
        where_clause = None
        where_match = re.search(r'WHERE\s+(.+?)\s+RETURN', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)

        # Parse RETURN clause
        return_match = re.search(r'RETURN\s+(.+)$', query, re.IGNORECASE)
        if not return_match:
            raise ValueError("Invalid query: RETURN clause required")

        return_clause = return_match.group(1)

        # Execute pattern matching
        bindings = self._match_pattern(pattern)

        # Apply WHERE filter
        if where_clause:
            bindings = self._apply_where(bindings, where_clause)

        # Format results
        return self._format_return(bindings, return_clause)

    def _match_pattern(self, pattern: str) -> list[dict]:
        """Match a graph pattern."""
        # Simple node pattern: (n:Label) or (n:Label {prop: value})
        node_match = re.match(r'\((\w+)(?::(\w+))?(?:\s*\{(.+)\})?\)', pattern)
        if node_match:
            var, label, props_str = node_match.groups()
            props = self._parse_props(props_str) if props_str else None
            nodes = self.db.find_nodes(label, props)
            return [{var: node} for node in nodes]

        # Relationship pattern: (a)-[:TYPE]->(b) or (a)-[:TYPE*min..max]->(b)
        rel_match = re.match(
            r'\((\w+)(?::(\w+))?\)-\[:(\w+)(?:\*(\d+)\.\.(\d+))?\]->\((\w+)(?::(\w+))?\)',
            pattern
        )
        if rel_match:
            a_var, a_label, rel_type, min_depth, max_depth, b_var, b_label = rel_match.groups()
            min_depth = int(min_depth) if min_depth else 1
            max_depth = int(max_depth) if max_depth else 1

            bindings = []
            start_nodes = self.db.find_nodes(a_label) if a_label else list(self.db._nodes.values())

            for start in start_nodes:
                if max_depth == 1:
                    # Direct neighbors
                    for neighbor in self.db.neighbors(start.id, "out", rel_type):
                        if b_label and b_label not in neighbor.labels:
                            continue
                        bindings.append({a_var: start, b_var: neighbor})
                else:
                    # Variable length paths
                    paths = self.db.traverse(start.id, rel_type, max_depth, "out")
                    for path in paths:
                        if len(path) >= min_depth + 1:
                            end = path[-1]
                            if b_label and b_label not in end.labels:
                                continue
                            bindings.append({a_var: start, b_var: end})

            return bindings

        raise ValueError(f"Unsupported pattern: {pattern}")

    def _parse_props(self, props_str: str) -> dict:
        """Parse property string like 'name: "Alice", age: 30'."""
        props = {}
        for part in props_str.split(','):
            key_val = part.strip().split(':')
            if len(key_val) == 2:
                key = key_val[0].strip()
                val = key_val[1].strip().strip('"\'')
                # Try to parse as number
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                props[key] = val
        return props

    def _apply_where(self, bindings: list[dict], where_clause: str) -> list[dict]:
        """Apply WHERE clause filter."""
        # Simple property comparison: n.prop = value
        match = re.match(r'(\w+)\.(\w+)\s*(=|<|>|<=|>=|<>)\s*(.+)', where_clause)
        if match:
            var, prop, op, value = match.groups()
            value = value.strip().strip('"\'')

            # Try to parse as number
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            ops = {
                '=': lambda a, b: a == b,
                '<>': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '>': lambda a, b: a > b,
                '<=': lambda a, b: a <= b,
                '>=': lambda a, b: a >= b,
            }

            return [
                b for b in bindings
                if var in b and ops[op](b[var].properties.get(prop), value)
            ]

        return bindings

    def _format_return(self, bindings: list[dict], return_clause: str) -> QueryResult:
        """Format RETURN clause results."""
        columns = [c.strip() for c in return_clause.split(',')]
        rows = []

        for binding in bindings:
            row = {}
            for col in columns:
                if '.' in col:
                    var, prop = col.split('.', 1)
                    if var in binding:
                        row[col] = binding[var].properties.get(prop)
                else:
                    if col in binding:
                        row[col] = binding[col].to_dict()
            rows.append(row)

        return QueryResult(columns, rows)


# Add query method to GraphDB
def query(self, cypher: str) -> QueryResult:
    """Execute a Cypher-like query."""
    parser = SimpleQueryParser(self)
    return parser.execute(cypher)

GraphDB.query = query


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Graph Database Demo")
    print("=" * 60)

    # Create database
    db = GraphDB("./graph_demo")

    # Create nodes
    print("\nCreating nodes...")

    alice = db.create_node(
        NodeBuilder()
        .label("Person")
        .property("name", "Alice")
        .property("age", 30)
        .build()
    )

    bob = db.create_node(
        NodeBuilder()
        .label("Person")
        .property("name", "Bob")
        .property("age", 25)
        .build()
    )

    charlie = db.create_node(
        NodeBuilder()
        .label("Person")
        .property("name", "Charlie")
        .property("age", 35)
        .build()
    )

    python = db.create_node(
        NodeBuilder()
        .label("Skill")
        .property("name", "Python")
        .property("level", "advanced")
        .build()
    )

    rust = db.create_node(
        NodeBuilder()
        .label("Skill")
        .property("name", "Rust")
        .property("level", "intermediate")
        .build()
    )

    print(f"Created {db.node_count()} nodes")

    # Create edges
    print("\nCreating edges...")

    db.create_edge(EdgeBuilder(alice, bob, "KNOWS").property("since", 2020).build())
    db.create_edge(EdgeBuilder(bob, charlie, "KNOWS").property("since", 2021).build())
    db.create_edge(EdgeBuilder(alice, charlie, "KNOWS").property("since", 2019).build())
    db.create_edge(EdgeBuilder(alice, python, "HAS_SKILL").build())
    db.create_edge(EdgeBuilder(alice, rust, "HAS_SKILL").build())
    db.create_edge(EdgeBuilder(bob, python, "HAS_SKILL").build())

    print(f"Created {db.edge_count()} edges")

    # Create hyperedge
    print("\nCreating hyperedge...")
    db.create_hyperedge(
        HyperedgeBuilder([alice, bob, charlie], "TEAM")
        .property("name", "Core Team")
        .build()
    )
    print(f"Created {db.hyperedge_count()} hyperedges")

    # Query: Find all people
    print("\n--- Query: MATCH (p:Person) RETURN p.name, p.age ---")
    results = db.query("MATCH (p:Person) RETURN p.name, p.age")
    for row in results:
        print(f"  {row}")

    # Query: Find people who know others
    print("\n--- Query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name ---")
    results = db.query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
    for row in results:
        print(f"  {row}")

    # Query: Find people with skills
    print("\n--- Query: MATCH (p:Person)-[:HAS_SKILL]->(s:Skill) RETURN p.name, s.name ---")
    results = db.query("MATCH (p:Person)-[:HAS_SKILL]->(s:Skill) RETURN p.name, s.name")
    for row in results:
        print(f"  {row}")

    # Query with WHERE
    print("\n--- Query: MATCH (p:Person) WHERE p.age > 28 RETURN p.name ---")
    results = db.query("MATCH (p:Person) WHERE p.age > 28 RETURN p.name")
    for row in results:
        print(f"  {row}")

    # Traversal
    print("\n--- Traversal: Paths from Alice (depth 2) ---")
    paths = db.traverse(alice, max_depth=2)
    for path in paths:
        path_str = " -> ".join(n.properties.get("name", n.id) for n in path)
        print(f"  {path_str}")

    # Shortest path
    print("\n--- Shortest path: Alice to Charlie ---")
    path = db.shortest_path(alice, charlie)
    if path:
        path_str = " -> ".join(n.properties.get("name", n.id) for n in path)
        print(f"  {path_str}")

    # Save
    db.save()
    print("\n" + "=" * 60)
    print(f"Stats: {db.stats()}")
    print("Demo complete!")
