from dataclasses import dataclass
from typing import (
    Dict, List, Tuple, Optional
)

class Node:
    def __init__(self, id: str, data: Optional[dict] = None):
        self.id = id
        self.data = data or {}

    def __repr__(self) -> str:
        return f"Node({self.id})"

 
@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    weight: float = 1.0
    directed: bool = False


class Graph:
    """
    Grafo genérico: nodos + arestas, lista de adjacência.
    Pode representar grafos de restrições, grafos viários, etc.
    """
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, List[Tuple[str, float]]] = {}


    def add_node(self, node_id: str, data: Optional[dict] = None) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, data)
            self.adj[node_id] = []
        return self.nodes[node_id]
    

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))


    def neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        return self.adj.get(node_id, [])
    

    def __repr__(self) -> str:
        return f"Graph(directed={self.directed}, |V|={len(self.nodes)}, |E|={sum(len(v) for v in self.adj.values())})"