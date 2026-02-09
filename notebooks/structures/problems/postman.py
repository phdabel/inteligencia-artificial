from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set, FrozenSet
import math
from structures.problem import Problem
from structures.graph import Graph

@dataclass(frozen=True)
class GoTo:
    to_id: str

    def __repr__(self) -> str:
        return f"go_to({self.to_id})"


@dataclass(frozen=True)
class PostmanState:
    """
    current_id: nó atual no grafo viário
    delivered: frozenset com os IDs dos endereços já entregues (apenas endereços; não precisa incluir depósito)
    """
    current_id: str
    delivered: FrozenSet[str]


# ===== Union-Find for Kruskal =====

class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        for x in items:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


# ===== Postman Problem (graph-based) =====

class PostmanProblem(Problem):
    """
    Carteiro: entregar em um conjunto de nós (endereços) no seu Graph.

    O grafo aqui representa o "mundo" (ruas/conexões).
    Cada ação é: ir para um nó vizinho via uma aresta (peso = custo).

    Entrega é considerada feita quando o carteiro *visita* o nó (endereço).
    """

    def __init__(
        self,
        road_graph: Graph,          # sua classe Graph
        depot_id: str,              # nó inicial/depósito
        addresses: Set[str],        # conjunto de nós que precisam ser visitados
        return_to_depot: bool = False,
    ):
        self.G = road_graph
        self.depot_id = depot_id
        self.addresses = set(addresses)
        self.return_to_depot = return_to_depot

        if depot_id not in self.G.nodes:
            raise ValueError(f"Depot '{depot_id}' not in graph nodes.")
        missing = [a for a in self.addresses if a not in self.G.nodes]
        if missing:
            raise ValueError(f"Some addresses not in graph nodes: {missing}")

        # cache de MST: chave = frozenset(U) onde U = nós relevantes na MST (tipicamente não visitados)
        self._mst_cache: Dict[FrozenSet[str], float] = {}

        # Pré-cálculo opcional: conjunto de arestas únicas do grafo (para uso no Kruskal)
        self._all_edges = self._extract_unique_edges()

    def initial_state(self) -> PostmanState:
        delivered = frozenset([self.depot_id]) & frozenset()  # depot não conta como entrega
        # Se o depósito também for endereço (não comum), você pode contabilizar aqui; por padrão não.
        return PostmanState(current_id=self.depot_id, delivered=frozenset())

    def is_goal(self, state: PostmanState) -> bool:
        all_delivered = self.addresses.issubset(state.delivered)
        if not all_delivered:
            return False
        if self.return_to_depot:
            return state.current_id == self.depot_id
        return True

    def successors(self, state: PostmanState) -> Iterable[Tuple[GoTo, PostmanState, float]]:
        """
        Sucessores: mover para qualquer vizinho no grafo viário.
        Ao chegar no vizinho, se ele for endereço, marca como entregue.
        """
        cur = state.current_id
        for nbr, w in self.G.neighbors(cur):
            new_delivered = state.delivered
            if nbr in self.addresses:
                new_delivered = frozenset(set(state.delivered) | {nbr})
            yield GoTo(nbr), PostmanState(nbr, new_delivered), w

    def predecessors(self, state: PostmanState) -> Iterable[Tuple[GoTo, PostmanState, float]]:
        """
        Para busca bidirecional: mover de volta para vizinhos.
        A lógica de "delivered" é a mesma, pois entregar é idempotente.
        """
        cur = state.current_id
        for nbr, w in self.G.neighbors(cur):
            new_delivered = state.delivered
            if nbr in self.addresses:
                new_delivered = frozenset(set(state.delivered) | {nbr})
            yield GoTo(nbr), PostmanState(nbr, new_delivered), w

    # ---------- Heurística MST (Kruskal) ----------

    def heuristic_mst(self, state: PostmanState) -> float:
        """
        Heurística admissível baseada em MST sobre os endereços ainda não entregues.

        U = addresses \ delivered
        h = MST_cost(U) + min_edge(current -> U) + (se return_to_depot) min_edge(U -> depot)

        Importante: isso assume que os pesos do grafo são custos reais e não-negativos.
        Se o grafo for desconexo no subgrafo induzido por U, MST não existe -> h = inf.
        """
        U = self.addresses.difference(state.delivered)
        if not U:
            # faltaria voltar ao depósito (se exigido)
            if self.return_to_depot and state.current_id != self.depot_id:
                # lower bound: pelo menos o custo de um caminho até o depósito.
                # Mas sem APSP, não temos distância mínima exata; usamos 0 como bound seguro
                # ou tentamos um bound via aresta direta se existir.
                # Aqui: 0 é sempre admissível (só mais fraca).
                return 0.0
            return 0.0

        U_frozen = frozenset(U)
        mst_cost = self._mst_cost_kruskal_over_subset(U_frozen)

        if math.isinf(mst_cost):
            return float("inf")

        # bound para conectar "current" ao conjunto U (uma aresta que sai do current e entra em U)
        min_from_cur = self._min_edge_to_set(state.current_id, U)
        if math.isinf(min_from_cur):
            # sem conexão direta; ainda pode existir caminho via intermediários,
            # mas sem APSP não dá para estimar >0 de forma segura.
            min_from_cur = 0.0

        if self.return_to_depot:
            min_to_depot = self._min_edge_to_set(self.depot_id, U)
            if math.isinf(min_to_depot):
                min_to_depot = 0.0
            return mst_cost + min_from_cur + min_to_depot

        return mst_cost + min_from_cur

    def _extract_unique_edges(self) -> List[Tuple[float, str, str]]:
        """
        Extrai lista de arestas únicas do grafo (para Kruskal).
        Para grafo não-direcionado, remove duplicatas (u,v) e (v,u).
        """
        edges: List[Tuple[float, str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for u, nbrs in self.G.adj.items():
            for v, w in nbrs:
                a, b = (u, v) if u <= v else (v, u)
                key = (a, b) if not self.G.directed else (u, v)
                if key in seen:
                    continue
                seen.add(key)
                edges.append((w, u, v))
        return edges

    def _mst_cost_kruskal_over_subset(self, nodes_subset: FrozenSet[str]) -> float:
        """
        Kruskal MST no subgrafo induzido por nodes_subset usando as arestas do Graph.
        Complexidade: O(E log E) no conjunto de arestas filtradas.
        """
        if len(nodes_subset) <= 1:
            return 0.0

        if nodes_subset in self._mst_cache:
            return self._mst_cache[nodes_subset]

        # Filtra arestas (u,v) com u e v dentro do subset
        edges_sub: List[Tuple[float, str, str]] = []
        subset = set(nodes_subset)
        for w, u, v in self._all_edges:
            if u in subset and v in subset:
                edges_sub.append((w, u, v))

        # Se não há arestas suficientes, pode ser desconexo
        # Kruskal vai detectar pelo número de unions feitas.
        edges_sub.sort(key=lambda x: x[0])  # O(E log E)

        uf = UnionFind(list(nodes_subset))
        total = 0.0
        picked = 0
        target = len(nodes_subset) - 1

        for w, u, v in edges_sub:
            if uf.union(u, v):
                total += w
                picked += 1
                if picked == target:
                    break

        if picked != target:
            total = float("inf")  # subset desconexo no grafo induzido

        self._mst_cache[nodes_subset] = total
        return total

    def _min_edge_to_set(self, from_id: str, targets: Set[str]) -> float:
        """
        Menor peso de aresta direta que conecta from_id a algum nó em targets.
        Isso é um lower bound para "conectar" sem considerar caminhos multi-hop.
        """
        best = float("inf")
        for nbr, w in self.G.neighbors(from_id):
            if nbr in targets and w < best:
                best = w
        return best