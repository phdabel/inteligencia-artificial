from problem import Problem, S, A, SearchResult, _reconstruct
from typing import Dict, Tuple, Optional, List
import time
from collections import deque
import heapq

def bfs(problem: Problem[S, A]) -> SearchResult[S, A]:
    """
    Busca em extensão (BFS) para problemas de busca não ponderados.
    Retorna um SearchResult com o resultado da busca.
    """
    t0 = time.perf_counter()
    start = problem.initial_state()

    frontier = deque([start])
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    cost: Dict[S, float] = {start: 0.0}

    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        s = frontier.popleft()

        if problem.is_goal(s):
            actions = _reconstruct(parent, s)
            return SearchResult(True, s, actions, cost[s], expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)
        
        expanded += 1
        for a, s2, step in problem.successors(s):
            if s2 not in parent:
                parent[s2] = (s, a)
                cost[s2] = cost[s] + step
                frontier.append(s2)
                generated += 1

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)


def dfs(problem: Problem[S, A], depth_limit: Optional[int] = None) -> SearchResult[S, A]:
    """
    Busca em profundidade (DFS) para problemas de busca não ponderados.
    Retorna um SearchResult com o resultado da busca.
    """
    t0 = time.perf_counter()
    start = problem.initial_state()

    stack = List[Tuple[S, int]] = [(start, 0)]
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    cost: Dict[S, float] = {start: 0.0}

    expanded = 0
    generated = 1
    max_frontier = 1

    while stack:
        max_frontier = max(max_frontier, len(stack))
        s, d = stack.pop()

        if problem.is_goal(s):
            actions = _reconstruct(parent, s)
            return SearchResult(True, s, actions, cost[s], expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)
        
        if depth_limit is not None and d >= depth_limit:
            continue

        expanded += 1
        for a, s2, step in problem.successors(s):
            if s2 not in parent:
                parent[s2] = (s, a)
                cost[s2] = cost[s] + step
                stack.append((s2, d + 1))
                generated += 1

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)


def ucs(problem: Problem[S, A]) -> SearchResult[S, A]:
    """
    Busca de custo uniforme (UCS) para problemas de busca ponderados.
    Se os custos forem todos iguais, UCS se comporta como BFS.
    Algoritmo ótimo e completo para custos não negativos.
    Retorna um SearchResult com o resultado da busca.
    """
    t0 = time.perf_counter()
    start = problem.initial_state()

    pq: List[Tuple[float, int, S]] = [] # priority queue de (custo, tie, estado)
    tie = 0
    heapq.heappush(pq, (0.0, tie, start))

    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    best_cost: Dict[S, float] = {start: 0.0}

    expanded = 0
    generated = 1
    max_frontier = 1

    while pq:
        max_frontier = max(max_frontier, len(pq))
        g, _, s = heapq.heappop(pq)

        if g != best_cost.get(s, float("inf")):
            continue

        if problem.is_goal(s):
            actions = _reconstruct(parent, s)
            return SearchResult(True, s, actions, g, expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)
        
        expanded += 1
        for a, s2, step in problem.successors(s):
            g2 = g + step
            if g2 < best_cost.get(s2, float("inf")):
                best_cost[s2] = g2
                parent[s2] = (s, a)
                tie += 1
                heapq.heappush(pq, (g2, tie, s2))
                generated += 1

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)


def iddfs(problem: Problem[S, A], max_depth: int) -> SearchResult[S, A]:
    """
    Busca por aprofundamento iterativo em profundidade (IDDFS).
    Retorna um SearchResult com o resultado da busca.
    Algoritmo completo para grafos finitos.
    Espaço: O(bd)
    Tempo: O(b^d)
    """
    t0 = time.perf_counter()
    total_expanded = 0
    total_generated = 0
    max_frontier = 0

    for limit in range(max_depth + 1):
        res = dfs(problem, depth_limit=limit)
        total_expanded += res.expanded
        total_generated += res.generated
        max_frontier = max(max_frontier, res.max_frontier)

        if res.found:
            return SearchResult(True, res.state, res.actions, res.path_cost, total_expanded, total_generated, max_frontier, (time.perf_counter() - t0) * 1000)
        
    return SearchResult(False, None, [], float("inf"), total_expanded, total_generated, max_frontier, (time.perf_counter() - t0) * 1000)


def bidirectional_search(problem: Problem[S, A], goal_state: S) -> SearchResult[S, A]:
    """
    Busca bidirecional para problemas de busca não ponderados.
    Retorna um SearchResult com o resultado da busca.
    
    """
