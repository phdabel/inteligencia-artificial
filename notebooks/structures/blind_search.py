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

    stack: List[Tuple[S, int]] = [(start, 0)]
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
    
    A busca bidirecional requer:
        - um estado objetivo específico, não apenas um objetivo teste.
        - a habilidade de gerar predecessores (estados anteriores), ou sucessores simétricos.
    """
    t0 = time.perf_counter()
    start = problem.initial_state()

    if start == goal_state:
        return SearchResult(True, start, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)
    
    q_f = deque([start])
    q_b = deque([goal_state])

    parent_f: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    parent_b: Dict[S, Tuple[Optional[S], Optional[A]]] = {goal_state: (None, None)}

    expanded = 0
    generated = 2
    max_frontier = 2

    meet: Optional[S] = None

    def expand_frontier(
            q: deque[S],
            parent_this: Dict[S, Tuple[Optional[S], Optional[A]]],
            parent_other: Dict[S, Tuple[Optional[S], Optional[A]]],
            forward: bool
    ) -> Optional[S]:
        nonlocal expanded, generated
        s = q.popleft()
        expanded += 1

        if forward:
            steps = problem.successors(s)
        else:
            steps = problem.predecessors(s)

        for a, s2, _ in steps:
            if s2 not in parent_this:
                parent_this[s2] = (s, a)
                q.append(s2)
                generated += 1
                if s2 in parent_other:
                    return s2
        return None
    
    while q_f and q_b:
        max_frontier = max(max_frontier, len(q_f) + len(q_b))

        # expandir a fronteira menor
        if len(q_f) < len(q_b):
            meet = expand_frontier(q_f, parent_f, parent_b, forward=True)
        else:
            meet = expand_frontier(q_b, parent_b, parent_f, forward=False)

        if meet is not None:
            break

    if meet is None:
        return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)
    
    # reconstruir o caminho: start (forward) -> meet -> goal (backward)
    actions_f = _reconstruct(parent_f, meet)

    # reconstruir o caminho backward e inverter
    actions_b_rev: List[A] = []
    cur = meet
    while cur != goal_state:
        prev, act = parent_b[cur]
        if prev is None:
            break
        actions_b_rev.append(act)
        cur = prev

    actions = actions_f + actions_b_rev

    return SearchResult(True, meet, actions, float("nan"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

