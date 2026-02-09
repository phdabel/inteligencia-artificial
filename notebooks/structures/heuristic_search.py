from dataclasses import dataclass, field
from problem import Problem, S, A, SearchResult, _reconstruct
from typing import Dict, Tuple, Optional, List, Callable, Generic
import time
import itertools
import heapq

def greedy_best_first_search(problem: Problem[S, A], heuristic: callable) -> SearchResult[S, A]:
    t0 = time.perf_counter()
    start = problem.initial_state()
    
    if problem.is_goal(start):
        return SearchResult(True, start, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)
    
    frontier: List[Tuple[float, S]] = [(heuristic(start), start)]
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    g_cost: Dict[S, float] = {start: 0.0} # para manter o custo acumulado, mesmo que não seja usado para ordenar o frontier
    
    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        _, s = heapq.heappop(frontier)
        expanded += 1

        if problem.is_goal(s):
            actions = _reconstruct(parent, s)
            return SearchResult(True, s, actions, g_cost[s], expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

        for a, s2, cost in problem.successors(s):
            if s2 not in parent:
                parent[s2] = (s, a)
                g_cost[s2] = g_cost[s] + cost
                heapq.heappush(frontier, (heuristic(s2), s2))
                generated += 1
        
        max_frontier = max(max_frontier, len(frontier))

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

def a_star_search(problem: Problem[S, A], heuristic: callable) -> SearchResult[S, A]:
    t0 = time.perf_counter()
    start = problem.initial_state()
    
    if problem.is_goal(start):
        return SearchResult(True, start, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)
    
    counter = itertools.count()    
    frontier: List[Tuple[float, int, S]] = [(heuristic(start), next(counter), start)]
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    g_cost: Dict[S, float] = {start: 0.0}

    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        _, _, s = heapq.heappop(frontier)
        expanded += 1

        if problem.is_goal(s):
            actions = _reconstruct(parent, s)
            return SearchResult(True, s, actions, g_cost[s], expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

        for a, s2, cost in problem.successors(s):
            tentative_g_cost = g_cost[s] + cost
            if s2 not in g_cost or tentative_g_cost < g_cost[s2]:
                parent[s2] = (s, a)
                g_cost[s2] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic(s2)
                heapq.heappush(frontier, (f_cost, next(counter), s2))
                generated += 1
        
        max_frontier = max(max_frontier, len(frontier))

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

@dataclass
class _SMANode(Generic[S, A]):
    state: S
    parent: Optional["_SMANode[S, A]"]
    action: Optional[A]
    g: float
    h: float
    f: float
    depth: int
    expanded: bool = False
    children: List["_SMANode[S, A]"] = field(default_factory=list)
    best_forgotten_f: float = float("inf")  # melhor f dentre filhos removidos (esquecidos)


def sma_star_search(
    problem: Problem[S, A],
    heuristic: Callable[[S], float],
    memory_limit: int = 1000,
) -> SearchResult[S, A]:
    """
    SMA* (Simplified Memory-Bounded A*).

    - OPEN armazena apenas folhas (nós atualmente candidatos a expansão).
    - Ao exceder memory_limit (número de nós armazenados), remove a pior folha (maior f),
      guarda seu f no pai (best_forgotten_f) e faz backup de f para cima.
    - Usa pathmax: f(child) = max(f(parent), g(child)+h(child)).
    """
    t0 = time.perf_counter()
    if memory_limit < 1:
        raise ValueError("memory_limit must be >= 1")

    start = problem.initial_state()
    if problem.is_goal(start):
        return SearchResult(True, start, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)

    def reconstruct_actions(goal_node: _SMANode[S, A]) -> List[A]:
        actions: List[A] = []
        cur: Optional[_SMANode[S, A]] = goal_node
        while cur is not None and cur.parent is not None:
            # cur.action é a ação que levou do pai -> cur
            actions.append(cur.action)  # type: ignore[arg-type]
            cur = cur.parent
        actions.reverse()
        return actions

    def is_on_path(n: _SMANode[S, A], state: S) -> bool:
        cur: Optional[_SMANode[S, A]] = n
        while cur is not None:
            if cur.state == state:
                return True
            cur = cur.parent
        return False

    def effective_best_child_f(n: _SMANode[S, A]) -> float:
        best_child = min((c.f for c in n.children), default=float("inf"))
        return min(best_child, n.best_forgotten_f)

    def recompute_f(n: _SMANode[S, A]) -> float:
        # Para folha: f tende a ser g+h (ou valor “esquecido” se re-virar folha)
        # Para interno: f é o melhor caminho via filhos (incluindo esquecidos), mas nunca < g+h
        return max(n.g + n.h, effective_best_child_f(n))

    def backup_from(n: Optional[_SMANode[S, A]]) -> None:
        # Propaga mudanças de f para cima (pais dependem do min f dos filhos)
        cur = n
        while cur is not None:
            old_f = cur.f
            cur.f = recompute_f(cur) if cur.expanded else max(cur.f, cur.g + cur.h)
            if cur.f == old_f:
                break
            cur = cur.parent

    # Root
    h0 = float(heuristic(start))
    root = _SMANode(state=start, parent=None, action=None, g=0.0, h=h0, f=h0, depth=0)
    open_list: List[_SMANode[S, A]] = [root]

    expanded = 0
    generated = 1
    max_frontier = 1
    nodes_in_memory = 1

    # Tie-breakers:
    # - Para expandir: menor f, e em empate maior profundidade (tende a reduzir branching cedo)
    # - Para esquecer: maior f, e em empate maior profundidade (remove folhas mais profundas)
    def best_leaf() -> _SMANode[S, A]:
        return min(open_list, key=lambda n: (n.f, -n.depth))

    def worst_leaf() -> _SMANode[S, A]:
        return max(open_list, key=lambda n: (n.f, n.depth))

    def forget_one_leaf() -> None:
        nonlocal nodes_in_memory
        leaf = worst_leaf()
        open_list.remove(leaf)
        nodes_in_memory -= 1

        p = leaf.parent
        if p is None:
            # Só seria possível se memory_limit == 0 (não permitido) ou OPEN degenerado.
            return

        # Remover do conjunto de filhos do pai, e registrar o melhor f “esquecido”
        p.best_forgotten_f = min(p.best_forgotten_f, leaf.f)
        if leaf in p.children:
            p.children.remove(leaf)

        # Se o pai ficou sem filhos armazenados, ele volta a ser folha (para permitir re-expansão)
        if p.expanded and len(p.children) == 0 and p not in open_list:
            # f do pai deve refletir o melhor filho esquecido (ou inf se dead-end)
            p.f = max(p.g + p.h, p.best_forgotten_f)
            open_list.append(p)

        backup_from(p)

    while open_list:
        n = best_leaf()
        open_list.remove(n)

        # Se n já é objetivo quando chega ao topo, retorna.
        if problem.is_goal(n.state):
            actions = reconstruct_actions(n)
            return SearchResult(
                True,
                n.state,
                actions,
                n.g,
                expanded,
                generated,
                max_frontier,
                (time.perf_counter() - t0) * 1000,
            )

        expanded += 1

        # Reexpansão: se ele virou folha por esquecimento, vamos (re)gerar filhos do zero
        n.expanded = True
        n.children.clear()
        n.best_forgotten_f = float("inf")

        any_child = False
        for a, s2, cost in problem.successors(n.state):
            # Evita ciclos no caminho (tree-search)
            if is_on_path(n, s2):
                continue

            g2 = n.g + float(cost)
            h2 = float(heuristic(s2))
            f2 = max(n.f, g2 + h2)  # pathmax

            child = _SMANode(
                state=s2,
                parent=n,
                action=a,
                g=g2,
                h=h2,
                f=f2,
                depth=n.depth + 1,
            )
            n.children.append(child)
            open_list.append(child)

            nodes_in_memory += 1
            generated += 1
            any_child = True

            # Enforce memory eagerly (evita crescer muito)
            while nodes_in_memory > memory_limit and len(open_list) > 0:
                forget_one_leaf()

        if not any_child:
            # Dead-end: força f = inf e faz backup
            n.f = float("inf")
            backup_from(n.parent)
        else:
            # Atualiza f do nó interno a partir dos filhos
            n.f = recompute_f(n)
            backup_from(n.parent)

        max_frontier = max(max_frontier, len(open_list))

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)