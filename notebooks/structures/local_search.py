from problem import Problem, S, A, SearchResult, _reconstruct
from typing import Dict, Tuple, Optional, List
import time
import random
import math
import heapq

def hill_climbing_search(problem: Problem[S, A], heuristic: callable = None, objective: callable = None) -> SearchResult[S, A]:
    """
    Busca local que se move para o melhor estado vizinho com base em uma função heurística (para minimização) ou função objetivo (para maximização).
        - Se `heuristic` for fornecida, o algoritmo tentará minimizar o valor da heurística.
        - Se `objective` for fornecida, o algoritmo tentará maximizar o valor da função objetivo.
        - O algoritmo termina quando não há vizinhos melhores ou quando um estado objetivo é encontrado.
        - Retorna um SearchResult indicando se a solução foi encontrada, o estado final, as ações tomadas, o custo total, e estatísticas de desempenho.
    """

    if heuristic is None and objective is None:
        raise ValueError("Either heuristic or objective function must be provided.")
    if heuristic is not None and objective is not None:
        raise ValueError("Only one of heuristic or objective function can be provided.")
    
    t0 = time.perf_counter()
    start = problem.initial_state()
    
    if problem.is_goal(start):
        return SearchResult(True, start, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)
    
    frontier: List[Tuple[float, S]] = [(heuristic(start) if heuristic else objective(start), start)]
    parent: Dict[S, Tuple[Optional[S], Optional[A]]] = {start: (None, None)}
    g_cost: Dict[S, float] = {start: 0.0}

    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        _, s = heapq.heappop(frontier)
        expanded += 1

        if problem.is_goal(s):
            actions = _reconstruct(parent, s)
            return SearchResult(True, s, actions, g_cost[s], expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

        neighbors = []
        for a, s2, cost in problem.successors(s):
            if s2 not in parent:
                neighbors.append((heuristic(s2) if heuristic else objective(s2), a, s2, cost))
        
        if not neighbors:
            continue
        
        neighbors.sort()
        best_heuristic, best_action, best_state, best_cost = neighbors[0]
        
        if heuristic is not None and best_heuristic < heuristic(s):
            parent[best_state] = (s, best_action)
            g_cost[best_state] = g_cost[s] + best_cost
            heapq.heappush(frontier, (best_heuristic, best_state))
            generated += 1
        elif objective is not None and best_heuristic > objective(s):
            parent[best_state] = (s, best_action)
            g_cost[best_state] = g_cost[s] + best_cost
            heapq.heappush(frontier, (best_heuristic, best_state))
            generated += 1
        
        max_frontier = max(max_frontier, len(frontier))

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

def simulated_annealing_search(problem: Problem[S, A], heuristic: callable, initial_temperature: float, cooling_rate: float) -> SearchResult[S, A]:
    """
    Busca local que utiliza o método de recozimento simulado para escapar de ótimos locais.
        - O algoritmo começa com uma temperatura inicial e diminui a temperatura ao longo do tempo.
        - Em cada iteração, um vizinho é escolhido aleatoriamente. Se o vizinho for melhor, ele é aceito. Se for pior, ele pode ser aceito com uma probabilidade que depende da diferença de heurística e da temperatura atual.
        - O algoritmo termina quando a temperatura atinge zero ou quando um estado objetivo é encontrado.
        - Retorna um SearchResult indicando se a solução foi encontrada, o estado final, as ações tomadas, o custo total, e estatísticas de desempenho.
    """
    if heuristic is None:
        raise ValueError("heuristic must be provided (minimization).")
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0.")
    if not (0 < cooling_rate < 1):
        raise ValueError("cooling_rate must be in the interval (0, 1).")

    t0 = time.perf_counter()

    s = problem.initial_state()
    if problem.is_goal(s):
        return SearchResult(True, s, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)

    temperature = float(initial_temperature)

    # Caminho atual (ações e custo) conforme o estado vai mudando
    actions: List[A] = []
    total_cost: float = 0.0

    # Melhor estado visto (mesmo que o atual piore temporariamente)
    h_s = float(heuristic(s))
    best_state = s
    best_h = h_s
    best_actions: List[A] = []
    best_cost: float = 0.0

    expanded = 0
    generated = 1
    max_frontier = 1  # não há fronteira real aqui; mantido por consistência com SearchResult

    eps = 1e-12

    while temperature > eps:
        expanded += 1

        if problem.is_goal(s):
            return SearchResult(True, s, actions, total_cost, expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

        neighbors = list(problem.successors(s))
        generated += len(neighbors)

        if not neighbors:
            break

        a, s2, step_cost = random.choice(neighbors)
        h2 = float(heuristic(s2))
        delta = h2 - h_s  # minimização: delta < 0 é melhor

        accept = False
        if delta < 0:
            accept = True
        else:
            # Probabilidade de aceitar piora: exp(-delta / T)
            p = math.exp(-delta / temperature) if temperature > 0 else 0.0
            accept = (random.random() < p)

        if accept:
            s = s2
            h_s = h2
            actions.append(a)
            total_cost += float(step_cost)

            if h_s < best_h:
                best_h = h_s
                best_state = s
                best_actions = actions.copy()
                best_cost = total_cost

        temperature *= cooling_rate

    # Se não encontrou objetivo, retorna o melhor estado visto durante a busca
    found = problem.is_goal(best_state)
    return SearchResult(found, best_state, best_actions, best_cost, expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)
