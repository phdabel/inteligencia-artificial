from problem import Problem, S, A, SearchResult, _reconstruct
from typing import Dict, Tuple, Optional, List
import time
import random
import math
import heapq
import itertools

def hill_climbing_search(problem: Problem[S, A], heuristic: callable = None, objective: callable = None) -> SearchResult[S, A]:
    """
    Busca local que se move para o melhor estado vizinho com base em uma função heurística (para minimização) ou função objetivo (para maximização).
        - Se `heuristic` for fornecida, o algoritmo tentará minimizar o valor da heurística.
        - Se `objective` for fornecida, o algoritmo tentará maximizar o valor da função objetivo.
        - O algoritmo termina quando não há vizinhos melhores ou quando um estado objetivo é encontrado.
        - Retorna um SearchResult indicando se a solução foi encontrada, o estado final, as ações tomadas, o custo total, e estatísticas de desempenho.

    Hill Climbing é um algoritmo de busca local que pode ser eficiente para encontrar soluções em grandes espaços de busca, mas pode ficar preso em ótimos locais. 
    Ele é adequado para problemas onde a função heurística ou objetivo tem uma estrutura que permite uma boa orientação da busca.
    Contudo, é importante notar que o Hill Climbing pode não encontrar a solução ótima global se houver múltiplos picos ou vales no espaço de busca, e pode ser sensível à escolha do estado inicial.
    """

    if heuristic is None and objective is None:
        raise ValueError("Either heuristic or objective function must be provided.")
    if heuristic is not None and objective is not None:
        raise ValueError("Only one of heuristic or objective function can be provided.")
    
    t0 = time.perf_counter()
    start = problem.initial_state()
    
    if problem.is_goal(start):
        return SearchResult(True, start, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)
    
    # Para evitar comparar S em empates no heap, adiciona-se um tie-breaker incremental
    tie = itertools.count()

    # Usar prioridade sempre em modo "min-heap" para facilitar a comparação, mesmo para maximização (negando o valor)
    # - heuristic: prioridade = h(s) (minimização)
    # - objective: prioridade = -f(s) (maximização)
    if heuristic is not None:
        start_priority = float(heuristic(start))
    else:
        start_priority = -float(objective(start))

    frontier: List[Tuple[float, int, S]] = [(start_priority, next(tie), start)]
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

        current_h = float(heuristic(s)) if heuristic is not None else None
        current_obj = float(objective(s)) if objective is not None else None

        best_candidate = None # (a, s2, step_Cost, h2, obj2)
        for a, s2, cost in problem.successors(s):
            if s2 in parent:
                continue

            if heuristic is not None:
                h2 = float(heuristic(s2))
                if best_candidate is None or h2 < best_candidate[3]:
                    best_candidate = (a, s2, float(cost), h2, None)
            else:
                obj2 = float(objective(s2))
                if best_candidate is None or obj2 > best_candidate[4]:
                    best_candidate = (a, s2, float(cost), None, obj2)

        if best_candidate is None:
            continue

        a, best_state, best_cost, best_h, best_obj = best_candidate
        
        improved = False
        if heuristic is not None:
            improved = (best_h is not None and current_h is not None and best_h < current_h)
            best_priority = best_h
        else:
            improved = (best_obj is not None and current_obj is not None and best_obj > current_obj)
            best_priority = -best_obj

        if improved:
            parent[best_state] = (s, a)
            g_cost[best_state] = g_cost[s] + best_cost
            heapq.heappush(frontier, (best_priority, next(tie), best_state))
            generated += 1

        max_frontier = max(max_frontier, len(frontier))

    return SearchResult(False, None, [], float("inf"), expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

def simulated_annealing_search(
    problem: Problem[S, A],
    heuristic: callable,
    initial_temperature: float,
    cooling_rate: float,
    *,
    steps_per_temp: int = 50,
    max_steps: int = 200_000,
    seed: Optional[int] = None,
) -> SearchResult[S, A]:
    """
    Têmpera simulada (minimização).

    Observação: aqui `cooling_rate` é um FATOR multiplicativo (alpha), então deve ser próximo de 1,
    por exemplo 0.995, 0.999. Valores pequenos como 0.04 congelam quase imediatamente.
    """
    if heuristic is None:
        raise ValueError("heuristic must be provided (minimization).")
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be > 0.")
    if not (0 < cooling_rate < 1):
        raise ValueError("cooling_rate must be in the interval (0, 1).")
    if steps_per_temp <= 0:
        raise ValueError("steps_per_temp must be > 0.")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0.")

    rng = random.Random(seed)

    t0 = time.perf_counter()

    s = problem.initial_state()
    if problem.is_goal(s):
        return SearchResult(True, s, [], 0.0, 0, 1, 1, (time.perf_counter() - t0) * 1000)

    temperature = float(initial_temperature)

    actions: List[A] = []
    total_cost: float = 0.0

    h_s = float(heuristic(s))
    best_state = s
    best_h = h_s
    best_actions: List[A] = []
    best_cost: float = 0.0

    expanded = 0
    generated = 1
    max_frontier = 1

    eps = 1e-12
    steps = 0

    while temperature > eps and steps < max_steps:
        for _ in range(steps_per_temp):
            if steps >= max_steps:
                break

            expanded += 1
            steps += 1

            if problem.is_goal(s):
                return SearchResult(True, s, actions, total_cost, expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)

            neighbors = list(problem.successors(s))
            generated += len(neighbors)
            if not neighbors:
                break

            a, s2, step_cost = rng.choice(neighbors)
            h2 = float(heuristic(s2))
            delta = h2 - h_s  # minimização

            if delta < 0:
                accept = True
            else:
                p = math.exp(-delta / temperature)
                accept = (rng.random() < p)

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

    found = problem.is_goal(best_state)
    return SearchResult(found, best_state, best_actions, best_cost, expanded, generated, max_frontier, (time.perf_counter() - t0) * 1000)
