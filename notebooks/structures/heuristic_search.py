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
