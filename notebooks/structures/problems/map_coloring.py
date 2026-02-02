from dataclasses import dataclass
from structures.problem import Problem
from structures.graph import Graph
from typing import (List, Dict, Tuple, Any, Optional, Iterable)

@dataclass(frozen=True)
class AssignAction:
    region: str
    color: int

    def __repr__(self) -> str:
        return f"assign({self.region}={self.color})"
    
@dataclass(frozen=True)
class UnassignAction:
    region: str

    def __repr__(self) -> str:
        return f"unassign({self.region})"
    
@dataclass(frozen=True)
class ColoringState:
    """
    O estado do problema de coloração de mapas é representado 
    por um conjunto de atribuições de cores às regiões.
    O estado inicial é o estado vazio, sem atribuições.

    O estado deve ser imutável e hashable para ser usado em buscas.
    Estado imutável para hashing:
        - assignments é uma tupla (region, color) ordenada por region
    """
    assignments: Tuple[Tuple[str, int], ...] # tupla ordenada de (região, cor)

    def as_dict(self) -> Dict[str, int]:
        return dict(self.assignments)
    
class MapColoringProblem(Problem[ColoringState, Any]):
    def __init__(self, constraint_graph: Graph, colors: List[int], order: List[str]):
        self.G = constraint_graph
        self.colors = colors
        self.order = order

        missing = [v for v in order if v not in self.G.nodes]
        if missing:
            raise ValueError(f"Order contains unknown regions: {missing}")
        
    def initial_state(self) -> ColoringState:
        return ColoringState(assignments=tuple())
    
    def is_goal(self, state: ColoringState) -> bool:
        # Objetivo: todas as regiões atribuídas
        # A consistência será feita na geração dos sucessores, então aqui só verificamos o número de atribuições.
        return len(state.assignments) == len(self.order)
    
    def _next_region(self, state: ColoringState) -> Optional[str]:
        k = len(state.assignments)
        if k >= len(self.order):
            return None
        return self.order[k]
    
    def _is_consistent(self, A: Dict[str, int], region: str, color: int) -> bool:
        for nbr, _w in self.G.neighbors(region):
            if nbr in A and A[nbr] == color:
                return False
        return True
    
    def successors(self, state: ColoringState) -> Iterable[Tuple[Any, ColoringState, float]]:
        A = state.as_dict()
        region = self._next_region(state)
        if region is None:
            return
        
        for c in self.colors:
            if self._is_consistent(A, region, c):
                A2 = dict(A)
                A2[region] = c

                # armazenar atribuição ordenada pela região para manter o hash consistente
                next_state = ColoringState(assignments=tuple(sorted(A2.items())))
                yield AssignAction(region, c), next_state, 1.0

    def predecessors(self, state: ColoringState) -> Iterable[Tuple[Any, ColoringState, float]]:
        """
        Para a busca bidirecional sobre uma atribuição objetivo conhecida.
        Gera estados predecessores removendo a última atribuição feita.
        """
        A = state.as_dict()

        # Determinar quantos estão atribuídos e qual é o último atribuído
        k = len(A)
        if k == 0:
            return
        
        last_region = self.order[k - 1]
        if last_region not in A:
            # Inconsistente, não deveria acontecer
            return
        
        A2 = dict(A)
        del A2[last_region]
        prev_state = ColoringState(assignments=tuple(sorted(A2.items())))
        yield UnassignAction(last_region), prev_state, 1.0
        
    
