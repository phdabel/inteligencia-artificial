from dataclasses import dataclass
from typing import (
    TypeVar, Generic, Iterable, Tuple, Optional, List, Dict
)

S = TypeVar("S") # Tipo do estado
A = TypeVar("A") # Tipo da ação

class Problem(Generic[S, A]):
    """
    Classe genérica para definição de problemas de busca.
    Deve ser especializada para cada problema específico.
    """
    def initial_state(self) -> S:
        raise NotImplementedError
    
    def is_goal(self, state: S) -> bool:
        raise NotImplementedError
    
    def successors(self, state: S) -> Iterable[Tuple[A, S, float]]:
        """
        Retorna uma lista de tuplas (ação, estado_resultante, custo)
        """
        raise NotImplementedError
    
    def predecessors(self, state: S) -> Iterable[Tuple[A, S, float]]:
        """
        Opcional: para busca bidirecional sobre problemas reversíveis.
        Retorna uma lista de tuplas (ação_reversa, estado_anterior, custo)
        """
        raise NotImplementedError
    

@dataclass
class SearchResult(Generic[S, A]):
    """
    Resultado de uma busca.
    """
    found: bool # Indica se o estado objetivo foi encontrado
    state: Optional[S] # Estado objetivo encontrado (ou None)
    actions: List[A] # Sequência de ações para alcançar o estado objetivo
    path_cost: float # Custo total do caminho
    expanded: int # Número de estados expandidos
    generated: int # Número de estados gerados
    max_frontier: int # Tamanho máximo da fronteira durante a busca
    elapsed_ms: float # Tempo gasto na busca em milissegundos


def _reconstruct(parent: Dict[S, Tuple[Optional[S], Optional[A]]], goal: S) -> List[A]:
    """
    Reconstrói a sequência de ações do estado inicial até o estado objetivo.
    """
    actions: List[A] = []
    cur: Optional[S] = goal
    while cur is not None:
        prev, act = parent[cur]
        if prev is None:
            break
        actions.append(act)
        cur = prev
    actions.reverse()
    return actions