from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Generic, Sequence, TypeVar
)
from abc import ABC, abstractmethod
import math

S = TypeVar('S') # state type
A = TypeVar('A') # action type

@dataclass(frozen=True)
class Transition(Generic[S]):
    """
    Uma saída de tomar uma ação a em um estado s: s' com probabilidade, recomepensa e um indicador de término.
    """
    s_next: S
    prob: float
    reward: float
    done: bool

class MDP(Generic[S, A], ABC):
    """
    Interface genérica para MDP

    Contrato:
    - states() retorna todos os estados definidos/alcançáveis (MDP finito) ou um conjunto de estados representativos (MDP infinito)
    - actions(s) reotrna as ações disponíveis em um estado s
    - transitions(s, a) retorna uma lista finita de saídas de Transition cuja probabilidade somam 1.0
    - gamma é o fator de desconto em (0,1]
    """

    @property
    @abstractmethod
    def gamma(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def states(self) -> Sequence[S]:
        raise NotImplementedError
    
    @abstractmethod
    def actions(self, s: S) -> Sequence[A]:
        raise NotImplementedError
    
    @abstractmethod
    def transitions(self, s: S, a: A) -> Sequence[Transition[S]]:
        raise NotImplementedError
    
    def is_terminal(self, s: S) -> bool:
        """
        Sobrescrita opcional.
        Por padrão: termina se e somente se todas as ações levam para done = True com probabilidade 1.
        """
        for a in self.actions(s):
            ts = self.transitions(s, a)
            if not ts:
                continue
            if not all(t.done for t in ts):
                return False
        return True
    
    def check_valid(self, tol: float = 1e-9) -> None:
        """
        Sanity check: probabilidades somam para ~1. para cada (s,a)
        """
        assert 0 < self.gamma <= 1.0
        for s in self.states():
            for a in self.actions(s):
                ts = self.transitions(s, a)
                p = sum(t.prob for t in ts)
                assert math.isclose(p, 1.0, rel_tol=tol, abs_tol=tol), (s, a, p)