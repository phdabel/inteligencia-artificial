from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Sequence, Tuple, List, TypeVar

S = TypeVar('S')  # tipo do estado
A = TypeVar('A')  # tipo da ação


class RLProblem(ABC, Generic[S, A]):
    """
    Interface genérica para ambientes de aprendizado por reforço.

    Ao contrário do MDP (que exige conhecimento completo das probabilidades de
    transição), o RLProblem expõe apenas uma interface de interação passo a
    passo: o agente aprende da experiência sem conhecer a dinâmica interna.

    Contrato:
    - reset()          → reinicia o ambiente e retorna o estado inicial
    - step(action)     → aplica a ação, retorna (próximo_estado, recompensa, done)
    - actions(state)   → sequência de ações disponíveis no estado
    - current_state    → propriedade com o estado corrente do ambiente

    Parâmetros de tipo
    ------------------
    S : tipo do estado  (deve ser hasheável para métodos tabulares)
    A : tipo da ação    (deve ser hasheável para métodos tabulares)
    """

    @abstractmethod
    def reset(self) -> S:
        """Reinicia o ambiente para o estado inicial e o retorna."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: A) -> Tuple[S, float, bool]:
        """
        Aplica *action* ao estado corrente.

        Retorna
        -------
        next_state : S      — próximo estado
        reward     : float  — recompensa recebida
        done       : bool   — True quando o episódio terminou
        """
        raise NotImplementedError

    @abstractmethod
    def actions(self, state: S) -> Sequence[A]:
        """Retorna as ações disponíveis em *state*."""
        raise NotImplementedError

    @property
    @abstractmethod
    def current_state(self) -> S:
        """Estado corrente (vivo) do ambiente."""
        raise NotImplementedError

    def render(self) -> None:
        """Exibição textual do estado corrente (opcional — sobrescrever se útil)."""


@dataclass
class RLResult:
    """
    Resultado agregado de uma execução de Q-learning ou SARSA.

    Campos
    ------
    q_table          : Q[estado][ação] = valor aprendido
    episode_rewards  : recompensa total de cada episódio de treino
    episodes         : número total de episódios executados
    epsilon_final    : valor de ε ao fim do treinamento
    """
    q_table: dict
    episode_rewards: List[float] = field(default_factory=list)
    episodes: int = 0
    epsilon_final: float = 0.0
