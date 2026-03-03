from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple

from structures.rl_problem import RLProblem


# ---------------------------------------------------------------------------
# Estado e Ação
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CliffState:
    """Posição (linha, coluna) no grid do Cliff Walking."""
    row: int
    col: int

    def __repr__(self) -> str:
        return f"({self.row},{self.col})"


# Constantes de ação
UP    = "UP"
RIGHT = "RIGHT"
DOWN  = "DOWN"
LEFT  = "LEFT"
_ALL_ACTIONS = (UP, RIGHT, DOWN, LEFT)

_DELTAS: dict = {
    UP:    (-1,  0),
    RIGHT: ( 0,  1),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
}


# ---------------------------------------------------------------------------
# Problema
# ---------------------------------------------------------------------------

class CliffWalking(RLProblem[CliffState, str]):
    """
    Cliff Walking — benchmark clássico de RL (Sutton & Barto, cap. 6).

    Layout do grid 4 × 12:

        .  .  .  .  .  .  .  .  .  .  .  .   ← linha 0
        .  .  .  .  .  .  .  .  .  .  .  .   ← linha 1
        .  .  .  .  .  .  .  .  .  .  .  .   ← linha 2
        S  C  C  C  C  C  C  C  C  C  C  G   ← linha 3
        ↑                                ↑
      início                            meta

    S = início  (linha 3, coluna 0)
    G = meta    (linha 3, coluna 11)
    C = penhasco (linha 3, colunas 1–10) → penalidade −100, reset ao início

    Recompensas
    -----------
    - Cada passo       : −1
    - Entrar no penasco: −100  (episódio continua, agente volta ao início)
    - Chegar à meta    : episódio termina

    Ações: UP, RIGHT, DOWN, LEFT (determinísticas).

    Este ambiente é ideal para comparar Q-learning e SARSA:
    - Q-learning aprende a rota ótima global (beira do penhasco)
    - SARSA aprende uma rota mais segura por cima, evitando o risco de
      escorregar para o penhasco durante a exploração
    """

    ROWS  = 4
    COLS  = 12
    START = CliffState(3, 0)
    GOAL  = CliffState(3, 11)
    CLIFF = frozenset(CliffState(3, c) for c in range(1, 11))

    def __init__(self) -> None:
        self._state: CliffState = self.START

    # --- RLProblem ---------------------------------------------------------

    @property
    def current_state(self) -> CliffState:
        return self._state

    def reset(self) -> CliffState:
        self._state = self.START
        return self._state

    def step(self, action: str) -> Tuple[CliffState, float, bool]:
        dr, dc = _DELTAS[action]
        row = max(0, min(self.ROWS - 1, self._state.row + dr))
        col = max(0, min(self.COLS - 1, self._state.col + dc))
        next_state = CliffState(row, col)

        if next_state in self.CLIFF:
            # Caiu no penhasco: penalidade e reset ao início
            self._state = self.START
            return self.START, -100.0, False

        self._state = next_state

        if next_state == self.GOAL:
            return next_state, -1.0, True

        return next_state, -1.0, False

    def actions(self, state: CliffState) -> Sequence[str]:
        return _ALL_ACTIONS

    # --- Utilitários -------------------------------------------------------

    def render(self) -> None:
        """Imprime uma representação textual do grid."""
        for r in range(self.ROWS):
            row_str = ""
            for c in range(self.COLS):
                s = CliffState(r, c)
                if s == self._state:
                    row_str += " A "
                elif s == self.GOAL:
                    row_str += " G "
                elif s in self.CLIFF:
                    row_str += " C "
                elif s == self.START:
                    row_str += " S "
                else:
                    row_str += " . "
            print(row_str)
        print()

    def render_policy(self, q_table: dict) -> None:
        """Imprime a política gulosa derivada da Q-table."""
        arrows = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}
        for r in range(self.ROWS):
            row_str = ""
            for c in range(self.COLS):
                s = CliffState(r, c)
                if s == self.GOAL:
                    row_str += " G "
                elif s in self.CLIFF:
                    row_str += " C "
                elif s in q_table:
                    best = max(_ALL_ACTIONS, key=lambda a: q_table[s].get(a, 0.0))
                    row_str += f" {arrows[best]} "
                else:
                    row_str += " ? "
            print(row_str)
        print()
