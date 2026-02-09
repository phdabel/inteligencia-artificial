from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
from structures.problem import Problem

@dataclass(frozen=True)
class Move:
    tile: int

    def __repr__(self) -> str:
        return f"move({self.tile})"
    
@dataclass(frozen=True)
class PuzzleState:
    """
    tiles: tupla de n*n inteiros representando a configuração do tabuleiro, onde 0 é o espaço vazio
    Exemplo 3x3: (1, 2, 3, 4, 5, 6, 7, 8, 0)
    Exemplo 4x4: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    """
    tiles: Tuple[int, ...]

class EightPuzzleProblem(Problem[PuzzleState, Move]):
    def __init__(
        self,
        initial: Tuple[int, ...],
        goal: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0),
    ):
        self._initial = PuzzleState(initial)
        self._goal = PuzzleState(goal)

        if len(self._initial.tiles) != len(self._goal.tiles):
            raise ValueError("initial e goal devem ter o mesmo tamanho")

        n = int(math.isqrt(len(self._initial.tiles)))
        if n * n != len(self._initial.tiles):
            raise ValueError("O número de tiles deve ser um quadrado perfeito (n*n).")

        self._n = n

    def initial_state(self) -> PuzzleState:
        return self._initial

    def is_goal(self, state: PuzzleState) -> bool:
        return state == self._goal

    def successors(self, state: PuzzleState) -> List[Tuple[Move, PuzzleState, float]]:
        n = self._n
        zero_index = state.tiles.index(0)
        row, col = divmod(zero_index, n)

        moves: List[Tuple[Move, PuzzleState, float]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < n and 0 <= new_col < n:
                new_zero_index = new_row * n + new_col
                tile_to_move = state.tiles[new_zero_index]
                new_tiles = list(state.tiles)
                new_tiles[zero_index], new_tiles[new_zero_index] = new_tiles[new_zero_index], new_tiles[zero_index]
                moves.append((Move(tile_to_move), PuzzleState(tuple(new_tiles)), 1.0))
        return moves