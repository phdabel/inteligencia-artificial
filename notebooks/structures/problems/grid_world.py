from structures.mdp import MDP, S, A, Transition
from typing import Dict, Tuple, Optional, Iterable, List, Sequence
import random
import math

State = Tuple[int, int] # (row, col)
Action = str

class GridWorld(MDP[S, A]):

    ACTIONS: Tuple[Action, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
    DELTAS: Dict[Action, Tuple[int, int]] = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    
    def __init__(
            self,
            n_rows: int,
            n_cols: int,
            walls: Optional[Iterable[State]] = None,
            terminals: Optional[Iterable[State]] = None,
            terminal_rewards: Optional[Dict[State, float]] = None,
            step_reward: float = -0.04,
            p_intended: float = 0.8,
            gamma: float = 0.99,
            seed: Optional[int] = 0
    ):
        
        assert 0 < p_intended <= 1.0
        assert 0 < gamma <= 1.0

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.walls = set(walls or [])
        self.terminals = set(terminals or [])
        self.terminal_rewards = dict(terminal_rewards or {})
        self.step_reward = step_reward
        self.p_intended = p_intended
        self.gamma = gamma
        self.rng = random.Random(seed)

        # Validar layout
        for (r, c) in self.walls | self.terminals:
            assert 0 <= r < n_rows and 0 <= c < n_cols, "Estado fora dos limites"
        assert len(self.walls & self.terminals) == 0, "Uma célula não pode ser parede e terminal ao mesmo tempo"

        # precomputar lista de estados
        self._states: List[State] = [
            (r, c)
            for r in range(n_rows)
            for c in range(n_cols)
            if (r, c) not in self.walls
        ]

    def gamma(self) -> float:
        return self.gamma

    def states(self) -> Sequence[State]:
        return self._states
    
    def actions(self, s: State) -> Sequence[Action]:
        return self.ACTIONS
    
    def is_terminal(self, s: State) -> bool:
        return s in self.terminals
    
    def _in_bounds(self, s: State) -> bool:
        r, c = s
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols
    
    def _move(self, s: State, a: Action) -> State:
        if self.is_terminal(s):
            return s
        
        dr, dc = self.DELTAS[a]
        r, c = s
        s2 = (r + dr, c + dc)
        if (not self._in_bounds(s2)) or (s2 in self.walls):
            return s
        return s2
    
    def transitions(self, s: State, a: Action) -> Sequence[Transition[State]]:
        if self.is_terminal(s):
            r_term = self.terminal_rewards.get(s, 0.0)
            return [Transition(s, 1.0, r_term, True)]
        
        p_other = (1.0 - self.p_intended) / 3.0
        dist: Dict[State, float] = {}

        for a2 in self.ACTIONS:
            p = self.p_intended if a2 == a else p_other
            s2 = self._move(s, a2)
            dist[s2] = dist.get(s2, 0.0) + p

        out: List[Transition] = []
        for s2, p in dist.items():
            done = self.is_terminal(s2)
            if done:
                r = self.terminal_rewards.get(s2, 0.0)
            else:
                r = self.step_reward
            out.append(Transition(s2, p, r, done))
        
        total_p = sum(t.prob for t in out)
        if not math.isclose(total_p, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            out = [Transition(t.s_next, t.prob / total_p, t.reward, t.done) for t in out]

        return out
    
    def sample_next(self, s: State, a: Action) -> Tuple[State, float, bool]:
        ts = self.transitions(s, a)
        x = self.rng.random()
        acc = 0.0
        for t in ts:
            acc += t.prob
            if x <= acc:
                return t.s_next, t.reward, t.done
            
        t = ts[-1]
        return t.s_next, t.reward, t.done
