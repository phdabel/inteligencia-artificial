from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from structures.rl_problem import RLProblem


# ---------------------------------------------------------------------------
# Constantes físicas do CartPole
# ---------------------------------------------------------------------------

_GRAVITY      = 9.8        # m/s²
_MASS_CART    = 1.0        # kg
_MASS_POLE    = 0.1        # kg
_HALF_LEN     = 0.5        # metade do comprimento do pêndulo [m]
_FORCE_MAG    = 10.0       # N — magnitude da força aplicada ao carrinho
_DT           = 0.02       # s  — passo de tempo de integração
_X_THRESHOLD  = 2.4        # m  — limite lateral do trilho
_THETA_THRESH = 12.0 * math.pi / 180.0   # rad ≈ 0.2094 — ângulo máximo


# ---------------------------------------------------------------------------
# Estado e Ação
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PendulumState:
    """
    Estado discretizado do CartPole em 4 dimensões.
    Cada campo é o índice do bin correspondente à variável contínua.
    """
    x_bin:        int   # posição do carrinho
    xdot_bin:     int   # velocidade do carrinho
    theta_bin:    int   # ângulo do pêndulo
    thetadot_bin: int   # velocidade angular do pêndulo


# Constantes de ação (inteiros para eficiência de indexação)
PUSH_LEFT  = 0
PUSH_RIGHT = 1
_ALL_ACTIONS = (PUSH_LEFT, PUSH_RIGHT)


# ---------------------------------------------------------------------------
# Problema
# ---------------------------------------------------------------------------

class InvertedPendulum(RLProblem[PendulumState, int]):
    """
    Pêndulo invertido no carrinho (CartPole) com dinâmica contínua e
    representação de estado discreta para Q-learning tabular.

    Física
    ------
    Um pêndulo rígido está preso à parte superior de um carrinho que se move
    sobre um trilho sem atrito. O agente aplica uma força discreta (+F ou −F)
    ao carrinho a cada passo de tempo.

    Estado contínuo: (x, ẋ, θ, θ̇)
        x   — posição do carrinho [m]
        ẋ   — velocidade do carrinho [m/s]
        θ   — ângulo do pêndulo a partir da vertical [rad]  (+ = direita)
        θ̇   — velocidade angular do pêndulo [rad/s]

    Ações: PUSH_LEFT (0) ou PUSH_RIGHT (1)

    O episódio termina quando:
        |x|  > 2.4 m  (carrinho sai dos limites)
        |θ|  > 12°    (pêndulo cai)
        t    ≥ max_steps  (equilíbrio bem-sucedido — vitória!)

    Recompensa: +1.0 por cada passo em equilíbrio.

    Discretização
    -------------
    O espaço de estado contínuo é mapeado para bins inteiros com n_bins
    divisões por dimensão, produzindo n_bins⁴ estados discretos (ex: 10⁴ =
    10 000 estados para n_bins=10). Isso permite o uso direto de Q-learning
    tabular sem aproximação de função.

    Parâmetros
    ----------
    n_bins    : número de bins por dimensão de estado (padrão 10)
    max_steps : limite de passos por episódio (padrão 200)
    seed      : semente para posições iniciais aleatórias
    """

    # Limites (lo, hi) de cada dimensão para discretização
    _BOUNDS = [
        (-_X_THRESHOLD * 2,   _X_THRESHOLD * 2),    # x
        (-3.0,                 3.0),                  # ẋ
        (-_THETA_THRESH * 2,  _THETA_THRESH * 2),    # θ
        (-3.5,                 3.5),                  # θ̇
    ]

    def __init__(
            self,
            n_bins: int = 10,
            max_steps: int = 200,
            seed: Optional[int] = None,
    ) -> None:
        self.n_bins    = n_bins
        self.max_steps = max_steps
        self._rng      = random.Random(seed)

        # Variáveis de estado contínuo (usadas internamente para física)
        self._x:         float = 0.0
        self._xdot:      float = 0.0
        self._theta:     float = 0.0
        self._thetadot:  float = 0.0
        self._steps:     int   = 0

    # --- RLProblem ---------------------------------------------------------

    @property
    def current_state(self) -> PendulumState:
        return self._discretize(self._x, self._xdot, self._theta, self._thetadot)

    def reset(self) -> PendulumState:
        """Reinicia próximo à posição vertical com pequena perturbação aleatória."""
        self._x        = self._rng.uniform(-0.05, 0.05)
        self._xdot     = self._rng.uniform(-0.05, 0.05)
        self._theta    = self._rng.uniform(-0.05, 0.05)
        self._thetadot = self._rng.uniform(-0.05, 0.05)
        self._steps    = 0
        return self.current_state

    def step(self, action: int) -> Tuple[PendulumState, float, bool]:
        force = _FORCE_MAG if action == PUSH_RIGHT else -_FORCE_MAG
        self._physics_step(force)
        self._steps += 1

        done = (
            abs(self._x)     > _X_THRESHOLD
            or abs(self._theta) > _THETA_THRESH
            or self._steps  >= self.max_steps
        )

        return self.current_state, 1.0, done

    def actions(self, state: PendulumState) -> Sequence[int]:
        return _ALL_ACTIONS

    # --- Física (Euler explícito) ------------------------------------------

    def _physics_step(self, force: float) -> None:
        """
        Integra as equações de movimento do CartPole por um passo de tempo _DT.

        Equações derivadas da mecânica Lagrangiana (Barto et al., 1983):

            temp       = (F + m_p · l · θ̇² · sin θ) / M_total
            θ̈  = (g · sin θ − cos θ · temp) / (l · (4/3 − m_p cos²θ / M))
            ẍ  = temp − m_p · l · θ̈ · cos θ / M_total

        Integração de Euler: x ← x + ẋ·dt, ẋ ← ẋ + ẍ·dt  (idem para θ).
        """
        cos_th = math.cos(self._theta)
        sin_th = math.sin(self._theta)
        m_total = _MASS_CART + _MASS_POLE
        pm_l    = _MASS_POLE * _HALF_LEN

        temp      = (force + pm_l * self._thetadot ** 2 * sin_th) / m_total
        theta_acc = (
            (_GRAVITY * sin_th - cos_th * temp)
            / (_HALF_LEN * (4.0 / 3.0 - _MASS_POLE * cos_th ** 2 / m_total))
        )
        x_acc = temp - pm_l * theta_acc * cos_th / m_total

        self._x        += _DT * self._xdot
        self._xdot     += _DT * x_acc
        self._theta    += _DT * self._thetadot
        self._thetadot += _DT * theta_acc

    # --- Discretização -----------------------------------------------------

    def _discretize(
            self, x: float, xdot: float, theta: float, thetadot: float
    ) -> PendulumState:
        """Mapeia o estado contínuo para o índice de bin discreto."""
        def to_bin(val: float, lo: float, hi: float) -> int:
            clipped = max(lo, min(hi - 1e-9, val))
            return int((clipped - lo) / (hi - lo) * self.n_bins)

        vals = [x, xdot, theta, thetadot]
        bins = [to_bin(v, lo, hi) for v, (lo, hi) in zip(vals, self._BOUNDS)]
        return PendulumState(*bins)

    # --- Utilitários -------------------------------------------------------

    @property
    def continuous_state(self) -> Tuple[float, float, float, float]:
        """Estado contínuo corrente (x, ẋ, θ, θ̇) para visualização."""
        return self._x, self._xdot, self._theta, self._thetadot

    def render(self) -> None:
        """Representação textual simples do carrinho e pêndulo."""
        bar_width = 41
        frac      = (self._x + _X_THRESHOLD) / (2.0 * _X_THRESHOLD)
        cart_col  = int(max(0, min(bar_width - 1, frac * bar_width)))
        track     = list("─" * bar_width)
        track[cart_col] = "█"
        angle_deg = math.degrees(self._theta)
        print(f"  {''.join(track)}")
        print(f"  x={self._x:+.3f} m  θ={angle_deg:+.2f}°  passo={self._steps}")
