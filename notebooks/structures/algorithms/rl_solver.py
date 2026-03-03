from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

from structures.rl_problem import RLProblem, RLResult

S = TypeVar('S')
A = TypeVar('A')


# ---------------------------------------------------------------------------
# Funções auxiliares internas
# ---------------------------------------------------------------------------

def _default_q() -> float:
    return 0.0


def _epsilon_greedy(Q: dict, state, actions: Sequence, epsilon: float):
    """Seleciona ação ε-gulosa: explora aleatoriamente com probabilidade ε."""
    if random.random() < epsilon:
        return random.choice(list(actions))
    return max(actions, key=lambda a: Q[state][a])


# ---------------------------------------------------------------------------
# Q-Learning (off-policy TD control)
# ---------------------------------------------------------------------------

def q_learning(
        problem: RLProblem,
        *,
        episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        max_steps: int = 1_000,
        seed: Optional[int] = None,
) -> RLResult:
    """
    Q-Learning — controle TD off-policy (Watkins, 1989).

    Mantém uma tabela de valores de ação Q[s][a] e a atualiza após cada
    transição usando a equação de otimalidade de Bellman:

        Q(s,a) ← Q(s,a) + α · [r + γ · max_{a'} Q(s',a') − Q(s,a)]

    A exploração segue uma política ε-gulosa com ε decaindo ao longo dos
    episódios. Por ser off-policy, o agente aprende o valor da política ótima
    independentemente da política de exploração usada.

    Parâmetros
    ----------
    problem       : ambiente RL (subclasse de RLProblem)
    episodes      : número total de episódios de treinamento
    alpha         : taxa de aprendizado  (0 < α ≤ 1)
    gamma         : fator de desconto   (0 < γ ≤ 1)
    epsilon_start : probabilidade de exploração inicial
    epsilon_end   : probabilidade de exploração mínima
    epsilon_decay : decaimento multiplicativo de ε por episódio
    max_steps     : número máximo de passos por episódio (proteção)
    seed          : semente aleatória para reprodutibilidade

    Retorna
    -------
    RLResult com q_table, episode_rewards, episodes e epsilon_final
    """
    if seed is not None:
        random.seed(seed)

    Q: Dict = defaultdict(lambda: defaultdict(_default_q))
    episode_rewards: List[float] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        state = problem.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            available = problem.actions(state)
            action = _epsilon_greedy(Q, state, available, epsilon)

            next_state, reward, done = problem.step(action)
            total_reward += reward

            # Atualização off-policy: usa max sobre as ações do próximo estado
            next_actions = problem.actions(next_state)
            max_next_q = (
                max(Q[next_state][a] for a in next_actions)
                if not done else 0.0
            )
            Q[state][action] += alpha * (
                reward + gamma * max_next_q - Q[state][action]
            )

            state = next_state
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return RLResult(
        q_table=dict(Q),
        episode_rewards=episode_rewards,
        episodes=episodes,
        epsilon_final=epsilon,
    )


# ---------------------------------------------------------------------------
# SARSA (on-policy TD control)
# ---------------------------------------------------------------------------

def sarsa(
        problem: RLProblem,
        *,
        episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        max_steps: int = 1_000,
        seed: Optional[int] = None,
) -> RLResult:
    """
    SARSA — controle TD on-policy.

    Atualiza Q usando a ação *realmente executada* no próximo estado (on-
    policy), em vez do máximo over todas ações (Q-learning):

        Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]

    Isso torna o SARSA mais conservador que o Q-learning em ambientes de risco:
    ele aprende a evitar ações perigosas que a política de exploração poderia
    ocasionalmente executar — como o clássico exemplo do Cliff Walking.

    Parâmetros: idênticos ao q_learning.
    """
    if seed is not None:
        random.seed(seed)

    Q: Dict = defaultdict(lambda: defaultdict(_default_q))
    episode_rewards: List[float] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        state = problem.reset()
        total_reward = 0.0

        # Escolhe a primeira ação antes de entrar no loop
        action = _epsilon_greedy(Q, state, problem.actions(state), epsilon)

        for _ in range(max_steps):
            next_state, reward, done = problem.step(action)
            total_reward += reward

            # On-policy: escolhe next_action *antes* de atualizar Q
            next_action = _epsilon_greedy(
                Q, next_state, problem.actions(next_state), epsilon
            )

            td_target = reward + (
                0.0 if done else gamma * Q[next_state][next_action]
            )
            Q[state][action] += alpha * (td_target - Q[state][action])

            state, action = next_state, next_action
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return RLResult(
        q_table=dict(Q),
        episode_rewards=episode_rewards,
        episodes=episodes,
        epsilon_final=epsilon,
    )


# ---------------------------------------------------------------------------
# Utilitários de política e avaliação
# ---------------------------------------------------------------------------

def greedy_action(result: RLResult, problem: RLProblem, state) -> object:
    """
    Retorna a ação gulosa para *state* de acordo com a Q-table aprendida.
    Estados não visitados durante o treino retornam uma ação aleatória.
    """
    available = list(problem.actions(state))
    q_state = result.q_table.get(state, {})
    return max(available, key=lambda a: q_state.get(a, 0.0))


def evaluate_policy(
        problem: RLProblem,
        result: RLResult,
        *,
        episodes: int = 100,
        max_steps: int = 1_000,
        seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Executa *episodes* episódios com a política gulosa extraída de *result*.

    Retorna
    -------
    mean_reward : float — recompensa total média por episódio
    std_reward  : float — desvio padrão
    """
    if seed is not None:
        random.seed(seed)

    rewards = []
    for _ in range(episodes):
        state = problem.reset()
        total = 0.0
        for _ in range(max_steps):
            action = greedy_action(result, problem, state)
            state, reward, done = problem.step(action)
            total += reward
            if done:
                break
        rewards.append(total)

    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    return mean, math.sqrt(variance)


def smooth(values: List[float], window: int = 50) -> List[float]:
    """
    Média móvel simples para suavizar curvas de aprendizado em visualizações.
    """
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result
