from typing import Dict, Mapping, Optional
from structures.mdp import MDP, S, A

def value_iteration(
        mdp: MDP[S, A],
        theta: float = 1e-8,
        max_iters: int = 10_000,
        init_v: float = 0.0
) -> Dict[S, float]:
    V: Dict[S, float] = {s: init_v for s in mdp.states()}

    for _ in range(max_iters):
        delta = 0.0

        for s in mdp.states():
            best = -float('inf')

            acts = mdp.actions(s)
            if not acts:
                continue

            for a in acts:
                q = 0.0
                for t in mdp.transitions(s, a):
                    q += t.prob * (t.reward + mdp.gamma * (0.0 if t.done else V[t.s_next]))
                best = max(best, q)

            old = V[s]
            V[s] = best
            delta = max(delta, abs(old - V[s]))

        if delta < theta:
            break

    return V


def greedy_policy_from_v(mdp: MDP[S, A], V: Mapping[S, float]) -> Dict[S, A]:
    pi: Dict[S, A] = {}
    for s in mdp.states():
        acts = mdp.actions(s)
        if not acts:
            continue

        best_a: Optional[A] = None
        best_q = -float('inf')
        for a in acts:
            q = 0.0
            for t in mdp.transitions(s, a):
                q += t.prob * (t.reward + mdp.gamma * (0.0 if t.done else V[t.s_next]))
            if q > best_q:
                best_q = q
                best_a = a

        if best_a is not None:
            pi[s] = best_a

    return pi
    
def evaluate_policy(
        mdp: MDP[S, A],
        pi: Mapping[S, A],
        theta: float = 1e-8,
        max_iters: int = 10_000,
        init_v: float = 0.0
) -> Dict[S, float]:
    V: Dict[S, float] = {s: init_v for s in mdp.states()}

    for _ in range(max_iters):
        delta = 0.0

        for s in mdp.states():
            if s not in pi:
                continue
            a = pi[s]
            v_new = 0.0
            for t in mdp.transitions(s, a):
                v_new += t.prob * (t.reward + mdp.gamma * (0.0 if t.done else V[t.s_next]))
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < theta:
            break

    return V




