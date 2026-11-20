"""Tabular Q-learning / SARSA on ONE fixed map (the classic "Q-table" baseline).

State = (row, col, battery): the map is fixed, so the terrain never changes and the table only needs the rover's
position and remaining battery. Q has shape (n, n, b_max + 1, 4).

    Q-learning (off-policy):  Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    SARSA      (on-policy):   Q(s,a) += alpha * (r + gamma * Q(s',a')        - Q(s,a))

A true terminal state contributes no bootstrap term; a time-limit truncation does (it is not a real end of the task).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5   # fraction of episodes over which epsilon decays linearly
    q_init: float = 0.0           # optimistic initialisation > 0 encourages exploration
    sarsa: bool = False


def state_of(env):
    """Tabular state from a RoverEnv (position + battery; the map is fixed)."""
    return env.pos[0], env.pos[1], int(env.battery)


class QTableAgent:
    def __init__(self, n: int, b_max: int, cfg: QConfig, n_actions: int = 4, seed: int = 0):
        self.cfg, self.n_actions = cfg, n_actions
        self.Q = np.full((n, n, b_max + 1, n_actions), cfg.q_init, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def epsilon(self, episode: int, episodes: int) -> float:
        c = self.cfg
        decay = max(1, int(episodes * c.eps_decay_frac))
        return max(c.eps_end, c.eps_start - (c.eps_start - c.eps_end) * episode / decay)

    def greedy(self, s) -> int:
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # random tie-break (matters while the table is empty)

    def act(self, s, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def update(self, s, a, r, s2, terminal: bool, a2=None) -> float:
        c = self.cfg
        if terminal:
            nxt = 0.0
        elif c.sarsa:
            nxt = self.Q[s2][a2]
        else:
            nxt = self.Q[s2].max()
        td = r + c.gamma * nxt - self.Q[s][a]
        self.Q[s][a] += c.alpha * td
        return float(td)
