"""Double DQN agent (van Hasselt et al., 2016): online net selects a*, target net evaluates it."""
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        s, a, r, s2, d = zip(*random.sample(self.buf, batch_size))
        return (torch.as_tensor(np.stack(s)), torch.as_tensor(a), torch.as_tensor(r, dtype=torch.float32),
                torch.as_tensor(np.stack(s2)), torch.as_tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


@dataclass
class DQNConfig:
    double: bool = True  # False -> vanilla DQN for ablation
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup: int = 1_000
    target_sync: int = 500  # hard-update period (gradient steps)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 5_000
    grad_clip: float = 10.0


class DQNAgent:
    def __init__(self, obs_dim: int, cfg: DQNConfig, n_actions: int = 4, device: str = "cpu"):
        self.cfg, self.n_actions, self.device = cfg, n_actions, torch.device(device)
        self.q = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.updates = 0

    def epsilon(self, episode: int) -> float:
        c = self.cfg
        return max(c.eps_end, c.eps_start - (c.eps_start - c.eps_end) * episode / c.eps_decay_episodes)

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        return int(self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0)).argmax(1))

    def update(self):
        c = self.cfg
        if len(self.buffer) < max(c.warmup, c.batch_size):
            return None
        s, a, r, s2, d = (t.to(self.device) for t in self.buffer.sample(c.batch_size))
        q_sa = self.q(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if c.double:
                a_star = self.q(s2).argmax(1, keepdim=True)          # select with online net
                q_next = self.target(s2).gather(1, a_star).squeeze(1)  # evaluate with target net
            else:
                q_next = self.target(s2).max(1).values
            y = r + c.gamma * (1.0 - d) * q_next
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % c.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss)
