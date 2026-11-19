"""PPO-Clip agent (Schulman et al., 2017) with GAE (AE 696 Lecture 18).

Actor-critic with a clipped surrogate  l(r, A) = min(r*A, clip(r, 1-eps, 1+eps)*A),
GAE advantages computed backward over a fixed rollout, critic targets R = A_raw + V_old,
and the combined loss  J = -L_clip + c_v * J_V - c_e * H.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4, hidden: int = 256):
        super().__init__()

        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out))
        self.actor, self.critic = mlp(n_actions), mlp(1)

    def dist(self, x) -> Categorical:
        return Categorical(logits=self.actor(x))

    def value(self, x):
        return self.critic(x).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    n_envs: int = 8          # E parallel environments
    n_steps: int = 128       # H steps per env per rollout  -> M = E*H transitions
    batch_size: int = 256    # minibatch size B
    n_epochs: int = 4        # K passes over each rollout
    vf_coef: float = 0.5     # c_v
    ent_coef: float = 0.01   # c_e
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02  # early-stop the epochs if approx KL exceeds this (None = off)
    normalize_adv: bool = True
    hidden: int = 256


def compute_gae(rewards, values, last_values, dones, gamma, lam):
    """GAE over arrays shaped (H, E), computed backward in time.

    dones[t] = 1 when the episode ended at step t (true terminal OR time-limit reset): both stop the trace (c_t = 0).
    Bootstrapping past a time-limit is handled by the caller (reward already includes gamma*V(final_obs)).
    Returns raw advantages (H, E); critic targets are advantages + values.
    """
    H = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_values)
    for t in reversed(range(H)):
        nxt_v = last_values if t == H - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]                                    # b_t = c_t = 0 at an episode boundary
        delta = rewards[t] + gamma * nonterm * nxt_v - values[t]    # TD residual
        last = delta + gamma * lam * nonterm * last                 # A_t = delta_t + gamma*lam*A_{t+1}
        adv[t] = last
    return adv


class PPOAgent:
    def __init__(self, obs_dim: int, cfg: PPOConfig, n_actions: int = 4, device: str = "cpu"):
        self.cfg, self.n_actions, self.device = cfg, n_actions, torch.device(device)
        self.ac = ActorCritic(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        """Greedy action (argmax of the policy) -- matches the DQN agent's act(obs, eps=0.0) used by evaluate()."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.ac.dist(x).probs.argmax(1))

    @torch.no_grad()
    def sample(self, obs_batch: np.ndarray):
        """Sample actions for a batch of observations; returns (actions, log_probs, values) as numpy."""
        x = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        d = self.ac.dist(x)
        a = d.sample()
        return a.cpu().numpy(), d.log_prob(a).cpu().numpy(), self.ac.value(x).cpu().numpy()

    @torch.no_grad()
    def values(self, obs_batch: np.ndarray) -> np.ndarray:
        return self.ac.value(torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)).cpu().numpy()

    def update(self, obs, actions, old_logp, adv_raw, returns) -> dict:
        """Optimise on one fixed rollout. Inputs are flattened (M, ...) numpy arrays; old_logp, adv_raw and
        returns (= adv_raw + V_old) stay fixed for all K epochs."""
        c = self.cfg
        dev = self.device
        obs, actions = torch.as_tensor(obs, device=dev), torch.as_tensor(actions, device=dev)
        old_logp, returns = torch.as_tensor(old_logp, device=dev), torch.as_tensor(returns, device=dev)
        adv = torch.as_tensor(adv_raw, device=dev)
        if c.normalize_adv:  # only for the actor weights; critic targets use the raw advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        M = obs.shape[0]
        stats = dict(policy_loss=[], value_loss=[], entropy=[], approx_kl=[], clip_frac=[])
        stop = False
        for _ in range(c.n_epochs):
            perm = torch.randperm(M, device=dev)
            for i in range(0, M, c.batch_size):
                idx = perm[i:i + c.batch_size]
                d = self.ac.dist(obs[idx])
                logp = d.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                l_clip = torch.min(ratio * a, torch.clamp(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * a).mean()
                v_loss = ((self.ac.value(obs[idx]) - returns[idx]) ** 2).mean()
                ent = d.entropy().mean()
                loss = -l_clip + c.vf_coef * v_loss - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    log_r = logp - old_logp[idx]
                    kl = float(((ratio - 1) - log_r).mean())  # k3 approximate KL(old || new)
                    stats["policy_loss"].append(float(-l_clip)); stats["value_loss"].append(float(v_loss))
                    stats["entropy"].append(float(ent)); stats["approx_kl"].append(kl)
                    stats["clip_frac"].append(float(((ratio - 1).abs() > c.clip_eps).float().mean()))
                if c.target_kl is not None and kl > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break
        return {k: float(np.mean(v)) for k, v in stats.items()}
