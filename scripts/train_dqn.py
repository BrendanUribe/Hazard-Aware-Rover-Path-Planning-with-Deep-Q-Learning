"""Train Double DQN on randomly generated maps. Usage: python scripts/train_dqn.py --episodes 20000 --seed 0"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from rover.dqn import DQNAgent, DQNConfig
from rover.env import RoverEnv
from rover.evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vanilla", action="store_true", help="use vanilla DQN instead of Double DQN")
    p.add_argument("--p-slip", type=float, default=0.30)
    p.add_argument("--b0", type=int, default=15)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--target-sync", type=int, default=500, help="hard target-update period (gradient steps)")
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-frac", type=float, default=0.25, help="fraction of episodes over which epsilon decays")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--eval-maps", type=int, default=5, help="unseen maps per evaluation (x20 episodes each)")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    env_kw = dict(p_slip=a.p_slip, b0=a.b0, size=a.size)
    env = RoverEnv(**env_kw)
    cfg = DQNConfig(double=not a.vanilla, gamma=a.gamma, lr=a.lr, batch_size=a.batch_size, buffer_size=a.buffer_size,
                    warmup=a.warmup, target_sync=a.target_sync, eps_end=a.eps_end,
                    eps_decay_episodes=max(1, int(a.episodes * a.eps_decay_frac)), grad_clip=a.grad_clip,
                    hidden=a.hidden)
    agent = DQNAgent(env.obs_dim, cfg)
    name = f"{'dqn' if a.vanilla else 'ddqn'}_seed{a.seed}"
    out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    log, ep_returns, steps = [], [], 0

    obs, _ = env.reset(seed=a.seed)
    for ep in range(1, a.episodes + 1):
        if ep > 1:
            obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            act = agent.act(obs, agent.epsilon(ep))
            obs2, r, term, trunc, _ = env.step(act)
            agent.buffer.push(obs, act, r, obs2, float(term))  # truncation is not a true terminal
            agent.update()
            obs, ret, done = obs2, ret + r, term or trunc
            steps += 1
        ep_returns.append(ret)
        if ep % a.eval_every == 0:
            m = evaluate(agent, n_maps=a.eval_maps, **env_kw)
            m.update(episode=ep, steps=steps, train_return=float(np.mean(ep_returns[-a.eval_every:])), eps=agent.epsilon(ep))
            log.append(m)
            print(json.dumps(m))
            (out / "log.json").write_text(json.dumps(log, indent=1))
            torch.save(agent.q.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
