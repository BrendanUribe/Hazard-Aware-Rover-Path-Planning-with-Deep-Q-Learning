"""Draw random maps with a rover trajectory. Usage:
  python scripts/visualize.py --seed 3                       # random policy
  python scripts/visualize.py --seed 3 --model runs/ddqn_seed0/model.pt   # trained policy
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover.env import RoverEnv
from rover.viz import draw_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=4, help="number of maps")
    p.add_argument("--model", default=None, help="path to model.pt (default: random policy)")
    p.add_argument("--out", default="maps.png")
    a = p.parse_args()

    agent = None
    if a.model:
        import torch
        from rover.dqn import DQNAgent, DQNConfig
        agent = DQNAgent(RoverEnv().obs_dim, DQNConfig())
        agent.q.load_state_dict(torch.load(a.model, map_location="cpu"))

    fig, axes = plt.subplots(1, a.n, figsize=(5 * a.n, 5))
    for i, ax in enumerate(axes if a.n > 1 else [axes]):
        env = RoverEnv()
        obs, _ = env.reset(seed=a.seed + i)
        path, done = [env.pos], False
        while not done:
            act = agent.act(obs, 0.0) if agent else random.randrange(4)
            obs, r, term, trunc, info = env.step(act)
            path.append(env.pos); done = term or trunc
        draw_map(env.grid, path, ax, f"seed {a.seed + i}: {info['outcome']} ({len(path) - 1} steps, battery {env.battery})")
    fig.tight_layout(); fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
