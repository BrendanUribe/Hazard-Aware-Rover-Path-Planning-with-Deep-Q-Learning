"""Greedy evaluation on unseen maps."""
import numpy as np

from rover.env import RoverEnv


def evaluate(agent, n_maps: int = 5, episodes_per_map: int = 20, seed: int = 10_000, **env_kwargs) -> dict:
    outcomes, steps, returns = [], [], []
    for m in range(n_maps):
        env = RoverEnv(**env_kwargs)
        obs, _ = env.reset(seed=seed + m)
        fixed = env.grid.copy()
        env = RoverEnv(fixed_map=fixed, **env_kwargs)  # same map, stochastic slip across episodes
        for e in range(episodes_per_map):
            obs, _ = env.reset(seed=seed + 1000 * m + e)
            done, ret, t = False, 0.0, 0
            while not done:
                obs, r, term, trunc, info = env.step(agent.act(obs, eps=0.0))
                ret, t, done = ret + r, t + 1, term or trunc
            outcomes.append(info["outcome"]); steps.append(t); returns.append(ret)
    n = len(outcomes)
    return {"success": outcomes.count("success") / n, "crater": outcomes.count("crater") / n,
            "battery": outcomes.count("battery") / n, "timeout": outcomes.count("timeout") / n,
            "mean_steps": float(np.mean(steps)), "mean_return": float(np.mean(returns))}
