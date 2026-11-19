"""Train PPO on randomly generated maps. Usage: python scripts/train_ppo.py --iterations 500 --seed 0

One iteration = collect n_envs * n_steps transitions, compute GAE, run n_epochs of minibatch updates.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from rover.env import RoverEnv
from rover.evaluate import evaluate
from rover.ppo import PPOAgent, PPOConfig, compute_gae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-slip", type=float, default=0.30)
    p.add_argument("--b0", type=int, default=15)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-decay", action="store_true", help="linearly decay the entropy bonus to 0 over training")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.02, help="<=0 disables KL early stopping")
    p.add_argument("--reward-scale", type=float, default=0.05,
                   help="multiply rewards by this for learning only (keeps relative penalties; logs/eval stay in env units)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--eval-maps", type=int, default=5, help="unseen maps per evaluation (x20 episodes each)")
    p.add_argument("--eval-every", type=int, default=25, help="iterations between evaluations")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    env_kw = dict(p_slip=a.p_slip, b0=a.b0, size=a.size)
    cfg = PPOConfig(n_envs=a.n_envs, n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.epochs, lr=a.lr,
                    clip_eps=a.clip, gamma=a.gamma, gae_lambda=a.gae_lambda, ent_coef=a.ent_coef,
                    hidden=a.hidden, target_kl=a.target_kl if a.target_kl > 0 else None)
    envs = [RoverEnv(**env_kw) for _ in range(cfg.n_envs)]
    agent = PPOAgent(envs[0].obs_dim, cfg)
    out = Path(a.out) / f"ppo_seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))

    H, E = cfg.n_steps, cfg.n_envs
    obs = np.stack([e.reset(seed=a.seed * 1000 + i)[0] for i, e in enumerate(envs)])
    ep_ret, ep_returns, log, steps_done = np.zeros(E), [], [], 0

    for it in range(1, a.iterations + 1):
        if a.ent_decay:
            agent.cfg.ent_coef = a.ent_coef * (1 - (it - 1) / a.iterations)
        B_obs = np.zeros((H, E) + obs.shape[1:], np.float32)
        B_act = np.zeros((H, E), np.int64)
        B_logp, B_val, B_rew, B_done = (np.zeros((H, E), np.float32) for _ in range(4))

        for t in range(H):  # --- 1. collect rollout with pi_old
            act, logp, val = agent.sample(obs)
            B_obs[t], B_act[t], B_logp[t], B_val[t] = obs, act, logp, val
            for i, e in enumerate(envs):
                o2, r, term, trunc, _ = e.step(int(act[i]))
                ep_ret[i] += r
                r_learn = r * a.reward_scale
                if trunc:  # time limit: bootstrap from the final observation, then cut the trace
                    r_learn += cfg.gamma * float(agent.values(o2[None])[0])
                B_rew[t, i], B_done[t, i] = r_learn, float(term or trunc)
                if term or trunc:
                    ep_returns.append(ep_ret[i]); ep_ret[i] = 0.0
                    o2, _ = e.reset()  # next episode starts on a fresh random map
                obs[i] = o2
            steps_done += E

        # --- 2. advantages (raw) and critic targets R = A + V_old
        adv = compute_gae(B_rew, B_val, agent.values(obs), B_done, cfg.gamma, cfg.gae_lambda)
        ret = adv + B_val

        # --- 3-4. minibatch optimisation against the fixed rollout reference
        stats = agent.update(B_obs.reshape(H * E, -1), B_act.reshape(-1), B_logp.reshape(-1),
                             adv.reshape(-1), ret.reshape(-1))

        if it % a.eval_every == 0 or it == a.iterations:
            m = evaluate(agent, n_maps=a.eval_maps, **env_kw)
            m.update(iteration=it, steps=steps_done, train_return=float(np.mean(ep_returns[-100:])) if ep_returns else 0.0,
                     **{k: round(v, 5) for k, v in stats.items()})
            log.append(m)
            print(json.dumps(m))
            (out / "log.json").write_text(json.dumps(log, indent=1))
            torch.save(agent.ac.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
