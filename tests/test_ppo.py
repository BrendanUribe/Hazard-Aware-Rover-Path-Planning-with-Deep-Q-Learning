import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rover.env import RoverEnv
from rover.ppo import PPOAgent, PPOConfig, compute_gae


def test_gae_matches_lecture_example():
    # Lecture 18: (delta0, delta1) = (1, 2), gamma = 1, lambda = 0.8 -> A = (2.6, 2.0); targets (5.6, 4.0)
    rewards = np.array([[2.0], [4.0]]); values = np.array([[3.0], [2.0]])
    dones = np.array([[0.0], [1.0]])
    adv = compute_gae(rewards, values, np.zeros(1), dones, 1.0, 0.8)
    assert np.allclose(adv[:, 0], [2.6, 2.0])
    assert np.allclose((adv + values)[:, 0], [5.6, 4.0])


def test_gae_cuts_trace_at_episode_boundary():
    rewards = np.array([[1.0], [1.0]]); values = np.zeros((2, 1)); dones = np.array([[1.0], [0.0]])
    adv = compute_gae(rewards, values, np.array([10.0]), dones, 0.9, 1.0)
    assert adv[0, 0] == pytest.approx(1.0)  # nothing leaks across the boundary


def test_ppo_update_runs_and_changes_policy():
    env = RoverEnv()
    cfg = PPOConfig(batch_size=32, n_epochs=2, target_kl=None)
    agent = PPOAgent(env.obs_dim, cfg)
    obs, _ = env.reset(seed=0)
    O, A, L, V, R = [], [], [], [], []
    for _ in range(64):
        a, lp, v = agent.sample(obs[None])
        o2, r, term, trunc, _ = env.step(int(a[0]))
        O.append(obs); A.append(a[0]); L.append(lp[0]); V.append(v[0]); R.append(r)
        obs = env.reset()[0] if term or trunc else o2
    adv = np.array(R, np.float32); ret = adv + np.array(V, np.float32)
    before = [p.clone() for p in agent.ac.actor.parameters()]
    stats = agent.update(np.stack(O), np.array(A), np.array(L, np.float32), adv, ret)
    assert set(stats) >= {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.ac.actor.parameters()))
    assert agent.act(obs) in range(4)
