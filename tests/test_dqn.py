import pytest

torch = pytest.importorskip("torch")
from rover.dqn import DQNAgent, DQNConfig
from rover.env import RoverEnv


def test_double_vs_vanilla_update_runs():
    env = RoverEnv()
    for double in (True, False):
        agent = DQNAgent(env.obs_dim, DQNConfig(double=double, warmup=64, batch_size=32))
        obs, _ = env.reset(seed=0)
        for _ in range(100):
            a = agent.act(obs, 1.0)
            o2, r, term, trunc, _ = env.step(a)
            agent.buffer.push(obs, a, r, o2, float(term))
            loss = agent.update()
            obs = env.reset()[0] if term or trunc else o2
        assert loss is not None
