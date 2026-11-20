import numpy as np

from rover.env import RoverEnv
from rover.qlearn import QConfig, QTableAgent, state_of


def test_q_update_matches_bellman_target():
    agent = QTableAgent(4, 20, QConfig(alpha=0.5, gamma=0.9))
    s, s2 = (0, 0, 10), (0, 1, 9)
    agent.Q[s2] = [1.0, 3.0, 2.0, 0.0]
    agent.update(s, 0, -1.0, s2, terminal=False)      # target = -1 + 0.9*3 = 1.7 ; Q = 0 + 0.5*1.7
    assert np.isclose(agent.Q[s][0], 0.85)
    agent.update(s, 1, 100.0, s2, terminal=True)      # terminal: no bootstrap
    assert np.isclose(agent.Q[s][1], 50.0)


def test_sarsa_uses_next_action_value():
    agent = QTableAgent(4, 20, QConfig(alpha=1.0, gamma=1.0, sarsa=True))
    s2 = (1, 1, 5)
    agent.Q[s2] = [1.0, 3.0, 2.0, 0.0]
    agent.update((0, 0, 6), 0, 0.0, s2, terminal=False, a2=2)
    assert agent.Q[(0, 0, 6)][0] == 2.0


def test_solves_tiny_fixed_map():
    probe = RoverEnv(size=4, p_slip=0.0, b0=20)
    probe.reset(seed=10_000)
    env = RoverEnv(fixed_map=probe.grid.copy(), size=4, p_slip=0.0, b0=20)
    episodes = 3000
    agent = QTableAgent(4, env.b_max, QConfig(alpha=0.3, eps_decay_frac=0.5), seed=0)
    for ep in range(1, episodes + 1):
        env.reset(seed=ep)
        s, eps, done = state_of(env), agent.epsilon(ep, episodes), False
        while not done:
            a = agent.act(s, eps)
            _, r, term, trunc, _ = env.step(a)
            s2 = state_of(env)
            agent.update(s, a, r, s2, term)
            s, done = s2, term or trunc
    env.reset(seed=1)
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(agent.greedy(state_of(env)))
        done = term or trunc
    assert info["outcome"] == "success"
