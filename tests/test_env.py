import numpy as np

from rover import BOULDER, CHARGER, CRATER, GOAL, GRAVEL, RoverEnv, generate_map


def flat(n=10):
    g = np.zeros((n, n), np.int8); g[n - 1, n - 1] = GOAL; return g


def test_generated_maps_valid():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = generate_map(rng)
        assert g[0, 0] == 0 and g[9, 9] == GOAL and (g == CHARGER).sum() == 1


def test_obs_shape_and_reset():
    env = RoverEnv(); obs, _ = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,) and env.observation_space.contains(obs)


def test_step_cost_and_wall():
    env = RoverEnv(fixed_map=flat()); env.reset()
    _, r, term, _, _ = env.step(0)  # North off-grid
    assert r == -1 and env.pos == (0, 0) and env.battery == 14 and not term


def test_boulder_blocks():
    g = flat(); g[1, 0] = BOULDER
    env = RoverEnv(fixed_map=g); env.reset()
    _, r, *_ = env.step(1)
    assert r == -16 and env.pos == (0, 0)


def test_crater_terminal():
    g = flat(); g[1, 0] = CRATER
    env = RoverEnv(fixed_map=g); env.reset()
    _, r, term, _, info = env.step(1)
    assert term and r == -101 and info["outcome"] == "crater"


def test_gravel_slip_costs_3():
    g = flat(); g[1, 0] = GRAVEL
    env = RoverEnv(fixed_map=g, p_slip=1.0); env.reset()
    _, _, _, _, info = env.step(1)
    assert info["slipped"] and env.battery == 12


def test_charger_and_battery_depletion():
    g = flat(); g[0, 1] = CHARGER
    env = RoverEnv(fixed_map=g); env.reset()
    env.step(2); assert env.battery == 20
    env2 = RoverEnv(fixed_map=flat(), b0=2); env2.reset()
    env2.step(0); _, r, term, _, info = env2.step(0)
    assert term and info["outcome"] == "battery"


def test_goal():
    g = flat(3); env = RoverEnv(fixed_map=g, size=3); env.reset()
    for a in (1, 1, 2, 2):
        _, r, term, _, info = env.step(a)
    assert term and info["outcome"] == "success"
