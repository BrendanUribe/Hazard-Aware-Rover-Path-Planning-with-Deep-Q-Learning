"""Hazardous lunar terrain Gymnasium environment (see README for the MDP)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rover.mapgen import BOULDER, CHARGER, CRATER, GOAL, GRAVEL, N_TERRAIN, NORMAL, START, generate_map

ACTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # N, S, E, W as (dx, dy) on grid[x, y]
PERPENDICULAR = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


class RoverEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, fixed_map: np.ndarray | None = None, size: int = 10, p_slip: float = 0.30,
                 b0: int = 15, b_max: int = 20, max_steps: int = 100, map_kwargs: dict | None = None):
        self.n, self.p_slip, self.b0, self.b_max, self.max_steps = size, p_slip, b0, b_max, max_steps
        self.fixed_map = None if fixed_map is None else np.asarray(fixed_map, dtype=np.int8)
        self.map_kwargs = dict(map_kwargs or {}, n=size, b0=b0, b_max=b_max)
        self.action_space = spaces.Discrete(4)
        # flat: terrain one-hot (6*n*n) + agent one-hot (n*n) + battery/b_max
        self.obs_dim = (N_TERRAIN + 1) * size * size + 1
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)

    def _obs(self) -> np.ndarray:
        terrain = np.eye(N_TERRAIN, dtype=np.float32)[self.grid].transpose(2, 0, 1)
        agent = np.zeros((self.n, self.n), np.float32)
        agent[self.pos] = 1.0
        return np.concatenate([terrain.ravel(), agent.ravel(), [self.battery / self.b_max]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self.fixed_map.copy() if self.fixed_map is not None else generate_map(self.np_random, **self.map_kwargs)
        self.pos, self.battery, self.t = START, self.b0, 0
        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        a, cost, reward, slipped, outcome = int(action), 1, -1.0, False, None
        dx, dy = ACTIONS[a]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        inb = 0 <= nx < self.n and 0 <= ny < self.n
        if inb and self.grid[nx, ny] == GRAVEL and self.np_random.random() < self.p_slip:
            slipped, cost = True, 3
            dx, dy = ACTIONS[PERPENDICULAR[a][self.np_random.integers(2)]]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy  # perpendicular cell
            inb = 0 <= nx < self.n and 0 <= ny < self.n
        if not inb:
            nx, ny = self.pos  # wall: stay put
        elif self.grid[nx, ny] == BOULDER:
            nx, ny = self.pos
            reward += -15.0
        self.pos = (nx, ny)
        self.battery = max(self.battery - cost, 0)
        cell = self.grid[self.pos]
        terminated = False
        if cell == GOAL:
            reward, terminated, outcome = reward + 100.0, True, "success"
        elif cell == CRATER:
            reward, terminated, outcome = reward - 100.0, True, "crater"
        else:
            if cell == CHARGER:
                self.battery = self.b_max
            if self.battery == 0:
                reward, terminated, outcome = reward - 100.0, True, "battery"
        truncated = not terminated and self.t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"outcome": outcome or ("timeout" if truncated else None), "slipped": slipped}

    def render(self):
        sym = ".~#XCG"
        rows = []
        for x in range(self.n):
            rows.append("".join("R" if (x, y) == self.pos else sym[self.grid[x, y]] for y in range(self.n)))
        return "\n".join(rows) + f"\nbattery={self.battery}"
