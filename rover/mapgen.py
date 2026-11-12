"""Random terrain generation with a feasibility guarantee."""
from collections import deque

import numpy as np

NORMAL, GRAVEL, BOULDER, CRATER, CHARGER, GOAL = range(6)
N_TERRAIN = 6
START = (0, 0)


def bfs_dist(grid: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    """Shortest 4-connected distance over cells that are neither boulder nor crater (inf if unreachable)."""
    n = grid.shape[0]
    dist = np.full(grid.shape, np.inf)
    dist[src] = 0
    q = deque([src])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and np.isinf(dist[nx, ny]) and grid[nx, ny] not in (BOULDER, CRATER):
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return dist


def is_feasible(grid: np.ndarray, b0: int, b_max: int) -> bool:
    """Optimistic (slip-free) check: goal reachable directly within b0-1 steps, or via the charger."""
    n = grid.shape[0]
    goal = (n - 1, n - 1)
    d_start = bfs_dist(grid, START)
    if d_start[goal] <= b0 - 1:
        return True
    cx, cy = np.argwhere(grid == CHARGER)[0]
    d_chg = bfs_dist(grid, (cx, cy))
    return d_start[cx, cy] <= b0 - 1 and d_chg[goal] <= b_max - 1


def generate_map(rng: np.random.Generator, n: int = 10, p_gravel: float = 0.15, p_boulder: float = 0.08,
                 p_crater: float = 0.06, b0: int = 15, b_max: int = 20, max_tries: int = 1000) -> np.ndarray:
    """Sample an n x n map. Start (0,0) and goal (n-1,n-1) are normal/goal cells; one charger is placed."""
    for _ in range(max_tries):
        grid = rng.choice([NORMAL, GRAVEL, BOULDER, CRATER], size=(n, n),
                          p=[1 - p_gravel - p_boulder - p_crater, p_gravel, p_boulder, p_crater])
        grid[START] = NORMAL
        grid[n - 1, n - 1] = GOAL
        free = [(x, y) for x in range(n) for y in range(n)
                if grid[x, y] in (NORMAL, GRAVEL) and (x, y) != START]
        cx, cy = free[rng.integers(len(free))]
        grid[cx, cy] = CHARGER
        if is_feasible(grid, b0, b_max):
            return grid.astype(np.int8)
    raise RuntimeError("could not generate a feasible map")
