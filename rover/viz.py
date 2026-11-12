"""Matplotlib drawing of rover maps and trajectories."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from rover.mapgen import BOULDER, CHARGER, CRATER, GOAL, GRAVEL, NORMAL

COLORS = {NORMAL: "#e8e4d8", GRAVEL: "#c9a66b", BOULDER: "#6b6b6b", CRATER: "#1a1a1a", CHARGER: "#3fae5a", GOAL: "#f2c230"}
LABELS = {NORMAL: "normal", GRAVEL: "gravel (slip)", BOULDER: "boulder", CRATER: "crater", CHARGER: "charger", GOAL: "goal"}


def draw_map(grid, path=None, ax=None, title=None):
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    cmap = ListedColormap([COLORS[i] for i in range(len(COLORS))])
    # grid[x, y]: x is the row (North/South), y is the column (East/West)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    n = grid.shape[0]
    ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=1); ax.tick_params(which="both", length=0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    for x, y in np.argwhere(grid == CRATER):
        ax.text(y, x, "x", ha="center", va="center", color="#ff6b6b", fontsize=14, fontweight="bold")
    ax.text(0, 0, "S", ha="center", va="center", fontweight="bold")
    if path is not None and len(path) > 1:
        p = np.array(path)
        ax.plot(p[:, 1], p[:, 0], "-o", color="#1f77b4", markersize=4, linewidth=2, label="rover path")
        ax.plot(p[-1, 1], p[-1, 0], "*", color="red", markersize=16)
    ax.legend(handles=[Patch(color=COLORS[i], label=LABELS[i]) for i in COLORS], loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=8)
    if title:
        ax.set_title(title)
    return ax
