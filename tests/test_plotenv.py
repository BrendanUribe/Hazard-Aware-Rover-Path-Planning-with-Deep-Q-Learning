import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rover.mapgen import (
    BOULDER,
    CHARGER,
    CRATER,
    GOAL,
    GRAVEL,
    NORMAL,
    START,
)

def plot_map(env, trajectory=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        NORMAL: "#B8A98A",   # lunar dust
        GRAVEL: "#C9B27C",   # loose gravel
        BOULDER: "#4A4642",  # dark rock
        CRATER: "#211D1B",   # crater
        CHARGER: "#4FA3A5",  # cyan
        GOAL: "#D99A32",      # gold/orange
        START: "#8F8170",    # gray-brown
    }
    terrain_colors = ["#B8A98A","#C9B27C","#4A4642","#211D1B","#4FA3A5","#D99A32",]
    
    labels = {
        BOULDER: "B",
        CRATER: "C",
        CHARGER: "CH",
        GOAL: "G",
        START: "S",
    }
    n = env.n

    for x in range(n):
        for y in range(n):
            terrain = int(env.grid[x, y])
            color = colors.get(terrain, "#B8A98A")
            ax.add_patch(
                Rectangle((y - 1, x - 1),1,1,facecolor=color,edgecolor="white",linewidth=1.5,)
            )
            if terrain in labels:
                ax.text(
                    y,
                    x,
                    labels[terrain],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
    # Plot the rover's current position
    rover_x, rover_y = env.pos
    ax.scatter(
        rover_y,
        rover_x,
        s=250,
        c="#6EC6FF",
        edgecolors="black",
        linewidths=1.5,
        label="Rover",
        zorder=3,
    )
    # Plot the rover trajectory if one is provided
    if trajectory is not None and len(trajectory) > 1:
        path_x = [position[1] for position in trajectory]
        path_y = [position[0] for position in trajectory]
        ax.plot(
            path_x,
            path_y,
            color="#E8E8E8",
            linewidth=2.5,
            marker="o",
            markersize=3,
            label="Trajectory",
            zorder=2,
        )
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"Lunar Rover Map | Battery: {env.battery}")
    ax.grid(True, color="white", linewidth=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    return ax

environment_figure = plot_map()
plt.show()