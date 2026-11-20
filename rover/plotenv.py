import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from rover.mapgen import (BOULDER,CHARGER,CRATER,GOAL,GRAVEL,NORMAL,START)


def plot_map(env, trajectory=None, ax=None):
    labels = {
    BOULDER: "X",
    CRATER: "XX",
    CHARGER: "CHARGER",
    GOAL: "GOAL",
    # START: "S",
    }
    
    # Map color palette
    colors = {
    NORMAL:  "#9A9488",    
    GRAVEL:  "#816115",    
    BOULDER: "#393633",    
    CRATER:  "#000000",    
    CHARGER: "#4FA3A5",    
    GOAL:    "#49744C",    
    # START:   "#7D3C3C",    
    }
    start_color       = "#7D3C3C"
    rover_color       = "#6EC6FF"
    trajectory_color  = "#932B2B"

    n = env.n

    # if we want to change map size in the future we can implement here
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    ax.set_xticks(range(n), labels=range(1, n+1))
    ax.set_yticks(range(n), labels=range(1, n+1))

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_facecolor("#F2F2F2")

    # Draw terrain cells
    for x in range(n):
        for y in range(n):
            terrain = int(env.grid[x, y])
            color = colors.get(terrain, "#FE09BD") # prints hot pink if terrain color is not found in colors (for troubleshooting)
            ax.add_patch(Rectangle((y - 0.5, x - 0.5),1,1,facecolor=color,edgecolor="white",linewidth=1.5,))
           
            if terrain in labels:
                ax.text(y, x, labels[terrain], ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    # Plot the random start position
    start_x, start_y = env.start
    ax.add_patch(Rectangle((start_y-0.5,start_x-0.5),1,1,facecolor=start_color,edgecolor="white",linewidth=1.5,))
    ax.text(start_y, start_x, "Start", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    
    # Plot the rover's current position
    rover_x, rover_y = env.pos
    ax.scatter(rover_y,rover_x,s=250,c=rover_color,edgecolors="black",linewidths=1.5,label="Rover",zorder=4)


    # Plot the rover trajectory if provided
    if trajectory is not None and len(trajectory) > 1:
        path_x = [position[1] for position in trajectory]
        path_y = [position[0] for position in trajectory]
        ax.plot(path_x,path_y,color=trajectory_color,linewidth=2.5,marker="o",markersize=3,label="Trajectory",zorder=2)


    ax.set_title(f"Lunar Rover Map | Battery: {env.battery}", color = colors[3], weight = "bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    
    return ax

# output plot for testing
if __name__ == "__main__":
    from rover.env import RoverEnv
    env = RoverEnv()
    env.reset(seed=5)
    environment_figure = plot_map(env)
    plt.show()

