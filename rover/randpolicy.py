"""Random Poicy - serve as baseline. No training."""
# import numpy as np
# import torch
# import random
import matplotlib.pyplot as plt
from rover.plotenv import plot_map
from rover.env import RoverEnv

def run_random_episode(env, seed=None):
    # initialize map and parameters
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_return = 0.0
    trajectory = [env.pos]

    while not (truncated or terminated):
        action = env.action_space.sample() # chooses random action
        obs, rwd, terminated, truncated, info = env.step(action)
        total_return += rwd
        trajectory.append(env.pos)

    return {"outcome": info["outcome"],"return": total_return,"steps": len(trajectory) - 1,"trajectory": trajectory}

def evaluate_rand_policy(n_episodes):
    env = RoverEnv() # create map
    results = []

    for episode in range(n_episodes):
        result = run_random_episode(env,seed=0) # change seed here !
        results.append(result)
        """ uncomment to plot each episode """
        plot_map(env,trajectory=result["trajectory"])
        plt.title(f"Random Policy - Episode {episode + 1}, Steps: {len(result['trajectory'])}")
        plt.show()

    success_count = sum(result["outcome"] == "success" for result in results)
    crater_count = sum(result["outcome"] == "crater" for result in results)
    battery_failures = sum(result["outcome"] == "battery" for result in results)  
    
    print("Episodes:", n_episodes)
    print("Success rate:", success_count / n_episodes)
    print("Crater collision rate:", crater_count / n_episodes)
    print("Battery depletion rate:", battery_failures / n_episodes)
    print("Average return:", sum(r["return"] for r in results) / n_episodes)
    print("Average steps:", sum(r["steps"] for r in results) / n_episodes)

    return results

if __name__ == "__main__":
    evaluate_rand_policy(n_episodes=20)



