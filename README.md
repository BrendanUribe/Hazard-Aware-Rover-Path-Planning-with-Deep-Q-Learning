# Hazard-Aware Rover Path Planning with Deep Q-Learning

**AE 696 – Reinforcement Learning with AI Applications in Engineering** · San Diego State University · Fall 2026
**Instructor:** Dr. Jun Chen
**Team:** Brendan Uribe · David Shaker

> 🚧 **Status:** In progress. This README reflects the project proposal; results will be added as experiments are completed.

---

## Overview

A lunar rover must drive from its deployment site to a science target across terrain containing **craters**, **boulders**, and **loose gravel**, all with a limited battery budget.

- **Craters** end the mission.
- **Boulders** block motion.
- **Gravel** causes stochastic wheel slip, which also drains more energy.
- A **charging station** can restore the battery at the cost of extra travel distance.

The rover gets a preloaded terrain map before each mission, but the terrain layout changes from mission to mission. Because slip is random and each action's outcome depends on terrain, position, and remaining battery, this is a **sequential decision problem under uncertainty**.

**Goal:** Learn a policy that reaches the target with a high success rate while minimizing distance, avoiding hazards, and managing battery, then find out whether a policy trained on randomly generated maps **generalizes to unseen maps**.

## Contributions

1. **Custom Gymnasium environment:** hazardous lunar terrain, stochastic slip, terrain-dependent energy use, battery limits, and a charging station.
2. **Fixed-map baselines:** tabular Q-learning, SARSA, and value iteration to verify the environment and set reference performance.
3. **DQN controller:** takes the full terrain map, rover position, and battery level as input and trains across randomly generated maps.
4. **Robustness study:** evaluates the learned policy on unseen maps with varying slip probability and initial battery charge.

---

## MDP Formulation

### State
$s = (M, x, y, b)$

| Component | Description |
|---|---|
| $M$ | Episode-specific 10 × 10 terrain map. The generator guarantees a valid start/goal and a feasible route. |
| $(x, y)$ | Rover cell location, $x, y \in \{0, \dots, 9\}$ |
| $b$ | Remaining battery, $b \in \{0, 1, \dots, 20\}$, $b_{max} = 20$ |

On a fixed map, the position–battery space has 10 × 10 × 21 = **2,100 states**. With the map randomized each episode, the full state space is too large for a tabular method, which is why DQN is used.

### Actions
$\mathcal{A} = \{\text{North}, \text{South}, \text{East}, \text{West}\}$

### Rewards

| Event | Reward |
|---|---|
| Each step | −1 |
| Reach science target (terminal) | +100 |
| Enter a crater (terminal) | −100 |
| Battery depleted before goal (terminal) | −100 |
| Drive into a boulder | −15 |

### Dynamics

| Parameter | Value |
|---|---|
| Normal terrain | Deterministic moves |
| Gravel slip | Slips to a perpendicular cell with $p_{slip} = 0.30$ |
| Battery cost | −1 per normal move, −3 per slip |
| Charging station | Resets battery to $b_{max}$ |
| Initial state | $(0, 0)$ with $b_0 = 15$ |
| Termination | Success at goal; failure on crater or $b = 0$ |
| Discount factor | $\gamma = 0.99$ |

**Model access:** Q-learning, SARSA, and DQN learn only from sampled transitions. Value iteration is the only method given the full transition model (fixed map only).

---

## Methods

### Primary: Deep Q-Network (DQN)
- ε-greedy exploration
- Input: full terrain map + rover position + battery level
- Output: one Q-value per action (4 total)
- Experience replay and a target network for training stability

### Baselines

| Method | Type | Setting | Purpose |
|---|---|---|---|
| Tabular Q-learning | Off-policy, model-free | Fixed map | Model-free baseline |
| SARSA | On-policy, model-free | Fixed map | On- vs. off-policy comparison under slip and hazards |
| Value iteration | Model-based | Fixed map | Reference policy with full transition model |
| Random policy | — | All | Lower-bound reference |

---

## Evaluation

### Metrics
- **Mission success rate** (primary): % of episodes that reach the goal before hitting a crater, depleting the battery, or timing out
- **Crater collision rate** and **battery depletion rate**, reported separately to tell failure modes apart
- **Average actions per episode** (efficiency)
- **Energy consumption** and **episode return**

All results are reported as mean ± standard deviation across seeds and evaluation maps.

### Protocol

**Fixed-map baseline**
- Q-learning and SARSA trained for 20,000 episodes × 3 seeds ($p_{slip} = 0.30$, $b_0 = 15$)
- Frozen policies evaluated greedily (ε = 0) for 20 episodes
- Value iteration and random policy evaluated under identical conditions

**Randomized-map experiment (main)**
- New terrain map generated at the start of every training episode
- DQN trained for 20,000 episodes × 3 seeds
- Each frozen policy evaluated on 5 unseen maps, 20 episodes per map
- No learning during evaluation

**Robustness sweep**

| Parameter | Values |
|---|---|
| Slip probability $p_{slip}$ | 0.0, 0.15, 0.30 |
| Initial battery $b_0$ | 10, 15, 20 |

### Hypotheses
1. Higher slip probability and lower initial battery reduce mission success.
2. DQN outperforms the random policy and approaches the value iteration reference on unseen maps.
3. Q-learning and SARSA show different safety/efficiency tradeoffs on the fixed map.

### Planned Outputs
- Training/learning curves
- Comparison tables
- Representative trajectory plots
- Robustness plots vs. slip probability and initial battery

---

## Tech Stack

| Tool | Use |
|---|---|
| Python | Core language |
| [Gymnasium](https://gymnasium.farama.org) | Environment API |
| NumPy | Map generation and transition calculations |
| [PyTorch](https://pytorch.org) | DQN implementation and training |
| Matplotlib | Learning curves, comparisons, and policy visualizations |

All experiments run **CPU-only** on personal laptops.

---

## Getting Started

### 1. Install
```bash
git clone https://github.com/BrendanUribe/Hazard-Aware-Rover-Path-Planning-with-Deep-Q-Learning.git
cd Hazard-Aware-Rover-Path-Planning-with-Deep-Q-Learning
py -m pip install -e ".[dev]"        # Windows; use `python -m pip ...` on macOS/Linux
```
Use the same interpreter (`py` or `python`) for every command below, otherwise you may get `No module named 'rover'`. A virtual environment is optional (`python -m venv .venv`).

### 2. Run the tests
```bash
pytest                               # 9 tests: environment rules + a Double DQN update runs
```
These check the environment and that the update runs; they do not show that the agent learns.

### 3. Look at the environment (no training needed)
```bash
py scripts/visualize.py --seed 0 --n 3 --out maps.png
```
Draws random maps with a rover taking random moves. Legend: beige normal, tan gravel (slip), gray boulder, black x crater, green charger, yellow goal, `S` start, blue line the path, red star where the rover ended. The title gives the outcome, steps used and battery left. Change `--seed` for different maps. To print one map as text: `py -c "from rover import RoverEnv; e=RoverEnv(); e.reset(seed=1); print(e.render())"`.

### 4. Train
```bash
py scripts/train_dqn.py --episodes 300 --eval-every 100 --out runs      # ~5 s pipeline check, expect no learning
py scripts/train_dqn.py --episodes 5000 --eval-every 500 --out runs     # first real test
py scripts/train_dqn.py --episodes 20000 --eval-every 1000 --seed 0     # full protocol; repeat with --seed 1 and 2
py scripts/train_dqn.py --episodes 5000 --seed 0 --vanilla              # vanilla DQN ablation
```
Flags: `--episodes`, `--seed`, `--vanilla` (plain DQN instead of Double DQN), `--p-slip` (default 0.30), `--b0` (initial battery, default 15), `--eval-every`, `--out`.

Each evaluation prints one JSON line (greedy policy, 5 unseen maps x 20 missions): `success` (the main number, want it to rise above 0), `crater`, `battery` (ran out), `timeout`, `mean_steps`, `mean_return`, `train_return`, `eps` (exploration rate).

**Outputs and overwriting:** results go to `<out>/<method>_seed<N>/` (e.g. `runs/ddqn_seed0/`) as `log.json` and `model.pt`. `model.pt` is overwritten at every evaluation, and a new run with the same `--out`, method and seed overwrites the old files. Use a different `--out` (e.g. `runs_5000`) to keep results apart.

### 5. View a trained rover
```bash
py scripts/visualize.py --seed 0 --n 3 --model runs/ddqn_seed0/model.pt --out trained.png
```
Without `--model` the rover moves randomly (the random baseline).

### Notes
- Episodes are short (about 15 steps when the battery runs out), so 300 episodes takes seconds and 20,000 should take minutes on a CPU. A GPU is not needed; the code runs on CPU only.
- If `success` stays at 0 after a few thousand episodes, the agent is not learning this task yet; debug on a single fixed map or an easier setup (more battery) before longer runs.

Layout: `rover/env.py` (Gymnasium env), `rover/mapgen.py` (feasible random maps), `rover/dqn.py` (Double DQN agent), `rover/evaluate.py` (greedy eval on unseen maps), `rover/viz.py` (map drawing), `scripts/` (training, visualization), `tests/`.

**Double DQN:** the target is $y = r + \gamma\, Q_{target}(s', \arg\max_a Q_{online}(s', a))$, which reduces the Q-value overestimation of vanilla DQN. Vanilla DQN is kept behind `--vanilla` for an ablation.

---

## Timeline

| Week | Milestone | Responsibilities |
|---|---|---|
| 10 | Environment setup & baseline testing | **David:** Gymnasium environment (terrain, slip, battery, charger, hazards, goal) · **Brendan:** test spaces, transitions, termination, rewards, map generation |
| 11 | Initial training & fixed-map baselines | **David:** value iteration · **Brendan:** Q-learning, SARSA, start DQN · **Both:** random baseline |
| 12–13 | Randomized-map experiments | **David:** value iteration comparisons, trajectory visualizations · **Brendan:** DQN training, evaluation scripts, metrics · **Both:** robustness runs |
| 14 | Final analysis & project check-in | **Both:** metrics analysis, plots, comparison tables |
| 15 | Presentation & final report | **Both:** reproducibility checks, documentation, report |

### Backup Plan
If DQN training is unstable or doesn't generalize because of the large map-based observation, the fallback removes the **charging station and battery state** from the randomized-map environment. It keeps stochastic slip, crater hazards, random map generation, and DQN path planning. The fixed-map Q-learning, SARSA, and value iteration experiments will be completed either way.

---

## Results

*Coming soon.*

---

## References

1. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018. [Link](http://incompleteideas.net/book/the-book-2nd.html)
2. C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, pp. 279–292, 1992.
3. G. A. Rummery and M. Niranjan, "On-line Q-learning using connectionist systems," Tech. Rep. CUED/F-INFENG/TR 166, Cambridge University, 1994.
4. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
5. M. Towers et al., "Gymnasium: A standard interface for reinforcement learning environments," arXiv:2407.17032, 2024. [Link](https://gymnasium.farama.org) *(FrozenLake served as a conceptual reference for the slip mechanic.)*
6. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," *NeurIPS 32*, 2019. [Link](https://pytorch.org)
7. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.
8. J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.
9. J. Chen, AE 696 lecture notes and DQN laboratory materials, San Diego State University, Fall 2026.
