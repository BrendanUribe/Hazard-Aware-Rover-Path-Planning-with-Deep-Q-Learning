| scenario | random | PPO default | PPO best | Double DQN default | Double DQN best | Q-learning default | Q-learning best |
|---|---|---|---|---|---|---|---|
| s4_p0.3 | 0.16 | 0.99 | 0.99 | 0.86 | 0.94 | 1.00 | 1.00 |
| s5_p0.3 | 0.04 | 0.59 | 0.99 | 0.72 | 0.94 | 1.00 | 1.00 |
| s6_p0.3 | 0.00 | 0.00 | 0.93 | 0.14 | 0.48 | 1.00 | 1.00 |
| s8_p0.3 | 0.00 | 0.00 | 0.05 | 0.00 | 0.03 | 1.00 | 1.00 |
| s10_p0.3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

Success rate = mean over seeds of the last-3-evaluation average. Q-learning is trained and tested on one fixed map (an easier problem than PPO/DQN, which are tested on unseen random maps).

Best configuration per cell (parameters changed from default):

- s4_p0.3 PPO: `ppo_000` {}
- s4_p0.3 Double DQN: `dqn_006` {"gamma": 0.95, "batch_size": 32, "target_sync": 200, "eps_end": 0.01, "hidden": 64, "double": false}
- s4_p0.3 Q-learning (fixed map): `qlearn_001` {"alpha": 0.18, "eps_decay_frac": 0.8, "q_init": 100.0}
- s5_p0.3 PPO: `ppo_001` {"lr": 0.0007, "gamma": 0.995, "gae_lambda": 0.9, "ent_coef": 0.1, "ent_decay": true, "n_envs": 16, "target_kl": 0.05, "reward_scale": 0.01}
- s5_p0.3 Double DQN: `dqn_015` {"lr": 0.00012, "gamma": 0.95, "batch_size": 32, "warmup": 5000, "target_sync": 1000, "grad_clip": 1.0, "hidden": 128}
- s5_p0.3 Q-learning (fixed map): `qlearn_001` {"alpha": 0.18, "eps_decay_frac": 0.8, "q_init": 100.0}
- s6_p0.3 PPO: `ppo_010` {"lr": 0.00083, "gamma": 0.995, "gae_lambda": 0.9, "ent_coef": 0.1, "n_envs": 16, "n_steps": 256, "batch_size": 512, "target_kl": 0.0, "reward_scale": 0.01, "hidden": 64}
- s6_p0.3 Double DQN: `dqn_019` {"lr": 0.00014, "gamma": 0.999, "buffer_size": 50000, "warmup": 5000, "target_sync": 2000, "eps_end": 0.1, "eps_decay_frac": 0.5, "double": false}
- s6_p0.3 Q-learning (fixed map): `qlearn_001` {"alpha": 0.18, "eps_decay_frac": 0.8, "q_init": 100.0}
- s8_p0.3 PPO: `ppo_017` {"lr": 0.0004, "gamma": 0.995, "gae_lambda": 0.98, "clip": 0.1, "ent_coef": 0.0, "ent_decay": true, "n_envs": 32, "batch_size": 512, "epochs": 8, "reward_scale": 0.1, "hidden": 128}
- s8_p0.3 Double DQN: `dqn_001` {"lr": 0.00029, "gamma": 0.999, "buffer_size": 50000, "warmup": 500, "eps_end": 0.1, "grad_clip": 1.0, "hidden": 64}
- s8_p0.3 Q-learning (fixed map): `qlearn_001` {"alpha": 0.18, "eps_decay_frac": 0.8, "q_init": 100.0}
- s10_p0.3 PPO: `ppo_003` {"lr": 0.00036, "gamma": 0.995, "ent_coef": 0.1, "n_envs": 32, "epochs": 8, "reward_scale": 0.01}
- s10_p0.3 Double DQN: `dqn_019` {"lr": 0.00014, "gamma": 0.999, "buffer_size": 50000, "warmup": 5000, "target_sync": 2000, "eps_end": 0.1, "eps_decay_frac": 0.5, "double": false}
- s10_p0.3 Q-learning (fixed map): `qlearn_001` {"alpha": 0.18, "eps_decay_frac": 0.8, "q_init": 100.0}
