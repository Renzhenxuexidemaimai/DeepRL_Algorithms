# GAIL

This GAIL implementation is highly correlated to [PPO](../PPO) algorithm:
- Expert trajectories generated according to **PPO** pre-trained model;
- GAIL learn policy utilizing **PPO** algorithm.

 
## 1. Usage

1. generate expert trajectories by [expert_trajectory_collector.py](expert_trajecotry_collector.py) (You should pre-train a model by specific **RL** algorithm);
2. write custom config file for gail, a template is provided in [config/config_bipedalwalker-v3.yml](config/config_bipedalwalker-v3.yml);
3. train GAIL from [main.py](main.py).


## 2. Performance

