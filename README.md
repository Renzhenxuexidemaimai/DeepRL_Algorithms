# About Deep Reinforcement Learning

The combination of Reinforcement Learning and Deep Learning produces a series of important algorithms. This project will focus on referring to 
relevant papers and implementing relevant algorithms as far as possible. 

This repo aims to implement Deep Reinforcement Learning algorithms using [Pytorch](https://pytorch.org/) and [Tensorflow 2](https://www.tensorflow.org/).


## 1.Why do this?

- Implementing all of this algorithms from scratch really helps you with your **parameter tuning**; 
- The coding process allows you to **better understand** the **principles** of the algorithm.

## 2.Lists of Algorithms

| No. | Status | Algorithm | Paper |
| --- | ------- | --------- | ----- |
| 1 | :white_check_mark: | DQN [Pytorch](Algorithms/pytorch/DQN) / [Tensorflow](Algorithms/tf2/DQN) | [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) |
| 2 | :white_check_mark: | Double DQN [Pytorch](Algorithms/pytorch/DoubleDQN) / [Tensorflow](Algorithms/tf2/DoubleDQN) | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) |
| 3 | :white_check_mark: | Dueling DQN [Pytorch](Algorithms/pytorch/DuelingDQN) / [Tensorflow](Algorithms/tf2/DuelingDQN)| [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) |
| 4 | :white_check_mark: | REINFORCE [Pytorch](Algorithms/pytorch/REINFORCE) | [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) |
| 5 | :white_check_mark: | VPG(Vanilla Policy Gradient) [Pytorch](Algorithms/pytorch/VPG) | [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) |
| 6 | <ul><li>- [ ] </li></ul> | A3C |  |
| 7 | <ul><li>- [ ] </li></ul> | A2C |  |
| 8 | <ul><li>- [ ] </li></ul> | DPG | [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) |
| 9 | :white_check_mark: | DDPG [Pytorch](Algorithms/pytorch/DDPG) | [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) |
| 10 | <ul><li>- [ ] </li></ul> | D4PG |  |
| 11 | <ul><li>- [ ] </li></ul> | MADDPG |  |
| 12 | :white_check_mark: | TRPO [Pytorch](Algorithms/pytorch/TRPO) | [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) |
| 13 | :white_check_mark: | PPO [Pytorch](Algorithms/pytorch/PPO) | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |
| 14 | <ul><li>- [ ] </li></ul> | ACER |  |
| 15 | <ul><li>- [ ] </li></ul> | ACTKR |  |
| 16 | <ul><li>- [ ] </li></ul> | SAC |  |
| 17 | <ul><li>- [ ] </li></ul> | SAC with Automatically Adjusted Temperature |
| 18 | :white_check_mark: | TD3(Twin Delayed DDPG) [Pytorch](Algorithms/pytorch/TD3) | [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) |
| 19 | <ul><li>- [ ] </li></ul> | SVPG |  |
| 20 | <ul><li>- [ ] </li></ul> | IMPALA |  |

### 3.Project Dependencies

- Python >=3.6  
- Tensorflow >= 2.1.0
- Pytorch >= 1.3.1  
- Seaborn >= 0.10.0  
- Click >= 7.0  

### 4.Run

Each algorithm is implemented in a single package including:
```
main.py --A minimal executable example for algorithm  
[algorithm].py --Main body for algorithm implementation  
test.py --Loading pretrained model and test performance of the algorithm
[algorithm]_step.py --Algorithm update core step 
````
The default `main.py` is a an executable example, the parameters are parsed by [click](https://click.palletsprojects.com/en/7.x/).

You can run algorithm from the  `main.py` or `bash scripts`. 
- You can simply type `python main.py --help` in the algorithm package to view all parameters. 
- The directory [Scripts](Scripts) gives some bash scripts, you can modify them at will.

### 5.Visualization of performance

[Utils/plot_util.py](Utils/plot_util.py) provide a simple plot tool based on `Seaborn` and `Matplotlib`.
All the plots in this project are drawn by this plot util.

#### 5.1 Benchmarks for DQNs

![bench_dqn](Algorithms/images/bench_dqn.png)
 
#### 5.2 Benchmarks for PolicyGradients

![bench_pg](Algorithms/images/bench_pg.png)

