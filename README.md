# About Deep Reinforcement Learning
强化学习和深度学习的结合产生了一系列重要的算法, 本项目将着重参考相关paper并尽可能实现相关算法.

*DQNs on CartPole-v0*:

<p float="left">
    <img src="DQN/images/DQN.png" width="280"/>
    <img src="DQN/images/DDQN.png" width="280"/>
    <img src="DQN/images/DuelingDQN.png" width="280"/>
</p>

*PGs on MountainCar-v0*:

<p float="left">
    <img src="PolicyGradient/images/reinforce-mountaincar.gif" width="280"/>
    <img src="PolicyGradient/images/Reinforce%20MountainCar-v0.png" width="280"/>
    <img src="PolicyGradient/images/Reinforce%20with%20Baseline%20MountainCar-v0.png" width="280"/>
</p>

*Actor-Critics on MountainCar-v0*:

<p float="left">
    <img src="ActorCritic/imgs/Actor-Critic.png" width="280"/>
</p>

## 1.算法列表
1. [DQN系列(Naive DQN, Double DQN, Dueling DQN etc.)][1]
    - [Naive DQN][2]
    - [Double DQN][3]
    - [Dueling DQN][4]
    
2. [Policy Gradient系列(Reinforce, Vanilla PG, TRPO, PPO etc.)][8]
    - [REINFORCE][10]
    - [REINFORCE with Baseline][12]

3. [Actor-Critic系列][13]
    - [Actor-Critic][14]

[1]: DQN
[2]: DQN/NaiveDQN.py
[3]: DQN/DoubleDQN.py
[4]: DQN/DuelingDQN.py
[5]: DQN/images/DQN.png
[6]: DQN/images/DDQN.png
[7]: DQN/images/DuelingDQN.png
[8]: PolicyGradient
[9]: PolicyGradient/images/Reinforce%20MountainCar-v0.png
[10]: PolicyGradient/REINFORCE/REINFORCE.py
[11]: PolicyGradient/images/reinforce-mountaincar.gif
[12]: PolicyGradient/REINFORCE/REINFORCE_BaseLine.py
[13]: ActorCritic
[14]: ActorCritic/Actor_Critic.py
