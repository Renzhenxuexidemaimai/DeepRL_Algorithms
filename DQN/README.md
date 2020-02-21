# DQN 系列算法实现

DQN 是 Deep Reinforcement Learning 的入门级算法。

Deep Q-learning Network 基于 `Q-learning` 是 `off-policy` 算法。它使用一个`Q`网络估计 action-value 映射函数，
使用 Experience-Replay 采样，可有效提升 data efficiency。

## 算法细节及论文

### 1.Basic DQN [Playing Atari with Deep Reinforcement Learning][1]
![basic DQN](images/DQN%20with%20Experience%20Replay.png)

### 2.Double DQN [Deep Reinforcement Learning with Double Q-learning][2]

与 Basic DQN 不同，这里使用了两个网络。一个作为在线更新 -> 训练网络(train net)，另一个用于策略评估 -> 目标网络(target policy)。
在更新上，训练网络立即更新,而目标网络的更新存在一个滞后性(freeze)；策略评估中，用训练网络找到最优动作:

$$a_{t + 1} =  \arg \max_{a \in A} \left[ r(s_{t}, a_{t}) + Q(s_{t + 1}, \arg \max_{a \in A} Q^{target} (s_{t + 1}, a_{t + 1}) ) \right]$$
 
基于 `train net` 进行采样，而使用 `target net` 进行评估。

Double DQN 的动作值估计形式如下(论文中说对偶形式等价即交换$\theta_t$和$\theta_t^{'}$):

![Double DQN eval](images/Double%20Q-learning%20eval.png)

*DDQN算法流程如下*:

![DDQN algorithm](images/Double%20DQN%20Algorithm.png)

### 3.Dueling DQN [Dueling Network Architectures for Deep Reinforcement Learning][3]

更改 Basic DQN 的网络结构.
![Dueling DQN Structure](images/Dueling%20DQN%20Network.png)

使用优化技巧使网络平衡地更新到 state value 和 advantage action(state-dependent) value.
![Dueling DQN algorithm](images/Dueling%20DQN%20optimization%20for%20identifiability.png)

## 实践效果
在gym的经典游戏CartPole-v0中的表现：

<p float="left">
    <img src="images/DQN.png" width="300"/>
    <img src="images/DDQN.png" width="300"/>
    <img src="images/DuelingDQN.png" width="300"/>
</p>


[1]: https://arxiv.org/abs/1312.5602
[2]: https://arxiv.org/abs/1509.06461
[3]: https://arxiv.org/abs/1511.06581
