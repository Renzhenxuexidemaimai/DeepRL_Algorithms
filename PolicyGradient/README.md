# Policy Gradient 系列算法

基于策略的梯度算法，包括Vanilla Policy Gradient(REINFORCE), 以及DDPG, TRPO, PPO等算法.

在理解算法的过程中，有一个难点是对于策略函数$\pi(a | s, \theta)$的梯度优化计算原理，因为策略函数是一个概率分布，

其梯度计算依赖于对分布的随机采样,pytorch等计算图框架中封装了这类的算法，其梯度计算原理见[Gradient Estimation Using Stochastic Computation Graphs][1]

PG算法主要有两大类:
- `On-policy`: REINFORCE(VPG), TRPO, PPO (on-policy 算法基于当前策略进行采样, 因此sample efficiency 较低);
- `Off-policy`: DDPG, TD3, SAC (off-policy 算法可以充分利用old data, 因此sample efficiency 较高, 相应地它无法确保策略的表现是否足够好).

两类算法的发展主要是针对各自的问题进行相应地优化, 但它们本质上都是对策略(Policy)的优化。

## 算法细节及论文
### 1. REINFORCE(VPG)

这算是PG最基本门的算法了,基于轨迹更新.当然也有基于time step更新的版本, 可以理解为将算法中第二步中对于i的求和式进行分解.
其算法流程如下:
![2]

#### 实践效果
在gym的经典游戏MountainCar-v0中的表现：

<p float="left">
    <img src="images/reinforce-mountaincar.gif" width="350"/>
    <img src="images/Reinforce%20MountainCar-v0.png" width="350"/>
</p>


#### REINFORCE with Baseline
可以证明，引入Baseline的策略梯度仍是无偏的，同时它能够有效降低方差。
这里的Baseline为之后的 GAE(Generalized Advantage Estimation) 提供了方向, [GAE论文][9]详细阐述了用Advantage更新的变种和优点。
之后的PG算法都将基于GAE。

<p float="left">
    <img src="images/Reinforce%20with%20Baseline%20MountainCar-v0.png" width="350"/>
</p>

可以看到，baseline 版本的reward曲线更加平稳，这就是baseline的重要作用：降低Value Net估计值的方差。


### 2. PPO (Proximal Policy Optimization)

这里把PPO放在TRPO之前，原因是其原理和形式简单，实现上也相对简单，其性能接近TRPO，实践效果较好。在PG算法中，PPO目前已经成为主流算法。

PPO能胜任较为复杂的控制任务，在[Mujoco][8]环境中已经能获得不错的效果。

PPO是对TRPO的变种，其优化目标是 Surrogate Loss:
![7]
其中 $\epsilon$ 是参数，一般取`0.1 - 0.2`，该目标确保更新策略参数靠近原策略参数，这也就是Proximal的来源。

其算法流程如下:
![6]

#### 实践效果

在gym的Box2d环境下[`BipedalWalker-v2`][16]中的表现(可能算法还有细节上的BUG, 收敛速度较慢, 参数方面我参照了比较官方的参数）：

<p float="left">
    <img src="images/ppo-bipedalWalker-v2.gif" width="350"/>
    <img src="images/PPO%20BipedalWalker-v2.png" width="350"/>
</p>

一般用`Mujoco`测试PPO, 但本机上没法装`Mujoco`，等之后在更新吧 :(

实现了基于[`mini_batch`][11]和[`double policy`][12]两个版本，但`double policy`的版本似乎有BUG，没法得到策略效果，
目前还没没查出来 :(

#### 训练

执行[main.py][13], 使用[click][14]解析命令行参数, 因此也可以使用命令行配置参数。

#### 测试

加载已经训练好的模型,执行[test.py][14], 同样也可以使用命令行, 可以测试模型性能。

[1]: https://arxiv.org/abs/1506.05254
[2]: images/REINFORCE%20alg.png
[3]: images/reinforce-mountaincar.gif
[4]: images/Reinforce%20MountainCar-v0.png
[5]: images/Reinforce%20with%20Baseline%20MountainCar-v0.png
[6]: images/PPO%20alg.png
[7]: images/PPO%20surrogate%20loss.png
[8]: https://gym.openai.com/envs/#mujoco
[9]: https://arxiv.org/abs/1506.02438
[10]: images/PPO%20BipedalWalker-v2.png
[11]: PPO/ppo_mini_batch.py
[12]: PPO/ppo.py
[13]: PPO/main.py
[14]: https://click.palletsprojects.com/en/7.x/
[15]: PPO/test.py
[16]: https://gym.openai.com/envs/BipedalWalker-v2/