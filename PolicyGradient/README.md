# Policy Gradient 系列算法

基于策略的梯度算法，包括Monte-Carlo Policy Gradient->REINFORCE, 以及DDPG, TRPO, PPO等算法.

在理解算法的过程中，有一个难点是对于策略函数$\pi(a | s, \theta)$的梯度优化计算原理，因为策略函数是一个概率分布，

其梯度计算依赖于对分布的随机采样,pytorch等计算图框架中封装了这类的算法，其梯度计算原理见[Gradient Estimation Using Stochastic Computation Graphs][1]

## 算法细节及论文
- *REINFORCE* 

这算是PG最入门的算法了,基于轨迹更新.当然也有time step更新的版本, 可以理解为将算法中第二步中对于i的求和式进行分解.

![2]



## 实践效果
在gym的经典游戏MountainCar-v0中的表现：

<p float="left">
    <img src="images/reinforce-mountaincar.gif" width="350"/>
    <img src="images/Reinforce%20MountainCar-v0.png" width="350"/>
</p>


- *REINFORCE with Baseline*
<p float="left">
    <img src="images/Reinforce%20with%20Baseline%20MountainCar-v0.png" width="350"/>
</p>

可以看到，baseline 版本的reward曲线更加平稳，这就是baseline的重要作用：降低Value Net估计值的方差。

[1]: https://arxiv.org/abs/1506.05254
[2]: images/REINFORCE%20alg.png
[3]: images/reinforce-mountaincar.gif
[4]: images/Reinforce%20MountainCar-v0.png
[5]: images/Reinforce%20with%20Baseline%20MountainCar-v0.png