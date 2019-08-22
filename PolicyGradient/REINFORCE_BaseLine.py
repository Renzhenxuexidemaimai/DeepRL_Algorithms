import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

from PolicyGradient.Nets.Advance_Policy_Net import AdvancePolicyNet
from Utils.env_utils import get_env_space


class REINFORCE_Baseline:
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate_policy=0.002,
                 learning_rate_value=0.05,
                 gamma=0.995,
                 eps=torch.finfo(torch.float32).eps,
                 enable_gpu=False
                 ):

        if enable_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.policy = AdvancePolicyNet(num_states, num_actions).to(self.device)
        self.value = AdvancePolicyNet(num_states, 1, is_value=True).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=learning_rate_policy)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=learning_rate_value)

        self.gamma = gamma
        self.eps = eps

        self.values = []  # 记录每个 time step 状态对应的 value 估计
        self.rewards = []  # 记录每个 time step 对应的及时回报 r_t
        self.log_probs = []  # 记录每个 time step 对应的 log_probability
        self.cum_rewards = []  # 记录每个 time step 对应的 累计回报 G_t

    def calc_cumulative_rewards(self):
        R = 0.0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            self.cum_rewards.insert(0, R)

    def choose_action(self, state):
        state = torch.tensor(state).unsqueeze(0).to(self.device).float()
        probs = self.policy(state)

        # 对action进行采样,并计算log probability
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.log_probs.append(log_prob)

        # 计算 state 状态的 base 值函数
        value = self.value(state)
        self.values.append(value)

        return action.item()

    def update_episode(self):
        self.calc_cumulative_rewards()
        assert len(self.cum_rewards) == len(self.values)

        rewards = torch.tensor(self.cum_rewards).unsqueeze(-1).to(self.device)
        values = torch.stack(self.values).squeeze(-1)
        log_probs = torch.stack(self.log_probs)

        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        advances = rewards - values
        # advances = (advances - advances.mean()) / (advances.std() + self.eps)

        # batch 梯度下降更新 base_value 参数
        loss_advance_criterion = torch.nn.MSELoss()

        self.optimizer_value.zero_grad()
        loss_advance = loss_advance_criterion(rewards, values)
        loss_advance.backward(retain_graph=True)
        self.optimizer_value.step()

        # 梯度上升更新策略参数
        self.optimizer_policy.zero_grad()
        loss_policy = -log_probs.mul(advances).mean()
        loss_policy.backward()
        self.optimizer_policy.step()

        # 清空 buffer
        self.rewards.clear()
        self.log_probs.clear()
        self.cum_rewards.clear()
        self.values.clear()


if __name__ == '__main__':
    env_id = 'MountainCar-v0'
    alg_id = 'REINFORCE_Baseline'
    env, num_states, num_actions = get_env_space(env_id)

    agent = REINFORCE_Baseline(num_states, num_actions, enable_gpu=True)
    episodes = 400

    writer = SummaryWriter()

    # 迭代所有episodes进行采样
    for i in range(episodes):
        # 当前episode开始
        state = env.reset()
        episode_reward = 0

        for t in range(8000):
            env.render()
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)

            episode_reward += reward
            agent.rewards.append(reward)

            # 当前episode　结束
            if done:
                writer.add_scalar(alg_id, episode_reward, i)
                print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))
                break

        agent.update_episode()

    env.close()
