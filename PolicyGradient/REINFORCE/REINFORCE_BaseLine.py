import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from PolicyGradient.Models.Advance_Policy import AdvancePolicy
from Utils.env_utils import get_env_space


class REINFORCE_Baseline:
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=0.005,
                 gamma=0.995,
                 eps=torch.finfo(torch.float32).eps,
                 enable_gpu=False
                 ):

        if enable_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.policy = AdvancePolicy(num_states, num_actions).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=learning_rate)

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
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        probs = self.policy.get_action_probs(state)

        # 对action进行采样,并计算log probability
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.log_probs.append(log_prob)

        # 计算 state 状态的 base 值函数
        value = self.policy.get_value(state)
        self.values.append(value)

        return action.item()

    def update_episode(self):
        self.calc_cumulative_rewards()

        assert len(self.cum_rewards) == len(self.values)


        advances = torch.tensor(self.cum_rewards).float().to(self.device) - torch.cat(self.values)
        advances = (advances - advances.mean()) / (advances.std() + self.eps)
        loss_policy = - (torch.cat(self.log_probs) * advances).mean() + advances.pow(2).mean()
        self.optimizer_policy.zero_grad()
        loss_policy.backward()
        self.optimizer_policy.step()

        self.rewards.clear()
        self.log_probs.clear()
        self.cum_rewards.clear()
        self.values.clear()


if __name__ == '__main__':
    env_id = 'MountainCar-v0'
    alg_id = 'REINFORCE_Baseline'
    env, num_states, num_actions = get_env_space(env_id)

    agent = REINFORCE_Baseline(num_states, num_actions, enable_gpu=True)

    max_episodes = 1000
    max_timestep = 10000

    writer = SummaryWriter()

    # 迭代所有episodes进行采样
    for episode in range(max_episodes):
        # 当前episode开始
        state = env.reset()
        episode_reward = 0

        for t in range(max_timestep):
            env.render()
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)

            episode_reward += reward
            agent.rewards.append(reward)

            # 当前episode　结束
            if done:
                break

        writer.add_scalar(alg_id, episode_reward, episode)
        print("Episode: {} , the episode reward is {}".format(episode, round(episode_reward, 3)))

        agent.update_episode()

    env.close()
