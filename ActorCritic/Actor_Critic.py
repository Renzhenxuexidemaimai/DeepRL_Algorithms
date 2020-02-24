import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

from ActorCritic.Models.Actor_Critic_Net import ActorCriticNet
from Utils.env_util import get_env_space


class ActorCritic:
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate=0.002,
                 gamma=0.995,
                 eps=torch.finfo(torch.float32).eps,
                 enable_gpu=False
                 ):

        if enable_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.actor_critc = ActorCriticNet(num_states, num_actions).to(self.device)

        self.optimizer = optim.Adam(self.actor_critc.parameters(), lr=learning_rate)

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
        probs, value = self.actor_critc(state)

        # 对action进行采样,并计算log probability
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        self.log_probs.append(log_prob)
        self.values.append(value)

        return action.item()

    def learn(self):
        self.calc_cumulative_rewards()
        assert len(self.cum_rewards) == len(self.values)

        rewards = torch.tensor(self.cum_rewards).unsqueeze(-1).to(self.device)
        values = torch.stack(self.values).squeeze(-1)
        log_probs = torch.stack(self.log_probs)

        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        advances = rewards - values

        loss = -(log_probs *advances).mean() + advances.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空 buffer
        self.rewards.clear()
        self.log_probs.clear()
        self.cum_rewards.clear()
        self.values.clear()


if __name__ == '__main__':
    env_id = 'MountainCar-v0'
    alg_id = 'ActorCritic'
    env, num_states, num_actions = get_env_space(env_id)

    agent = ActorCritic(num_states, num_actions, enable_gpu=True)
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

        agent.learn()

    env.close()
