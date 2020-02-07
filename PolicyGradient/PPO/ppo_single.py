#!/usr/bin/env python
# Created at 2020/1/31
import math

import gym
import torch
import torch.nn as nn
import torch.optim as opt
from gym.spaces import Discrete
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.tensorboard import SummaryWriter

from Common.MemoryCollector import MemoryCollector
from Common.GAE import estimate_advantages
from PolicyGradient.algorithms.ppo_step import ppo_step
from Utils.torch_utils import FLOAT, device
from Utils.zfilter import ZFilter

"""
    a minimal ppo example in a single file
"""


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class ActorContinuous(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=128, activation=nn.LeakyReLU):
        super(ActorContinuous, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        # mapping states to actions with probability distribution
        self.actor = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_actions)
        )

        self.action_log_std = nn.Parameter(torch.zeros(1, n_actions))
        self.apply(init_weight)

    def forward(self, x):
        action_mean = self.actor(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        dist = MultivariateNormal(action_mean, torch.diag_embed(action_std))
        return dist

    def get_action_log_prob(self, states):
        dist = self.forward(states)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action, action_log_prob

    def get_log_prob(self, states, actions):
        dist = self.forward(states)
        action_log_prob = dist.log_prob(actions)

        return action_log_prob

    def get_entropy(self, states):
        dist = self.forward(states)
        return dist.entropy()


class ActorDiscontinuous(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden=128, activation=nn.LeakyReLU):
        super(ActorDiscontinuous, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden

        # mapping states to actions with probability distribution
        self.actor = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_actions),
            nn.Softmax(dim=-1)
        )

        self.apply(init_weight)

    def forward(self, x):
        action_prob = self.actor(x)
        dist = Categorical(probs=action_prob)
        return dist

    def get_action_log_prob(self, states):
        dist = self.forward(states)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action, action_log_prob

    def get_log_prob(self, states, actions):
        dist = self.forward(states)
        action_log_prob = dist.log_prob(actions)

        return action_log_prob

    def get_entropy(self, states):
        dist = self.forward(states)
        return dist.entropy()


class Critic(nn.Module):
    def __init__(self, n_states, n_hidden=128, activation=nn.LeakyReLU):
        super(Critic, self).__init__()

        # mapping states to value
        self.critic = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_hidden),
            activation(),
            nn.Linear(n_hidden, 1)
        )

        self.apply(init_weight)

    def forward(self, x):
        return self.critic(x)


class PPO(object):
    def __init__(self, actor, critic, lr, gamma, tau, epsilon, mini_batch_size, ppo_epochs):
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.policy = actor
        self.value = critic

        self.opt_p = opt.Adam(self.policy.parameters(), lr=lr)
        self.opt_v = opt.Adam(self.value.parameters(), lr=lr)

    def select_action(self, states):
        states_tensor = FLOAT(states).unsqueeze(0).to(device)
        action, log_prob = self.policy.get_action_log_prob(states_tensor)

        return action, log_prob

    def train(self, memory):
        batch = memory.sample()
        batch_states = FLOAT(batch.state).to(device)
        batch_actions = FLOAT(batch.action).to(device)
        batch_log_probs = FLOAT(batch.log_prob).to(device)
        batch_masks = FLOAT(batch.mask).to(device)
        batch_rewards = FLOAT(batch.reward).to(device)
        batch_size = batch_states.shape[0]

        with torch.no_grad():
            batch_values = self.value(batch_states)

        batch_advantages, batch_returns = estimate_advantages(batch_rewards, batch_masks, batch_values, self.gamma,
                                                              self.tau)

        # mini-batch ppo update
        mini_batch_num = int(math.ceil(batch_size / self.mini_batch_size))
        for _ in range(self.ppo_epochs):
            idx = torch.randperm(batch_size)

            for i in range(mini_batch_num):
                mini_batch_idx = idx[i * self.mini_batch_size: min((i + 1) * self.mini_batch_size, batch_size)]

                mini_batch_states, mini_batch_actions, mini_batch_log_probs, mini_batch_returns, mini_batch_advantages = \
                    batch_states[mini_batch_idx], batch_actions[mini_batch_idx], batch_log_probs[mini_batch_idx], \
                    batch_returns[mini_batch_idx], batch_advantages[mini_batch_idx]

                self.ppo_step(mini_batch_states, mini_batch_actions, mini_batch_returns, mini_batch_advantages,
                              mini_batch_log_probs, self.epsilon, 1e-3)

    def ppo_step(self, states, actions,
                 returns, advantages, old_log_probs, clip_epsilon, l2_reg, optim_value_iternum=1):

        for _ in range(optim_value_iternum):
            values_pred = self.value(states)
            value_loss = nn.MSELoss()(values_pred, returns)
            # weight decay
            for param in self.value.parameters():
                value_loss += param.pow(2).sum() * l2_reg

            self.opt_v.zero_grad()
            value_loss.backward()
            self.opt_v.step()

        """update policy"""
        log_probs = self.policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()

        self.opt_p.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
        self.opt_p.step()

        return value_loss.item(), policy_surr.item()


env_id = 'BipedalWalker-v2'
lr = 3e-4
gamma = 0.99
tau = 0.95
epsilon = 0.2
batch_size = 2048
mini_batch_size = 64
ppo_epochs = 10

num_iters = 2000
env = gym.make(env_id)
# env = env.unwrapped

num_states = env.observation_space.shape[0]
if type(env.action_space) == Discrete:
    num_actions = env.action_space.n
else:
    num_actions = env.action_space.shape[0]

actor = ActorContinuous(num_states, num_actions).to(device)
critic = Critic(num_states).to(device)

running_state = ZFilter((num_states,), clip=5)
agent = MemoryCollector(env, actor, running_state=running_state, num_process=4)

opt_p = opt.Adam(actor.parameters(), lr=lr)
opt_v = opt.Adam(critic.parameters(), lr=lr)


def train(memory):
    batch = memory.sample()
    batch_states = FLOAT(batch.state).to(device)
    batch_actions = FLOAT(batch.action).to(device)
    batch_log_probs = FLOAT(batch.log_prob).to(device)
    batch_masks = FLOAT(batch.mask).to(device)
    batch_rewards = FLOAT(batch.reward).to(device)
    batch_size = batch_states.shape[0]

    with torch.no_grad():
        batch_values = critic(batch_states)

    batch_advantages, batch_returns = estimate_advantages(batch_rewards, batch_masks, batch_values, gamma,
                                                          tau)

    # mini-batch ppo update
    mini_batch_num = int(math.ceil(batch_size / mini_batch_size))
    for _ in range(ppo_epochs):
        idx = torch.randperm(batch_size)

        for i in range(mini_batch_num):
            mini_batch_idx = idx[i * mini_batch_size: min((i + 1) * mini_batch_size, batch_size)]

            mini_batch_states, mini_batch_actions, mini_batch_log_probs, mini_batch_returns, mini_batch_advantages = \
                batch_states[mini_batch_idx], batch_actions[mini_batch_idx], batch_log_probs[mini_batch_idx], \
                batch_returns[mini_batch_idx], batch_advantages[mini_batch_idx]

            ppo_step(actor, critic, opt_p, opt_v, 1, mini_batch_states, mini_batch_actions, mini_batch_returns, mini_batch_advantages,
                          mini_batch_log_probs, epsilon, 1e-3)



def main():
    # ppo = PPO(actor, critic, lr, gamma, tau, epsilon, mini_batch_size, ppo_epochs)
    writer = SummaryWriter("ppo_single")

    for iter in range(num_iters):
        memory, log = agent.collect_samples(batch_size)
        train(memory)

        print(f"Iter: {iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}, sample time: {log['sample_time']: .4f}")

        writer.add_scalar("PPO", log['max_episode_reward'], iter)

        torch.cuda.empty_cache()

    env.close()


if __name__ == '__main__':
    main()
