#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午10:30
import math
import pickle

import torch
import torch.optim as optim

from Common.MemoryCollector import MemoryCollector
from Common.GAE import estimate_advantages
from PolicyGradient.Models.Policy import Policy
from PolicyGradient.Models.Policy_discontinuous import DiscretePolicy
from PolicyGradient.Models.Value import Value
from PolicyGradient.algorithms.ppo_step import ppo_step
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.torch_util import DOUBLE, device
from Utils.zfilter import ZFilter


class PPO_Minibatch:
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=4,
                 min_batch_size=2048,
                 lr_p=3e-4,
                 lr_v=3e-4,
                 gamma=0.99,
                 tau=0.95,
                 clip_epsilon=0.2,
                 ppo_mini_batch_size=64,
                 ppo_epochs=10,
                 model_path=None,
                 seed=1):
        self.env_id = env_id
        self.gamma = gamma
        self.tau = tau
        self.clip_epsilon = clip_epsilon
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.ppo_epochs = ppo_epochs

        self.render = render
        self.num_process = num_process
        self.min_batch_size = min_batch_size
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.model_path = model_path
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.env, env_continuous, num_states, num_actions = get_env_info(self.env_id)
        # seeding
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        if self.model_path:
            print("Loading Saved Model {}_ppo_mini.p".format(self.env_id))
            self.policy_net, self.value_net, self.running_state = pickle.load(
                open('{}/{}_ppo_mini.p'.format(self.model_path, self.env_id), "rb"))
        else:
            if env_continuous:
                self.policy_net = Policy(num_states, num_actions).double().to(device)
            else:
                self.policy_net = DiscretePolicy(num_states, num_actions).double().to(device)

            self.value_net = Value(num_states).double().to(device)
            self.running_state = ZFilter((num_states,), clip=5)

        self.collector = MemoryCollector(self.env, self.policy_net, render=self.render,
                                         running_state=self.running_state,
                                         num_process=self.num_process)

        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=self.lr_v)

    def choose_action(self, state):
        """select action according to policy"""
        state = DOUBLE(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_log_prob(state)
        return action, log_prob

    def eval(self, i_iter):
        """evaluate current model"""
        state = self.env.reset()
        test_reward = 0
        while True:
            self.env.render()
            state = self.running_state(state)

            action, _ = self.choose_action(state)
            action = action.cpu().numpy()[0]
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter):
        """learn model"""
        memory, log = self.collector.collect_samples(self.min_batch_size)

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}, sample time: {log['sample_time']: .4f}")

        # record reward information
        writer.add_scalars("PPOMiniBatch_exp{}".format(self.seed),
                           {"total reward": log['total_reward'],
                            "average reward": log['avg_reward'],
                            "min reward": log['min_episode_reward'],
                            "max reward": log['max_episode_reward'],
                            "num steps": log['num_steps']
                            }, i_iter)

        batch = memory.sample()  # sample all items in memory

        batch_state = DOUBLE(batch.state).to(device)
        batch_action = DOUBLE(batch.action).to(device)
        batch_log_prob = DOUBLE(batch.log_prob).to(device)
        batch_reward = DOUBLE(batch.reward).to(device)
        batch_mask = DOUBLE(batch.mask).to(device)
        batch_size = batch_state.shape[0]

        with torch.no_grad():
            batch_values = self.value_net(batch_state)

        batch_advantages, batch_returns = estimate_advantages(batch_reward, batch_mask, batch_values, self.gamma,
                                                              self.tau)
        v_loss, p_loss = torch.empty(1), torch.empty(1)

        mini_batch_num = int(math.ceil(batch_size / self.ppo_mini_batch_size))

        # update with mini-batch
        for _ in range(self.ppo_epochs):
            index = torch.randperm(batch_size)

            for i in range(mini_batch_num):
                ind = index[slice(i * self.ppo_mini_batch_size, min(batch_size, (i + 1) * self.ppo_mini_batch_size))]
                state, action, returns, advantages, old_log_pis = batch_state[ind], batch_action[ind], batch_returns[
                    ind], batch_advantages[ind], batch_log_prob[ind]

                v_loss, p_loss = ppo_step(self.policy_net, self.value_net, self.optimizer_p, self.optimizer_v, 1,
                                          state,
                                          action, returns, advantages, old_log_pis, self.clip_epsilon,
                                          1e-3)

        return v_loss, p_loss

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.policy_net, self.value_net, self.running_state),
                    open('{}/{}_ppo_mini.p'.format(save_path, self.env_id), 'wb'))
