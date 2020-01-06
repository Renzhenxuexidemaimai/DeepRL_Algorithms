#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午10:30
import torch
import torch.nn as nn
import torch.optim as optim

from GAE.GAE import estimate_advantages
from PolicyGradient.Models.Mujoco_Policy_continuous import Policy, Value
from Utils.replay_memory import Memory


class PPOAgent:
    def __init__(self,
                 num_states,
                 num_actions,
                 lr_p=3e-4,
                 lr_v=3e-4,
                 gamma=0.995,
                 tau=0.96,
                 clip_epsilon=0.1,
                 batch_size=64,
                 memory_size=100000,
                 enable_gpu=False):

        if enable_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon

        self.memory = Memory(memory_size)
        self.policy_net_new, self.policy_net_old = Policy(num_states, num_actions, 64, nn.Tanh).to(self.device), \
                                                   Policy(num_states, num_actions, 64, nn.Tanh).to(self.device)

        self.policy_net_old.load_state_dict(self.policy_net_new.state_dict())

        self.value_net = Value(num_states).to(self.device)

        self.optimizer_p = optim.Adam(self.policy_net_new.parameters(), lr=lr_p)
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=lr_v)

    #  策略动作选择
    def choose_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        action = self.policy_net_new.get_action(state)[0].detach().cpu().numpy()
        return action

    def learn(self):
        batch = self.memory.sample()
        batch_state = torch.stack(batch.state, 0).to(self.device).squeeze(1)
        batch_action = torch.stack(batch.action, 0).to(self.device).squeeze(1)
        batch_reward = torch.stack(batch.reward, 0).to(self.device).squeeze(1)
        batch_mask = torch.stack(batch.mask, 0).to(self.device).squeeze(1)

        with torch.no_grad():
            batch_values = self.value_net(batch_state)
            old_log_pi = self.policy_net_old.get_log_prob(batch_state, batch_action)

        batch_advantages, batch_returns = estimate_advantages(batch_reward, batch_mask, batch_values, self.gamma,
                                                              self.tau, self.device)
        for _ in range(10):
            self.ppo_step(self.policy_net_new, self.value_net, self.optimizer_p, self.optimizer_v, 1, batch_state,
                          batch_action, batch_returns, batch_advantages, old_log_pi, self.clip_epsilon)

            self.policy_net_old.load_state_dict(self.policy_net_new.state_dict())

    def ppo_step(self, policy, value, opt_p, opt_v, iter_v, state, action, reward, advantage, old_log_pi, clip_epsilon):
        # update value net
        for i in range(iter_v):
            opt_v.zero_grad()

            v = value(state)
            v_loss = nn.MSELoss()(v, reward)

            v_loss.backward()
            opt_v.step()

        # update policy net
        log_pi = policy.get_log_prob(state, action)
        ratio = (log_pi - old_log_pi).exp()
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        opt_p.zero_grad()
        p_loss = -torch.min(clipped_ratio * advantage, ratio * advantage).mean()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 40)

        p_loss.backward()
        opt_p.step()
