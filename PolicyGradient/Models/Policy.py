#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from PolicyGradient.Models.BasePolicy import BasePolicy


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Policy(BasePolicy):

    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=nn.LeakyReLU, log_std=0):
        super(Policy, self).__init__(dim_state, dim_action, dim_hidden)

        self.policy = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action)
        )

        self.log_std = nn.Parameter(torch.ones(1, self.dim_action) * log_std)

        self.apply(init_weight)

    def forward(self, x):
        mean = self.policy(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std) # 收敛更快
        return dist

    def get_log_prob(self, state, action):
        dist = self.forward(state)
        log_prob = dist.log_prob(action)
        return log_prob

    def get_action_log_prob(self, states):
        dist = self.forward(states)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_entropy(self, states):
        dist = self.forward(states)
        return dist.entropy()
