#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52

import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=nn.LeakyReLU):
        super(Policy, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

        self.policy = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action * 2)
        )

    def forward(self, x):
        x = self.policy(x)
        mean, log_std = x[:, :self.dim_action], x[:, self.dim_action:]
        std = log_std.exp()

        return mean, std, log_std

    def get_action(self, state):
        mean, std, log_std = self.forward(state)
        normal = Normal(mean, std)

        action = normal.rsample()
        return action

    def get_log_prob(self, state, action):
        mean, std, log_std = self.forward(state)
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action)
        return log_prob


class Value(nn.Module):
    def __init__(self, dim_state, dim_hidden=128, activation=nn.ReLU):
        super(Value, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden

        self.value = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, 1)
        )

    def forward(self, x):
        value = self.value(x)

        return value
