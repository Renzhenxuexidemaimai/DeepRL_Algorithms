#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal


class Policy(nn.Module):
    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=nn.LeakyReLU, log_std=0.0):
        super(Policy, self).__init__()

        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

        self.policy = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_action)
        )

        self.log_std = nn.Parameter(torch.ones(1, self.dim_action) * log_std)

    def forward(self, x):
        mean = self.policy(x)
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()

        return mean, std, log_std

    def get_action(self, state):
        mean, std, log_std = self.forward(state)
        normal = MultivariateNormal(mean, torch.diag_embed(std))

        action = normal.sample()
        return action

    def get_log_prob(self, state, action):
        mean, std, log_std = self.forward(state)
        normal = MultivariateNormal(mean, torch.diag_embed(std))
        log_prob = normal.log_prob(action)
        return log_prob


class Value(nn.Module):
    def __init__(self, dim_state, dim_hidden=128, activation=nn.LeakyReLU):
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
