#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import torch.nn as nn
from torch.distributions import Categorical

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

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
            nn.Linear(self.dim_hidden, self.dim_action),
            nn.Softmax(dim=1)
        )
        self.policy.apply(init_weight)


    def forward(self, x):
        probs = self.policy(x)
        return probs

    def get_action(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action

    def get_log_prob(self, state, action):
        probs = self.forward(state)
        m = Categorical(probs)
        log_prob = m.log_prob(action)
        return log_prob
