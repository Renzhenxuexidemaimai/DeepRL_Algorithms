#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:30
import torch
import torch.nn.functional as F
import torch.nn as nn


class DuelingMLPPolicy(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DuelingMLPPolicy, self).__init__()
        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.state_out = nn.Linear(50, 1)
        self.state_out.weight.data.normal_(0, 0.1)
        self.advantage_out = nn.Linear(50, num_actions)
        self.advantage_out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        state_value = self.state_out(x)
        advantage_action_value = self.advantage_out(x)
        mean_advantage = torch.mean(advantage_action_value, 1, keepdim=True)
        return state_value + advantage_action_value - mean_advantage
