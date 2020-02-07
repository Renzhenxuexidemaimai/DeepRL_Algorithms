#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午6:48
from Utils.torch_utils import device, FLOAT


def estimate_advantages(rewards, masks, values, gamma, tau):
    deltas = FLOAT(rewards.size(0), 1).to(device)
    advantages = FLOAT(rewards.size(0), 1).to(device)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    return advantages, returns
