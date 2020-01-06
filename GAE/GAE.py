#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午6:48


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1).to(device)
    advantages = tensor_type(rewards.size(0), 1).to(device)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, returns
