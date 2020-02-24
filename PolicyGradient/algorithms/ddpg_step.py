#!/usr/bin/env python
# Created at 2020/1/22
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def ddpg_step(policy_net, policy_net_target, value_net, value_net_target, optimizer_policy, optimizer_value,
              optim_value_iternum, states, rewards, values, next_states, masks, gamma, l2_reg, polyak):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    """update critic"""
    for _ in range(optim_value_iternum):
        target_actions = policy_net_target(next_states)
        target_values = value_net_target(next_states, target_actions)
        target_pred = rewards + gamma * masks * target_values

        value_loss = nn.MSELoss()(values, target_pred)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update actor"""

    gradient_actions = policy_net(states)
    policy_loss = - value_net(states, gradient_actions).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()


    """soft update target nets"""
    policy_net_flat_params = get_flat_params(policy_net)
    policy_net_target_flat_params = get_flat_params(policy_net_target)
    set_flat_params(policy_net_target, polyak * policy_net_target_flat_params + (1 - polyak) * policy_net_flat_params)

    value_net_flat_params = get_flat_params(value_net)
    value_net_target_flat_params = get_flat_params(value_net_target)
    set_flat_params(value_net_target, polyak * value_net_target_flat_params + (1 - polyak) * value_net_flat_params)

    return value_loss, policy_loss
