#!/usr/bin/env python
# Created at 2020/3/25
import torch
import torch.nn as nn

from Utils.torch_util import get_flat_params, set_flat_params


def sac_step(policy_net, value_net, value_net_target, q_net_1, q_net_2, optimizer_policy, optimizer_value,
             optimizer_q_net_1, optimizer_q_net_2, states, actions, rewards, next_states, masks, gamma, polyak,
             update_policy=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    """update value net"""
    with torch.no_grad():
        next_actions, next_log_probs = policy_net.rsample(next_states)
        target_value = torch.min(
            q_net_1(next_states, next_actions),
            q_net_2(next_states, next_actions)
        ) - next_log_probs.unsqueeze(-1)
    value_loss = nn.MSELoss()(target_value, value_net(next_states))

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update qvalue net"""

    q_value_1 = q_net_1(states, actions)
    q_value_2 = q_net_2(states, actions)
    with torch.no_grad():
        target_next_value = rewards + gamma * masks * value_net_target(next_states)

    q_value_loss_1 = nn.MSELoss()(target_next_value, q_value_1)
    optimizer_q_net_1.zero_grad()
    q_value_loss_1.backward()
    optimizer_q_net_1.step()

    q_value_loss_2 = nn.MSELoss()(target_next_value, q_value_2)
    optimizer_q_net_2.zero_grad()
    q_value_loss_2.backward()
    optimizer_q_net_2.step()

    policy_loss = None
    if update_policy:
        """update policy net"""
        new_actions, log_probs = policy_net.rsample(states)
        min_q = torch.min(
            q_net_1(states, new_actions),
            q_net_2(states, new_actions)
        )
        policy_loss = (log_probs.unsqueeze(-1) - min_q).mean()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        """ update target value net """
        value_net_target_flat_params = get_flat_params(value_net_target)
        value_net_flat_params = get_flat_params(value_net)

        set_flat_params(value_net_target, (1 - polyak) * value_net_flat_params + polyak * value_net_target_flat_params)

    return value_loss, q_value_loss_1, q_value_loss_2, policy_loss
