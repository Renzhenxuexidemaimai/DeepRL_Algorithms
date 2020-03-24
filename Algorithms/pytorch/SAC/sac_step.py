#!/usr/bin/env python
# Created at 2020/3/1


def sac_step(policy_net, value_net, value_net_target, q_net_1, q_net_2, optimizer_policy, optimizer_value,
             optimizer_q_net_1, optimizer_q_net_2, states, actions, rewards, next_states, masks, gamma, polyak,
             target_action_noise_std, target_action_noise_clip, action_high, update_policy=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)
