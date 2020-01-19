#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/3 下午4:40

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from PolicyGradient.PPO.ppo import PPOAgent
from Utils.env_utils import get_env_space


@click.command()
@click.option("--env_id", type=str, default="Ant-v2", help="Environment Id")
@click.option("--enable_gpu", type=bool, default=True, help='Use CUDA or not')
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=3e-4, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--render", type=bool, default=True, help="Render environment or not")
@click.option("--epsilon", type=float, default=0.1, help="Clip rate for PPO")
@click.option("--batch_size", type=int, default=2048, help="Batch size")
@click.option("--num_episodes", type=int, default=1000, help="Episodes to run")
def main(env_id, enable_gpu, lr_p, lr_v, gamma, tau, render, epsilon, batch_size, num_episodes):
    env, num_states, num_actions = get_env_space(env_id)

    memory_size = 100000
    agent = PPOAgent(num_states, num_actions, lr_p, lr_v, gamma, tau, epsilon, batch_size=batch_size,
                     memory_size=memory_size, enable_gpu=enable_gpu)

    iterations_, rewards_ = [], []

    for i in range(num_episodes):
        print(f"episode {i}")
        state = env.reset()
        episode_reward = 0
        for step in range(10000):
            if render:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.memory.push(torch.tensor([state]).float(),
                              torch.tensor([action]).float(),
                              torch.tensor([reward]).float(),
                              torch.tensor([next_state]).float(),
                              torch.tensor([0]).float() if done else torch.tensor([1]).float())
            episode_reward += reward

            # 当前episode　结束
            if done:
                break
            state = next_state

        v_loss, p_loss = agent.learn()
        agent.memory.clear()
        iterations_.append(i)
        rewards_.append(episode_reward)

        writer.add_scalar("{}/episode reward".format(env_id), episode_reward, i)
        writer.add_scalar("{}/v loss".format(env_id), v_loss, i)
        writer.add_scalar("{}/p loss".format(env_id), p_loss, i)

        print(f"episode: {i} , the episode reward is {episode_reward:.4f}")
    env.close()


if __name__ == '__main__':
    writer = SummaryWriter()

    main()

    writer.close()
