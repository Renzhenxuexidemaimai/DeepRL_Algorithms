#!/usr/bin/env python
# Created at 2020/2/9

import click

import torch
from torch.utils.tensorboard import SummaryWriter

from PolicyGradient.TRPO.trpo import TRPO


@click.command()
@click.option("--env_id", type=str, default="BipedalWalker-v2", help="Environment Id")
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_v", type=float, default=3e-4, help="Learning rate for Value Net")
@click.option("--gamma", type=float, default=0.99, help="Discount factor")
@click.option("--tau", type=float, default=0.95, help="GAE factor")
@click.option("--max_kl", type=float, default=1e-2, help="kl constraint for TRPO")
@click.option("--damping", type=float, default=1e-2, help="damping for TRPO")
@click.option("--batch_size", type=int, default=4096, help="Batch size")
@click.option("--mini_batch", type=bool, default=False, help="Update by mini-batch strategy")
@click.option("--trpo_mini_batch_size", type=int, default=64, help="PPO mini-batch size")
@click.option("--max_iter", type=int, default=500, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=50, help="Iterations to save the model")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(env_id, render, num_process, lr_p, lr_v, gamma, tau, max_kl, damping, batch_size, mini_batch,
         trpo_mini_batch_size, max_iter, eval_iter, save_iter, model_path, seed):
    # writer = SummaryWriter()

    trpo = TRPO(env_id, render, num_process, batch_size, lr_p, lr_v, gamma, tau, max_kl, damping,
                seed=seed)

    # for i_iter in range(1, max_iter + 1):
    #     trpo.learn(writer, i_iter)
    #
    #     if i_iter % eval_iter == 0:
    #         trpo.eval(i_iter)
    #
    #     if i_iter % save_iter == 0:
    #         trpo.save(model_path)
    #
    #     torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
