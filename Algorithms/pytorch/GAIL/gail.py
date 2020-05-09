#!/usr/bin/env python
# Created at 2020/5/9

import math
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from Algorithms.pytorch.Models.ConfigPolicy import Policy
from Algorithms.pytorch.Models.Discriminator import Discriminator
from Algorithms.pytorch.Models.Value import Value
from Algorithms.pytorch.PPO.ppo_step import ppo_step
from Common.GAE import estimate_advantages
from Common.MemoryCollector import MemoryCollector
from Common.replay_memory import Memory
from Utils.env_util import get_env_info
from Utils.torch_util import FLOAT, to_device, device


class ExpertDataSet(Dataset):
    def __init__(self, data_set_path, num_states, num_actions):
        self.expert_data = np.array(pd.read_csv(data_set_path))
        assert num_states + num_actions == self.expert_data.size(1), "Trajectory data format not consistent !!!"
        self.state = FLOAT(self.expert_data[:, :num_states])
        self.action = FLOAT(self.expert_data[:, num_states:])
        self.length = self.state.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]


class GAIL:
    def __init__(self,
                 render=False,
                 num_process=4,
                 config=None,
                 env_id=None,
                 expert_trajectory_path=None):
        """
        note that env should match trajectory
        :param config:
        :param env_id:
        :param expert_trajectory_path:
        """
        self.render = render
        self.env_id = env_id
        self.num_process = num_process
        self.config = config
        self.expert_trajectory_path = expert_trajectory_path

        self._load_expert_trajectory()
        self._init_model()

    def _load_expert_trajectory(self):
        num_expert_states = self.config["general"]["num_states"]
        num_expert_actions = self.config["general"]["num_actions"]
        expert_batch_size = self.config["general"]["expert_batch_size"]

        self.expert_dataset = ExpertDataSet(data_set_path=self.config["general"]["expert_data_path"],
                                            num_states=num_expert_states,
                                            num_actions=num_expert_actions)
        self.expert_data_loader = DataLoader(dataset=self.expert_dataset,
                                             batch_size=expert_batch_size,
                                             shuffle=True,
                                             num_workers=multiprocessing.cpu_count() // 2)

    def _init_model(self):
        """seeding"""
        seed = self.config["general"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env, env_continuous, num_states, num_actions = get_env_info(self.env_id)

        self.value = Value(dim_state=self.config["value"]["dim_state"],
                           dim_hidden=self.config["value"]["dim_hidden"],
                           activation=self.config["value"]["activation"]
                           )
        self.policy = Policy(config=self.config["policy"])

        self.discriminator = Discriminator(dim_state=self.config["discriminator"]["dim_state"],
                                           dim_action=self.config["discriminator"]["dim_action"],
                                           dim_hidden=self.config["discriminator"]["dim_hidden"],
                                           activation=self.config["discriminator"]["activation"]
                                           )

        self.collector = MemoryCollector(self.env, self.policy, render=self.render,
                                         running_state=None,
                                         num_process=self.num_process)

        print("Model Structure")
        print(self.policy)
        print(self.value)
        print(self.discriminator)
        print()

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.config["policy"]["learning_rate"])
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=self.config["value"]["learning_rate"])
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.config["discriminator"]["learning_rate"])

        self.discriminator_func = nn.BCELoss()

        to_device(self.value, self.policy, self.discriminator, self.discriminator_func)

    def choose_action(self, state):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy.get_action_log_prob(state)
        return action, log_prob

    def collect_samples(self, batch_size, render=False):
        # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
        memory = Memory()
        state = self.env.reset()
        while True:
            if render:
                self.env.render()

            action, log_prob = self.choose_action(state)
            next_state, _, done, _ = self.env.step(action)  # ignore env reward
            mask = 0 if done else 1
            reward = self.discriminator(state, action)

            memory.push(state, action, reward, next_state, mask, log_prob)

            if done:
                break
            state = next_state

        self.env.close()
        return memory.sample(batch_size=batch_size)

    def learn(self, writer, i_iter):
        self.policy.train()
        self.value.train()
        self.discriminator.train()

        # collect generated batch
        gen_batch = self.collect_samples(self.config["ppo"]["sample_batch_size"])
        # batch: ('state', 'action', 'next_state', 'log_prob', 'mask')
        gen_batch_state = torch.stack(gen_batch.state)  # [trajectory length * parallel size, state size]
        gen_batch_action = torch.stack(gen_batch.action)  # [trajectory length * parallel size, action size]
        gen_batch_old_log_prob = torch.stack(gen_batch.log_prob)  # [trajectory length * parallel size, 1]
        gen_batch_mask = torch.stack(gen_batch.mask)  # [trajectory length * parallel size, 1]

        ####################################################
        # update discriminator
        ####################################################
        for expert_batch_state, expert_batch_action in self.expert_data_loader:
            gen_r = self.discriminator(gen_batch_state, gen_batch_action)
            expert_r = self.discriminator(expert_batch_state.to(device), expert_batch_action.to(device))

            expert_labels = torch.ones_like(expert_r)
            gen_labels = torch.zeros_like(gen_r)

            e_loss = self.discriminator_func(expert_r, expert_labels)
            g_loss = self.discriminator_func(gen_r, gen_labels)
            d_loss = e_loss + g_loss

            self.optimizer_discriminator.zero_grad()
            d_loss.backward()
            self.optimizer_discriminator.step()

        writer.add_scalar('train/loss/d_loss', d_loss.item(), i_iter)
        writer.add_scalar("train/loss/e_loss", e_loss.item(), i_iter)
        writer.add_scalar("train/loss/g_loss", g_loss.item(), i_iter)
        writer.add_scalar('train/reward/expert_r', expert_r.mean().item(), i_iter)
        writer.add_scalar('train/reward/gen_r', gen_r.mean().item(), i_iter)

        ####################################################
        # update policy by ppo [mini_batch]
        ####################################################
        if i_iter > 100:
            with torch.no_grad():
                gen_batch_value = self.value(gen_batch_state)
                gen_batch_reward = self.discriminator(gen_batch_state, gen_batch_action)

            gen_batch_advantage, gen_batch_return = estimate_advantages(gen_batch_reward, gen_batch_mask,
                                                                        gen_batch_value, self.config["gae"]["gamma"],
                                                                        self.config["gae"]["tau"])

            ppo_optim_i_iters = self.config["ppo"]["ppo_optim_i_iters"]
            ppo_mini_batch_size = self.config["ppo"]["ppo_mini_batch_size"]
            gen_batch_size = gen_batch_state.shape[0]
            optim_iter_num = int(math.ceil(gen_batch_size / ppo_mini_batch_size))

            for _ in range(ppo_optim_i_iters):
                perm = torch.randperm(gen_batch_size)

                for i in range(optim_iter_num):
                    ind = perm[slice(i * ppo_mini_batch_size,
                                     min((i + 1) * ppo_mini_batch_size, gen_batch_size))]
                    mini_batch_state, mini_batch_action, mini_batch_advantage, mini_batch_return, \
                    mini_batch_old_log_prob = gen_batch_state[ind], gen_batch_action[ind], \
                                              gen_batch_advantage[ind], gen_batch_return[ind], gen_batch_old_log_prob[
                                                  ind]

                    v_loss, p_loss = ppo_step(policy_net=self.policy,
                                              value_net=self.value,
                                              optimizer_policy=self.optimizer_policy,
                                              optimizer_value=self.optimizer_value,
                                              optim_value_iternum=self.config["value"]["optim_value_iter"],
                                              states=mini_batch_state,
                                              actions=mini_batch_action,
                                              returns=mini_batch_return,
                                              old_log_probs=mini_batch_old_log_prob,
                                              advantages=mini_batch_advantage,
                                              clip_epsilon=self.config["ppo"]["clip_ratio"],
                                              l2_reg=self.config["value"]["l2_reg"])

                    writer.add_scalar('train/loss/p_loss', p_loss, i_iter)
                    writer.add_scalar('train/loss/v_loss', v_loss, i_iter)

        print(f" Training episode:{i_iter} ".center(80, "#"))
        print('gen_r:', gen_r.mean().item())
        print('expert_r:', expert_r.mean().item())
        print('d_loss', d_loss.item())

    def eval(self, i_iter, render=False):
        self.policy.eval()
        self.value.eval()
        self.discriminator.eval()

        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            action, _ = self.choose_action(state)
            action = action.cpu().numpy()[0]
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # torch.save((self.discriminator, self.policy, self.value), f"{save_path}/{self.exp_name}.pt")
        torch.save(self.discriminator, f"{save_path}/{self.env_id}_Discriminator.pt")
        torch.save(self.policy, f"{save_path}/{self.env_id}_Policy.pt")
        torch.save(self.value, f"{save_path}/{self.env_id}_Value.pt")

    def load_model(self, model_path):
        # load entire model
        # self.discriminator, self.policy, self.value = torch.load(model_path, map_location=device)
        self.discriminator = torch.load(f"{model_path}_Discriminator.pt", map_location=device)
        self.policy = torch.load(f"{model_path}_Policy.pt", map_location=device)
        self.value = torch.load(f"{model_path}_Value.pt", map_location=device)
