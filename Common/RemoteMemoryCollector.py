#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午2:47
import math
import time

import ray
import torch

from Utils.replay_memory import Memory
from Utils.torch_util import device, FLOAT, DOUBLE

@ray.remote
class RemoteCollector:
    def __init__(self, pid, env, policy, render, running_state, min_batch_size):
        self.pid = pid
        self.env = env
        self.policy = policy
        self.render = render
        self.running_state = running_state
        self.min_batch_size = min_batch_size
        self.log = dict()
        self.memory = Memory()

    def collect(self):
        num_steps = 0
        num_episodes = 0

        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')
        total_reward = 0

        while num_steps < self.min_batch_size:
            state = self.env.reset()
            episode_reward = 0
            if self.running_state:
                state = self.running_state(state)

            for t in range(10000):
                if self.render:
                    self.env.render()
                state_tensor = DOUBLE(state).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob = self.policy.get_action_log_prob(state_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                if self.running_state:
                    next_state = self.running_state(next_state)

                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.memory.push(state, action, reward, next_state, mask, log_prob)
                num_steps += 1

                if done or num_steps >= self.min_batch_size:
                    break

                state = next_state

            # num_steps += (t + 1)
            num_episodes += 1
            total_reward += episode_reward
            min_episode_reward = min(episode_reward, min_episode_reward)
            max_episode_reward = max(episode_reward, max_episode_reward)

        self.log['num_steps'] = num_steps
        self.log['num_episodes'] = num_episodes
        self.log['total_reward'] = total_reward
        self.log['avg_reward'] = total_reward / num_episodes
        self.log['max_episode_reward'] = max_episode_reward
        self.log['min_episode_reward'] = min_episode_reward

    def get_log_memory(self):
        return self.log, self.memory

def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_episode_reward'] = max([x['max_episode_reward'] for x in log_list])
    log['min_episode_reward'] = min([x['min_episode_reward'] for x in log_list])

    return log


class MemoryCollector:
    def __init__(self, env, policy, render=False, running_state=None, num_process=1):
        self.env = env
        self.policy = policy
        self.running_state = running_state
        self.render = render
        self.num_process = num_process

        ray.init(num_cpus=num_process, ignore_reinit_error=True)

    def collect_samples(self, min_batch_size):
        self.policy.to(torch.device('cpu'))
        t_start = time.time()
        process_batch_size = int(math.floor(min_batch_size / self.num_process))

        workers = [RemoteCollector.remote(i, self.env, self.policy, self.render, self.running_state, process_batch_size)
                   for i in range(self.num_process)]

        task_ids = [worker.collect.remote() for worker in workers]
        results = ray.get([worker.get_log_memory.remote() for worker in workers])

        worker_logs = []
        memory = Memory()

        for result in results:
            worker_logs += result[0],
            memory.append(result[1])

        log = merge_log(worker_logs)
        log['sample_time'] = time.time() - t_start

        self.policy.to(device)
        return memory, log
