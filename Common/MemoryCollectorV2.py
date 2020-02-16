#!/usr/bin/env python
# Created at 2020/2/15
import math
import multiprocessing
import time

import torch

from Utils.replay_memory import Memory
from Utils.torch_utils import FLOAT, device


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_episode_reward'] = max([x['max_episode_reward'] for x in log_list])
    log['min_episode_reward'] = min([x['min_episode_reward'] for x in log_list])

    return log


class CollectorWoker(multiprocessing.Process):
    def __init__(self, queue, env, policy, render, running_state, min_batch_size):
        super().__init__()
        self.queue = queue
        self.env = env
        self.policy = policy
        self.render = render
        self.running_state = running_state
        self.min_batch_size = min_batch_size

        # statistic of episodes data
        self._num_steps = 0
        self._num_episodes = 0
        self._total_reward = 0
        self._min_episode_reward = float('inf')
        self._max_episode_reward = float('-inf')
        self._memory = Memory()
        self._log = dict()

    def run(self):
        torch.randn(self.pid)
        while self._num_steps < self.min_batch_size:
            state = self.env.reset()
            episode_reward = 0
            if self.running_state:
                state = self.running_state(state)

            for t in range(10000):
                if self.render:
                    self.env.render()

                state_tensor = FLOAT(state).unsqueeze(0)
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
                self._memory.push(state, action, reward, next_state, mask, log_prob)
                if done:
                    break

                state = next_state

            self._num_steps += (t + 1)
            self._num_episodes += 1
            self._total_reward += episode_reward
            self._min_episode_reward = min(episode_reward, self._min_episode_reward)
            self._max_episode_reward = max(episode_reward, self._max_episode_reward)

        print(self._num_steps)
        self._log['num_steps'] = self._num_steps
        self._log['num_episodes'] = self._num_episodes
        self._log['total_reward'] = self._total_reward
        self._log['avg_reward'] = self._total_reward / self._num_episodes
        self._log['max_episode_reward'] = self._max_episode_reward
        self._log['min_episode_reward'] = self._min_episode_reward

        self.queue.put([self._memory, self._log])


class MemoryCollectorV2:
    def __init__(self, env, policy, render=False, running_state=None, num_process=1):
        self.env = env
        self.policy = policy
        self.running_state = running_state
        self.render = render
        self.num_process = num_process

    def collect_samples(self, min_batch_size):
        self.policy.to(torch.device('cpu'))
        t_start = time.time()
        process_batch_size = int(math.floor(min_batch_size / self.num_process))
        queue = multiprocessing.Queue()
        workers = [CollectorWoker(queue, self.env, self.policy,
                                  self.render, self.running_state, process_batch_size)]

        # don't render other parallel processes
        for i in range(self.num_process - 1):
            worker_args = (queue, self.env, self.policy,
                           False, self.running_state, process_batch_size)
            worker = CollectorWoker(*worker_args)
            workers.append(worker)

        for worker in workers:
            worker.start()

        worker_logs = []
        worker_memories = []
        for _ in workers:
            worker_memory, worker_log = queue.get()
            worker_memories += worker_memory,
            worker_logs += worker_log,

        memory = Memory()
        # concat all memories
        for worker_memory in worker_memories:
            memory.append(worker_memory)

        log = merge_log(worker_logs)
        t_end = time.time()
        log['sample_time'] = t_end - t_start

        self.policy.to(device)
        return memory, log
