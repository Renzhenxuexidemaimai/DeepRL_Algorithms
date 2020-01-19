#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午2:47
import math
import multiprocessing
import time
from multiprocessing import Queue, Process, current_process

import numpy as np

from Utils.torch_utils import to_device


def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size):
    pass


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class BaseAgent(object):
    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads - 1):
            worker_args = (i + 1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log


def work(pid, q, xs):
    print(pid, current_process().name)
    q.put([current_process().name, [x ** 2 for x in xs]])


def main():
    q = Queue()
    t_start = time.time()
    workers = []

    print(multiprocessing.cpu_count())

    for i in range(5):
        p = Process(target=work, args=(i + 1, q, [x for x in range(2500)]))
        workers.append(p)
        p.start()

    for worker in workers:
        worker.join()

    t_end = time.time()
    res = []
    for i in range(5):
        proc_name, res_list = q.get()
        res.extend(res_list)
        print("%s result obtained!!! " % proc_name)

    print("%.2f s take !!!" % ((t_end - t_start),))


if __name__ == '__main__':
    main()
