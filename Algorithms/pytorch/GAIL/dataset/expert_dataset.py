#!/usr/bin/env python
# Created at 2020/5/13
import multiprocessing

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from Utils.torch_util import FLOAT

"""
This dataset implementation reference stable_baselines and make some adaptation to pytorch
https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/dataset.py
"""


class ExpertDataset:
    def __init__(self, expert_data_path, train_fraction=0.7, traj_limitation=-1,
                 shuffle=True, batch_size=64, num_workers=multiprocessing.cpu_count()):
        """
        Custom dataset deal with gail expert dataset
        """
        traj_data = np.load(expert_data_path, allow_pickle=True)

        if 'state' in traj_data:
            states = traj_data['state']
            self._num_states = traj_data['state'].shape[-1]
        else:
            states = traj_data['obs']
            self._num_states = traj_data['obs'].shape[-1]
        if 'action' in traj_data:
            actions = traj_data['action']
            self._num_actions = traj_data['action'].shape[-1]
        else:
            actions = traj_data['acs']
            self._num_actions = traj_data['acs'].shape[-1]

        if 'ep_reward' in traj_data:
            self.ep_ret = traj_data['ep_reward']
        else:
            self.ep_ret = traj_data['ep_rets']

        if traj_limitation < 0:
            traj_limitation = len(self.ep_ret)
            self.ep_ret = self.ep_ret[:traj_limitation]

        # states, actions: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        if len(states.shape) > 2:
            self.states = np.reshape(states, [-1, np.prod(actions.shape[2:])])
            self.actions = np.reshape(states, [-1, np.prod(actions.shape[2:])])
        else:
            self.states = np.vstack(states)
            self.actions = np.vstack(actions)

        self.avg_ret = sum(self.ep_ret) / len(self.ep_ret)
        self.std_ret = np.std(np.array(self.ep_ret))
        self.shuffle = shuffle

        assert len(self.states) == len(self.actions), "The number of actions and observations differ " \
                                                      "please check your expert dataset"
        if 'ep_reward' in traj_data:
            self.num_traj = min(traj_limitation, len(traj_data['ep_reward']))
        else:
            self.num_traj = min(traj_limitation, len(traj_data['ep_rets']))

        self.num_transition = len(self.states)

        self.data_loader = DataLoader(
            TensorDataset(FLOAT(self.states),
                          FLOAT(self.actions),
                          ),
            shuffle=self.shuffle,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.train_loader = DataLoader(
            TensorDataset(FLOAT(self.states[:int(self.num_transition * train_fraction), :]),
                          FLOAT(self.actions[:int(self.num_transition * train_fraction), :]),
                          ),
            shuffle=self.shuffle,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.val_loader = DataLoader(
            TensorDataset(FLOAT(self.states[int(self.num_transition * train_fraction):, :]),
                          FLOAT(self.actions[int(self.num_transition * train_fraction):, :]),
                          ),
            shuffle=self.shuffle,
            batch_size=batch_size,
            num_workers=num_workers
        )

        self.log_info()

    def log_info(self):
        print("Total trajectories: %d" % self.num_traj)
        print("Total transitions: %d" % self.num_transition)
        print("Average returns: %f" % self.avg_ret)
        print("Std for returns: %f" % self.std_ret)

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions
