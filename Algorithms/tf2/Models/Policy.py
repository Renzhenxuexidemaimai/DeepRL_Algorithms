#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from Algorithms.tf2.Models.BasePolicy import BasePolicy
from Utils.tf2_util import NDOUBLE, DiagonalGaussian


class Policy(BasePolicy):

    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=tf.nn.leaky_relu, log_std=0):
        super(Policy, self).__init__(dim_state, dim_action, dim_hidden)
        self.dist = DiagonalGaussian(dim=dim_action)

        self.policy = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation),
            layers.Dense(self.dim_hidden, activation=activation),
            layers.Dense(self.dim_action)
        ])
        self.policy.build(input_shape=(None, self.dim_state))

        self.log_std = tf.Variable(name="action_log_std",
                                   initial_value=np.ones((dim_action, ), dtype=NDOUBLE) * log_std,
                                   trainable=True)

    def _get_dist(self, states):
        mean = self.policy(states)
        log_std = tf.tile(
            input=tf.expand_dims(self.log_std, axis=0),
            multiples=[mean.shape[0], 1])
        return {"mean": mean, "log_std": log_std}

    def call(self, states, **kwargs):
        param = self._get_dist(states)
        action = self.dist.sample(param)
        log_prob = self.dist.log_likelihood(action, param)
        return action, log_prob

    def get_log_prob(self, states, actions):
        param = self._get_dist(states)
        log_prob = self.dist.log_likelihood(actions, param)
        return log_prob

    def get_action_log_prob(self, states):
        return self.call(states)

    def get_entropy(self, states):
        param = self._get_dist(states)
        return self.dist.entropy(param)

    def get_kl(self, states):
        pass

# if __name__ == '__main__':
#     tf.keras.backend.set_floatx('float64')
#     x = tf.random.uniform((3, 4))
#     model = Policy(4, 3)
#
#     for _ in range(4):
#         with tf.GradientTape() as tape:
#             a, logp = model.get_action_log_prob(x)
#             print(a, logp)
#             ratio = logp * 3
#             loss = tf.reduce_mean(ratio, axis=-1)
#         opt = tf.keras.optimizers.Adam(lr=1e-4)
#         print(tape.watched_variables())
#         grads = tape.gradient(loss, model.trainable_variables)
#         opt.apply_gradients(zip(grads, model.trainable_variables))
