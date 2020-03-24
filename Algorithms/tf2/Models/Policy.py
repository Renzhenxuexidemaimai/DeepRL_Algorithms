#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/2 下午9:52
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from Algorithms.tf2.Models.BasePolicy import BasePolicy
from Utils.tf2_util import NDOUBLE, TDOUBLE


class Policy(BasePolicy):

    def __init__(self, dim_state, dim_action, dim_hidden=128, activation=tf.nn.leaky_relu, log_std=0):
        super(Policy, self).__init__(dim_state, dim_action, dim_hidden)

        self.policy = tf.keras.models.Sequential([
            layers.Dense(self.dim_hidden, activation=activation),
            layers.Dense(self.dim_hidden, activation=activation),
            layers.Dense(self.dim_action)
        ], name="Policy")

        self.log_std = tf.Variable(name="action_log_std",
                                   initial_value=np.ones((dim_action,), dtype=NDOUBLE) * log_std,
                                   trainable=True)
        self.policy.build(input_shape=(None, self.dim_state))

    def call(self, states, **kwargs):
        mean = self.policy(states)
        return mean, self.log_std

    def get_log_prob(self, states, actions):
        mean, log_std = self.call(states)
        log_prob = self.gaussian_prob(actions, mean, log_std)
        return log_prob

    def get_action_log_prob(self, states):
        mean, log_std = self.call(states)
        std = tf.exp(self.log_std)  # Take exp. of Std deviation
        action = mean + tf.random.normal(tf.shape(mean), dtype=TDOUBLE) * std  # Sample action from Gaussian Dist
        log_prob = self.gaussian_prob(mean, action, self.log_std)  # Calculate logp at timestep t for actions
        return action, log_prob

    def get_entropy(self, _):
        entropy = tf.reduce_sum(self.log_std.read_value() + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return entropy

    def get_kl(self, x):
        pass

    def gaussian_prob(self, x, mean, log_std):
        pre_sum = -0.5 * (((x - mean) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=-1)
#
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
