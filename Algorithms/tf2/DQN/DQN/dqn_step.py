#!/usr/bin/env python
# Created at 2020/3/3
import tensorflow as tf

from Utils.tf2_util import FLOAT


def dqn_step(value_net, optimizer_value, value_net_target, states, actions, rewards, next_states, masks, gamma):
    """update value net"""

    with tf.GradientTape() as tape:
        q_values = tf.reduce_sum(value_net(states) * tf.one_hot(actions, depth=value_net.dim_action, dtype=FLOAT), axis=-1)
        q_target_next_values = value_net_target(next_states)
        q_target_values = rewards + gamma * masks * tf.reduce_max(q_target_next_values, axis=-1)
        value_loss = tf.keras.losses.mean_squared_error(q_values, q_target_values)

    value_grads = tape.gradient(value_loss, value_net.trainable_variables)
    optimizer_value.apply_gradients(grads_and_vars=zip(value_grads, value_net.trainable_variables))
    return value_loss
