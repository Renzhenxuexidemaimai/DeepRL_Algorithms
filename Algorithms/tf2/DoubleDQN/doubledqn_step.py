#!/usr/bin/env python
# Created at 2020/3/22
import tensorflow as tf

from Utils.tf2_util import TDOUBLE


def doubledqn_step(value_net, optimizer_value, value_net_target, states, actions, rewards, next_states, masks, gamma,
                   polyak, update_target=False):
    """update q value net"""


    q_target_actions = tf.argmax(value_net(next_states), axis=-1)
    q_next_max_values = tf.reduce_sum(
        value_net_target(next_states) * tf.one_hot(q_target_actions, depth=value_net.dim_action, dtype=TDOUBLE), axis=-1)
    q_target_values = rewards + gamma * masks * q_next_max_values

    with tf.GradientTape() as tape:
        q_values = tf.reduce_sum(value_net(states) * tf.one_hot(actions, depth=value_net.dim_action, dtype=TDOUBLE),
                                 axis=-1)
        value_loss = tf.keras.losses.mean_squared_error(q_values, q_target_values)

    value_grads = tape.gradient(value_loss, value_net.trainable_variables)
    optimizer_value.apply_gradients(grads_and_vars=zip(value_grads, value_net.trainable_variables))

    if update_target:
        """update target q value net"""
        for v_p, v_t_p in zip(value_net.trainable_variables, value_net_target.trainable_variables):
            v_t_p.assign((1 - polyak) * v_p + polyak * v_t_p)

    return value_loss
