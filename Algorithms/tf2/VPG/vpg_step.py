#!/usr/bin/env python
# Created at 2020/3/23

import tensorflow as tf

def vpg_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages):
    """update critic"""
    for _ in range(optim_value_iternum):
        with tf.GradientTape() as tape:
            values_pred = value_net(states)
            value_loss = tf.keras.losses.mean_squared_error(returns, values_pred)

        grads = tape.gradient(value_loss, value_net.trainable_variables)
        optimizer_value.apply_gradients(grads_and_vars=zip(grads, value_net.trainable_variables))

    """update policy"""
    with tf.GradientTape() as tape:
        log_probs = policy_net.get_log_prob(states, actions)
        policy_loss = - tf.reduce_mean(log_probs * advantages, axis=-1)

    grads = tape.gradient(policy_loss, policy_net.trainable_variables)
    optimizer_policy.apply_gradients(grads_and_vars=zip(grads, policy_net.trainable_variables))

    return value_loss, policy_loss
