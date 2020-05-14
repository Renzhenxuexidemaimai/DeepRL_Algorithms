#!/usr/bin/env python
# Created at 2020/1/22
import tensorflow as tf

def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, old_log_probs, clip_epsilon):

    """update critic"""
    for _ in range(optim_value_iternum):
        with tf.GradientTape() as tape:
            values_pred = value_net(states)
            value_loss = tf.keras.losses.mean_squared_error(returns, y_pred=values_pred)

        grads = tape.gradient(value_loss, value_net.trainable_variables)
        optimizer_value.apply_gradients(grads_and_vars=zip(grads, value_net.trainable_variables))


    """update policy"""
    with tf.GradientTape() as tape:
        log_probs = policy_net.get_log_prob(states, actions)
        ratio =  tf.exp(log_probs - tf.stop_gradient(old_log_probs))
        surr1 = ratio * tf.stop_gradient(advantages)
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1 + clip_epsilon) * tf.stop_gradient(advantages)
        policy_surr = - tf.reduce_mean(tf.minimum(surr1, surr2), axis=-1)

    grads = tape.gradient(policy_surr, policy_net.trainable_variables)
    # grads, grad_norm = tf.clip_by_global_norm(grads, 40)
    optimizer_policy.apply_gradients(grads_and_vars=zip(grads, policy_net.trainable_variables))

    return value_loss, policy_surr
