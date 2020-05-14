#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/19 下午3:32

import tensorflow as tf
import numpy as np

TLONG = tf.int64
TFLOAT = tf.float32
TDOUBLE = tf.float64

NLONG = np.int64
NFLOAT = np.float64
NDOUBLE = np.float64


"""
taken from
https://github.com/keiohta/tf2rl/blob/master/tf2rl/distributions/
"""

class Distribution(object):
    def __init__(self, dim):
        self._dim = dim
        self._tiny = 1e-8

    @property
    def dim(self):
        raise self._dim

    def kl(self, old_dist, new_dist):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio(self, x, old_dist, new_dist):
        raise NotImplementedError

    def entropy(self, dist):
        raise NotImplementedError

    def log_likelihood_sym(self, x, dist):
        raise NotImplementedError

    def log_likelihood(self, xs, dist):
        raise NotImplementedError


class Categorical(Distribution):
    def __init__(self, dim):
        super().__init__(dim)

    def kl(self, old_param, new_param):
        """
        Compute the KL divergence of two Categorical distribution as:
            p_1 * (\log p_1  - \log p_2)
        """
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob + self._tiny) - tf.math.log(new_prob + self._tiny)))

    def likelihood_ratio(self, x, old_param, new_param):
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return (tf.reduce_sum(new_prob * x) + self._tiny) / (tf.reduce_sum(old_prob * x) + self._tiny)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
            \log \sum(p_i * x_i)
        :param x (tf.Tensor or np.ndarray): Values to compute log likelihood
        :param param (Dict): Dictionary that contains probabilities of outputs
        :return (tf.Tensor): Log probabilities
        """
        probs = param["prob"]
        assert probs.shape == x.shape, \
            "Different shape inputted. You might have forgotten to convert `x` to one-hot vector."
        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, param):
        probs = param["prob"]
        # NOTE: input to `tf.random.categorical` is log probabilities
        # For more details, see https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/random/categorical
        # [probs.shape[0], 1]
        return tf.random.categorical(tf.math.log(probs), 1)

    def entropy(self, param):
        probs = param["prob"]
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        super().__init__(dim)

    @property
    def dim(self):
        return self._dim

    def kl(self, old_param, new_param):
        """
        Compute KL divergence of two distributions as:
            {(\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2} / (2 * \sigma_2^2) + ln(\sigma_2 / \sigma_1)
        :param old_param (Dict):
            Gaussian distribution to compare with that contains
            means: (batch_size * output_dim)
            std: (batch_size * output_dim)
        :param new_param (Dict): Same contents with old_param
        """
        old_means, old_log_stds = old_param["mean"], old_param["log_std"]
        new_means, new_log_stds = new_param["mean"], new_param["log_std"]
        old_std = tf.math.exp(old_log_stds)
        new_std = tf.math.exp(new_log_stds)

        numerator = tf.math.square(old_means - new_means) \
            + tf.math.square(old_std) - tf.math.square(new_std)
        denominator = 2 * tf.math.square(new_std) + 1e-8
        return tf.math.reduce_sum(numerator / denominator + new_log_stds - old_log_stds)

    def likelihood_ratio(self, x, old_param, new_param):
        llh_new = self.log_likelihood(x, new_param)
        llh_old = self.log_likelihood(x, old_param)
        return tf.math.exp(llh_new - llh_old)

    def log_likelihood(self, x, param):
        """
        Compute log likelihood as:
        """
        means = param["mean"]
        log_stds = param["log_std"]
        assert means.shape == log_stds.shape
        zs = (x - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * self.dim * tf.math.log(2 * NDOUBLE(np.pi))

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        # reparameterization
        return means + tf.random.normal(shape=means.shape, dtype=TDOUBLE) * tf.math.exp(log_stds)

    def entropy(self, param):
        log_stds = param["log_std"]
        return tf.reduce_sum(log_stds + tf.math.log(tf.math.sqrt(2 * np.pi * np.e)), axis=-1)