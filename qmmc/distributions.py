""" Custom distributions for the structural model."""

__author__ = "arnaud.rachez@gmail.com"

import numpy as np
from scipy.stats import norm, truncnorm


def truncnorm_rvs(lower, upper, loc, scale, shape):

    a = np.empty(shape)

    try:
        m, n = shape
        if np.isinf(lower).any():
            for i in xrange(m):
                a[i] = truncnorm.rvs(
                        lower, upper[i], loc=loc, scale=scale, size=n)
        elif np.isinf(upper).any():
            for i in xrange(m):
                a[i] = truncnorm.rvs(
                        lower[i], upper, loc=loc, scale=scale, size=n)
        else:
            for i in xrange(m):
                a[i] = truncnorm.rvs(
                        lower[i], upper[i], loc=loc, scale=scale, size=n)

    except ValueError:
        m = shape[0]
        if np.isinf(lower).any():
            for i in xrange(m):
                a[i] = truncnorm.rvs(lower, upper[i], loc=loc, scale=scale)
        elif np.isinf(upper).any():
            for i in xrange(m):
                a[i] = truncnorm.rvs(lower[i], upper, loc=loc, scale=scale)
        else:
            for i in xrange(m):
                a[i] = truncnorm.rvs(lower[i], upper[i], loc=loc, scale=scale)

    return a


def truncnorm_logpdf(value, lower, upper, loc, scale):

    logp = 0
    try:
        m, n = value.shape
        if np.isinf(lower).any():
            for i in xrange(m):
                logp += truncnorm.logpdf(
                        value[i], lower, upper[i], loc=loc, scale=scale)
        elif np.isinf(upper).any():
            for i in xrange(m):
                logp += truncnorm.logpdf(
                        value[i], lower[i], upper, loc=loc, scale=scale)
        else:
            for i in xrange(m):
                logp += truncnorm.logpdf(
                        value[i], lower[i], upper[i], loc=loc, scale=scale)
    
    except ValueError:
        m = value.shape[0]
        if np.isinf(lower).any():
            for i in xrange(m):
                logp += truncnorm.logpdf(
                        value[i], lower, upper[i], loc=loc, scale=scale)
        elif np.isinf(upper).any():
            for i in xrange(m):
                logp += truncnorm.logpdf(
                        value[i], lower[i], upper, loc=loc, scale=scale)
        else:
            for i in xrange(m):
                logp += truncnorm.logpdf(
                        value[i], lower[i], upper[i], loc=loc, scale=scale)

    return np.sum(logp)


def mintruncnorm_rvs(m, mu_W, sigma_W, shape):
    """ Sample l normal variables w_j per line s.t. min(w_j) < m.
    """

    k, l = shape
    W_traded_away = np.empty((k, l))
    u = np.random.rand(k)
    t = u * (1 - (1 - norm.cdf(m, loc=mu_W, scale=sigma_W))**l)
    W_min = norm.ppf(1 - (1 - t)**(1. / l), loc=mu_W, scale=sigma_W)
    W_traded_away[:, 0] = W_min
    W_traded_away[:, 1:] = truncnorm_rvs(
            (W_min - mu_W) / sigma_W,  np.inf, loc=mu_W, scale=sigma_W,
            shape=(k, l-1))
    return W_traded_away


def mintruncnorm_logpdf(W, m, mu_W, sigma_W):

    l = W.shape[0]
    F_0_m = 1 - (1 - norm.cdf(m, mu_W, sigma_W))
    logp = np.sum(
            np.log(1. / F_0_m * l)
            + norm.logpdf(W[0], loc=mu_W, scale=sigma_W)
            + np.log((1 - norm.cdf(W[1:], loc=mu_W, scale=sigma_W))**(l-1)))

    return logp

