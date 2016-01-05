""" Custom distributions."""

__author__ = "arnaud.rachez@gmail.com"

import numpy as np
import scipy.stats
from scipy.special import gamma, gammaincinv
from scipy.stats import beta, norm, truncnorm, uniform


def ep_rvs(mu=0, alpha=1, beta=1, size=1):

    u = uniform.rvs(loc=0, scale=1, size=size)
    z = 2 *    np.abs(u - 1. / 2)
    z = gammaincinv(1. / beta, z)
    y = mu + np.sign(u - 1. / 2) * alpha * z**(1. / beta)
    return y


def ep_logpdf(x, mu=0, alpha=1., beta=1.):

    Z = (beta / (2 * alpha * gamma(1. / beta)))
    ker = np.exp(-(np.abs(x - mu) / alpha)**beta)
    logp = np.log(Z * ker)

    return logp


def ep2_rvs(mu, sigma, alpha, size=1):

    u = uniform.rvs(loc=0, scale=1, size=size)
    b = beta.rvs(1. / alpha, 1 - 1. / alpha, size=size)
    r = np.sign(uniform.rvs(loc=0, scale=1, size=size) - .5)
    z = r * (-alpha * b * np.log(u))**(1. / alpha)

    return z


def ep2_logpdf(x, mu=0, sigma=1, alpha=2):

    z = (x - mu) / sigma
    c = 2 * alpha**(1. / alpha - 1) * gamma(1. / alpha)
    d = np.exp(-np.abs(z)**alpha / alpha) / (sigma * c)
    return np.log(d)


def sep_rvs(mu=0, sigma=1, nu=0, tau=2, size=1):

    y = ep2_rvs(0, 1, tau, size=size)
    w = np.sign(y) * np.abs(y)**(tau / 2) * nu * np.sqrt(2. / tau)
    r = - np.sign(uniform.rvs(loc=0, scale=1, size=size) - scipy.stats.norm.cdf(w))
    z = r * y

    return mu + sigma * z


def sep_logpdf(x, mu=0., sigma=1., nu=0, tau=2):

    z = (x - mu) / sigma
    w = np.sign(z) * np.abs(z)**(tau / 2) * nu * np.sqrt(2. / tau)
    # Note: There is a sigma division in the paper
    logp = np.log(2) + norm.logcdf(w) + ep2_logpdf(x, mu, sigma, tau)

    return logp


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

