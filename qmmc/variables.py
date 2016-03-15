""" Custom variables for the structural model."""

__author__ = "arnaud.rachez@gmail.com"

import numpy as np
from scipy.stats import norm, laplace, bernoulli

from minipgm.variables import BaseVariable
from minipgm.distributions import sep_rvs, sep_logpdf


class BernoulliFlip(BaseVariable):
    
    def __init__(self, p, x, k=2, value=None, observed=False, name=None,
                 size=None):
        
        self.k = k
        parents = {'p': p, 'x': x}     
        super(BernoulliFlip, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, p, x, size):
        
        try:
            size = x.shape[0]
        except AttributeError:
            size = None
        b = bernoulli.rvs(p, size=size)
        self.value = (x + b) % self.k
        return self.value

    def _logp(self, value, p, x):
        
        diff = np.array(value == x, dtype=int)
        pp = (1 - p) * diff + p * (1 - diff)
        return sum(np.log(pp))

class BernoulliNormal(BaseVariable):
    
    def __init__(self, mu, sigma, k, value=None, observed=False, name=None,
                 size=None):
        """ Sample a vector of dynamic size k from a normal distribution.
        
        Each time the `_sample` method is called, the size of `_value` can
        change.
        """
        
        parents = {'mu': mu, 'sigma': sigma, 'k': k}
        if size is not None:
            raise ValueError("size is variable and cannot be specified.")
        super(BernoulliNormal, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)
    
    def _sample(self, mu, sigma, k, size):
        
        return norm.rvs(loc=mu, scale=sigma, size=k) 

    def _logp(self, value, mu, sigma, k):

        return np.sum(norm.logpdf(value, loc=mu, scale=sigma))

class BernoulliLaplace(BaseVariable):
    
    def __init__(self, loc, scale, k, value=None, observed=False, name=None,
                 size=None):
        """ Sample a laplace vector of dynamic size k.
        
        Each time the `_sample` method is called, the size of `_value` can
        change.
        
        Parameters
        ----------
        loc: float
            location parameter of the laplace.
        scale: float
            scale parameter of the laplace.
        k: int
            size of the vector to be sampled.
        """
        
        parents = {'loc': loc, 'scale': scale, 'k': k}
        if size is not None:
            raise ValueError("size is variable and cannot be specified.")
        super(BernoulliLaplace, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)
    
    def _sample(self, loc, scale, k, size):
        
        return laplace.rvs(loc=loc, scale=scale, size=k) 

    def _logp(self, value, loc, scale, k):

        return np.sum(laplace.logpdf(value, loc=loc, scale=scale))

class BernoulliSEP(BaseVariable):

    def __init__(self, mu, sigma, nu, tau, k, value=None, observed=False,
                 name=None, size=None):
        """ Sample an SEP vector of dynamic size k.
        
        Each time the `_sample` method is called, the size of `_value` can
        change.
        """
        
        parents = {'mu': mu, 'sigma': sigma, 'nu': nu, 'tau': tau, 'k': k}
        if size is not None:
            raise ValueError("size is variable and cannot be specified.")
        super(BernoulliSEP, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)
    
    def _sample(self, mu, sigma, nu, tau, k, size):
        
        return sep_rvs(mu=mu, sigma=sigma, nu=nu, tau=tau, size=k) 

    def _logp(self, value, mu, sigma, nu, tau, k):
        
        logp = sep_logpdf(value, mu=mu, sigma=sigma, nu=nu, tau=tau)
        return np.sum(logp)
