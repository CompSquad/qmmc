from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stat import norm


class BaseVariable(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, parents={}, value=None, **kwargs):
        
        self.parents = parents
        for parent in parents.itervalues():
            parent.children.append(self)
        self.value = value
        self.logp = self._get_logp


    @abstractmethod
    def _get_logp(self):
        pass


class GaussianVariable(object):
    
    def __init__(self, mu, sigma, observed=False):
        
        self.mu = mu
        self.sigma = sigma
        self.value = None
        self.observed = observed
        
    
    def logp(self, x):
        mu = self.mu.value
        sigma = self.sigma.value
        return norm.pdf(x, mu, sigma)