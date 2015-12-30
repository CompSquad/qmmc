""" Default conditional samplers."""

__author__ = "arnaud.rachez@gmail.com" 

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import invgamma, norm



class MHSampler(object):

    __metaclass__ = ABCMeta
    
    def __init__(self, variable):
        
        self.assigned = {variable}
        
        self.variable = variable
        self.dependent = set(variable.children)
        
        self.rejected = 0
        self.accepted = 0

        for child in self.dependent:
            if child._deterministic:
                self.dependent |= child.children      
                self.dependent.remove(child)  
        self.history = []
        
    def sum_logp(self):
        
        sum_logp = self.variable.logp()
        for child in self.dependent:
            sum_logp += child.logp()
            
        return sum_logp
    
    @abstractmethod
    def _propose(self):
        pass
    
    def sample(self):
        
        logp_prev = self.sum_logp()
        self._propose()
        logp_new = self.sum_logp()
        
        if np.log(np.random.rand()) > logp_new - logp_prev:
            self.variable.reject()
            self.rejected +=1
        else:
            self.accepted += 1
        
        self.history.append(self.variable.value)
    
    def logp(self):
        
        return self.variable.logp()
    
    def get_history(self):
        
        return self.history

 
class NormalMHSampler(MHSampler):
    
    def __init__(self, variable, scaling=.1):  
         
        self.scaling = scaling
        super(NormalMHSampler, self).__init__(variable)
        
    def _propose(self):
        size = self.variable._size
        scaling = self.scaling
        value = self.variable.value + norm.rvs(loc=0, scale=scaling, size=size)
        self.variable.value = value


class NormalConjugateSampler(object):
    
    def __init__(self, mu, sigma):
        
        self.assigned = {mu, sigma}
        self.mu = mu
        self.sigma = sigma
        self.history = {'mu': [], 'sigma': []}
        
    def sample(self):
        
        mu_0 = self.mu.parents['mu'].value
        sigma_0 = self.mu.parents['sigma'].value
        scale_0 = self.sigma.parents['scale'].value
        shape_0 = self.sigma.parents['shape'].value
        mu = self.mu.value
        sigma = self.sigma.value
            
        
        # Sample mu
        s = 0.
        n = 0.
        
        children = self.mu.children
        for child in children:
            s += np.sum(child.value)
            n += np.product(child.value.shape)
        
        loc = (mu_0 / sigma_0**2 + s / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
        scale = np.sqrt(1. / (1. / sigma_0**2 + n / sigma**2))
        
        self.mu.value = norm.rvs(loc=loc, scale=scale)
        self.history['mu'].append(self.mu.value)
        
        # Sample sigma
        m = 0.
        n = 0.
        children = self.sigma.children
        for child in children:
            m += np.sum((child.value - mu)**2)
            n += np.product(child.value.shape) 
        
        scale = scale_0 + m / 2
        shape = shape_0 + n / 2
        self.sigma.value = np.sqrt(invgamma.rvs(shape, scale=scale))
        self.history['sigma'].append(self.sigma.value)

