""" Default conditional samplers."""

__author__ = "arnaud.rachez@gmail.com" 

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm



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
