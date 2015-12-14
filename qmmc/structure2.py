""" QMMC v2.0, probabilistic programming attempt. """

__author__ = "arnaud.rachez@gmail.com"

from abc import ABCMeta, abstractmethod
from copy import copy
from inspect import getargspec
from types import NoneType

import numpy as np
from scipy.stats import beta, invgamma, norm, binom, bernoulli


class Error(Exception):
    """ Base class for handling Errors. """
    pass


class SamplingObservedVariableError(Error):
    """ Sampling observed variables is forbidden. """
    pass


class BaseVariable(object):

    __meta__ = ABCMeta

    def __init__(self, parents, value, observed, name, size):

        self.parents = parents
        self.children = set()
        for parent in parents.values():
            parent.children |= set([self])

        self._value = copy(value)
        self._size = size
        self._observed = observed
        self._deterministic = False

        self.name = name
        if type(name) not in [str, NoneType]:
            raise ValueError("`name` keyword must be str, got %s." % type(name))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name or super(BaseVariable, self).__repr__()

    @property
    def value(self):
        if self._value is None and not self._observed:
            self.sample()
        return self._value

    @value.setter
    def value(self, value):
        self._last_value = copy(self._value)
        self._value = copy(value)

    def logp(self):
        kwargs = {key: parent.value for key, parent in self.parents.iteritems()}
        kwargs['value'] = self.value
        return self._logp(**kwargs)

    def sample(self):
        if self._observed:
            raise SamplingObservedVariableError()

        kwargs = {key: parent.value for key, parent in self.parents.iteritems()}
        kwargs['size'] = self._size
        self._last_value = self._value
        self._value = self._sample(**kwargs)
        return self.value
    
    def reject(self):
        self._value = self._last_value
    
    @abstractmethod
    def _logp(self):
        pass
    
    @abstractmethod
    def _sample(self):
        pass


class Value(object):

    def __init__(self, value):

        self.value = value
        self.observed = True
        self.parent = {}
        self.children = set()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.value)

    def sample(self):
        raise SamplingObservedVariableError()

    def logp(self):
        return 0.


class Function(BaseVariable):
    
    def __init__(self, function, name=None):
        
        args, _, _, default = getargspec(function)
        parents = dict(zip(args, default))
        
        super(Function, self).__init__(
            parents=parents, value=None, observed=False, name=None, size=None)
        
        self.function = function
        self._deterministic = True

    
    def sample(self):
        kwargs = {key: parent.value for key, parent in self.parents.iteritems()}
        self.value = self.function(**kwargs)
        return self.value


class BernoulliFlip(BaseVariable):
    
    def __init__(self, p, x, value=None, observed=False, name=None,
                 size=None):
        
        parents = {'p': p, 'x': x}     
        super(BernoulliFlip, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, p, x, size):
        
        b = bernoulli.rvs(p, size=x.shape[0])
        self.value = (x + b) % 3
        return self.value

    def _logp(self, value, p, x):
        
        diff = np.array(value == x, dtype=int)
        pp = (1 - p) * diff + p * (1 - diff)
        return sum(np.log(pp))

class Beta(BaseVariable):
    
    def __init__(self, a, b, value=None, observed=False, name=None, size=None):
        
        parents = {'a': a, 'b': b}        
        super(Beta, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, a, b, size):

        return beta.rvs(a, b, size=size)

    def _logp(self, value, a, b):

        return np.sum(beta.logpdf(value, a, b))


class Binomial(BaseVariable):

    def __init__(self, p, k, value=None, observed=False, name=None, size=None):

        parents = {'p': p, 'k': k}
        super(Binomial, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, p, k, size):

        return binom.rvs(k, p, size=size)

    def _logp(self, value, p, k):

        return np.sum(binom.logpmf(value, k, p, loc=0))


class Normal(BaseVariable):

    def __init__(self, mu, sigma, value=None, observed=False, name=None,
                 size=None):

        parents = {'mu': mu, 'sigma': sigma}
        super(Normal, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, mu, sigma, size):

        return norm.rvs(mu, sigma, size=size) 

    def _logp(self, value, mu, sigma):

        return np.sum(norm.logpdf(value, mu, sigma))


class InvGamma(BaseVariable):

    def __init__(self, shape, scale, value=None, observed=False, name=None,
                 size=None):

        parents = {'shape': shape, 'scale': scale}
        super(InvGamma, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, shape, scale, size):

        return invgamma.rvs(shape, scale=scale, size=size)

    def _logp(self, value, shape, scale):

        return np.sum(np.log(invgamma.pdf(value, shape, scale=scale)))


class MHSampler(object):

    __meta__ = ABCMeta
    
    def __init__(self, variable):
        
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


class PriorMHSampler(MHSampler):
    
    def _propose(self):
        self.variable.sample()
    
    def sample(self):
        
        logp_prev = self.sum_logp()
        q_prev = self.variable.logp()
        self._propose()
        logp_new = self.sum_logp()
        q_new = self.variable.logp()
        if np.log(np.random.rand()) > logp_new + q_new - logp_prev - q_prev:
            self.variable.reject()

 
class NormalMHSampler(MHSampler):
    
    def __init__(self, variable, scaling=0.1):  
         
        self.scaling = scaling
        super(NormalMHSampler, self).__init__(variable)
        
    def _propose(self):
        size = self.variable._size
        if size is not None:
            self.variable.value = self.variable.value + \
                    self.scaling * np.random.randn(size)
        else:
            self.variable.value = self.variable.value + \
                    self.scaling * np.random.randn()
        

class Model(object):
    
    def __init__(self, variables, samplers=None): 

        self.lik = []
        
        # Variables to sample according to their posterior.
        self.variables = variables

        # If no samplers have been provided, automatically assign a default
        # sampler to all unobserved, stochastic variables.
        if samplers is None:
            samplers = {}
            for v in variables:
                if not v._deterministic and not v._observed:
                    samplers[v.name] = NormalMHSampler(v)
        self.samplers = samplers

    
    def logp(self):
        """ Complete log-likelihood of stochastic variables."""
        
        return sum(v.logp() for v in self.variables if not v._deterministic)

    def estimate(self, n_iter):
        
        for _ in xrange(n_iter):
            for sampler in self.samplers.values():
                sampler.sample()
            self.lik.append(self.logp())
        self.logp_history = self.lik
     
        