""" Base variables to build probabilistic graph. """

__author__ = "arnaud.rachez@gmail.com"

from abc import ABCMeta, abstractmethod
from copy import copy
from inspect import getargspec

import numpy as np
from scipy.stats import beta, invgamma, laplace, norm
from scipy.stats import binom, bernoulli, uniform
from qmmc.distrib import sep_rvs, sep_logpdf


class Error(Exception):
    """ Base class for handling Errors. """
    pass


class SamplingObservedVariableError(Error):
    """ Sampling observed variables is forbidden. """
    pass


class BaseVariable(object):

    __metaclass__ = ABCMeta

    def __init__(self, parents, value, observed, name, size):

        self.parents = parents
        self.children = set()
        for parent in parents.values():
            parent.children |= set([self])

        self._value = copy(value)
        self._size = size
        if value is not None:
            try:
                self._size = value.shape
            except:
                self._size = None 
        self._observed = observed
        self._deterministic = False

        self.name = name
        if type(name) is not str:
            raise ValueError("You must provide a `name` for your variable")

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
        self._observed = True
        self._deterministic = True
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
    
    def __init__(self, function):
        
        args, _, _, default = getargspec(function)
        parents = dict(zip(args, default))
        name = str(function)
        
        super(Function, self).__init__(
            parents=parents, value=None, observed=False, name=name, size=None)
        
        self.function = function
        self._deterministic = True

    @property
    def value(self):
        kwargs = {key: parent.value for key, parent in self.parents.iteritems()}
        return self.function(**kwargs)
    
    def sample(self):
        kwargs = {key: parent.value for key, parent in self.parents.iteritems()}
        self._last_value = self._value
        self._value = self.function(**kwargs)
        return self.value
    
    def _sample(self):
        raise NotImplementedError()
    
    def _logp(self):
        raise NotImplementedError()


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

class Beta(BaseVariable):
    
    def __init__(self, a, b, value=None, observed=False, name=None, size=None):
        
        parents = {'a': a, 'b': b}        
        super(Beta, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, a, b, size):

        return beta.rvs(a, b, size=size)

    def _logp(self, value, a, b):
        
        if value < 0 or value > 1:
            raise ValueError("Domain Error.")
        
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

        return np.sum(invgamma.logpdf(value, shape, scale=scale))


class Normal(BaseVariable):

    def __init__(self, mu, sigma, value=None, observed=False, name=None,
                 size=None):

        parents = {'mu': mu, 'sigma': sigma}
        super(Normal, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)

    def _sample(self, mu, sigma, size):

        return norm.rvs(loc=mu, scale=sigma, size=size) 

    def _logp(self, value, mu, sigma):

        return np.sum(norm.logpdf(value, loc=mu, scale=sigma))


class BernoulliNormal(BaseVariable):
    
    def __init__(self, mu, sigma, k, value=None, observed=False, name=None,
                 size=None):
        
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


class Laplace(BaseVariable):
    
    def __init__(self, loc, scale, value=None, observed=False, name=None,
                 size=None):
        
        parents = {'loc': loc, 'scale': scale}
        super(Laplace, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)
        
    def _sample(self, loc, scale, size):

        return laplace.rvs(loc=loc, scale=scale, size=size)

    def _logp(self, value, loc, scale):

        return np.sum(laplace.logpdf(value, loc=loc, scale=scale))  


class BernoulliLaplace(BaseVariable):
    
    def __init__(self, loc, scale, k, value=None, observed=False, name=None,
                 size=None):
        
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


class SEP(BaseVariable):
    
    def __init__(self, mu, sigma, nu, tau, value=None, observed=False,
                 name=None, size=None):
        
        parents = {'mu': mu, 'sigma': sigma, 'nu': nu, 'tau': tau}
        super(SEP, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)
    
    def _sample(self, mu, sigma, nu, tau, size):
        
        return sep_rvs(mu=mu, sigma=sigma, nu=nu, tau=tau, size=size) 

    def _logp(self, value, mu, sigma, nu, tau):
        
        logp = sep_logpdf(value, mu=mu, sigma=sigma, nu=nu, tau=tau)
        return np.sum(logp)


class BernoulliSEP(BaseVariable):

    def __init__(self, mu, sigma, nu, tau, k, value=None, observed=False,
                 name=None, size=None):
        
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


class Uniform(BaseVariable):
    
    
    def __init__(self, lower, upper, value=None, observed=False, name=None,
                 size=None):
    
        parents = {'lower': lower, 'upper': upper}
        super(Uniform, self).__init__(
            parents=parents, value=value, observed=observed, name=name,
            size=size)
    
    def _sample(self, lower, upper, size):
        
        return uniform.rvs(loc=lower, scale=upper-lower, size=size)

    def _logp(self, value, lower, upper):
        
        if value < lower or value > upper:
            raise ValueError("Domain Error.") 

        return np.sum(uniform.logpdf(value, loc=lower, scale=upper-lower)) 
    
