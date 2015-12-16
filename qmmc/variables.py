""" QMMC v2.0, probabilistic programming attempt. """

__author__ = "arnaud.rachez@gmail.com"

from abc import ABCMeta, abstractmethod
from copy import copy
from inspect import getargspec
from types import NoneType

import numpy as np
from scipy.stats import beta, invgamma, norm, binom, bernoulli
from scipy.stats import truncnorm

from numba import jit


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
        
        b = bernoulli.rvs(p, size=x.shape[0])
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

        return norm.rvs(loc=mu, scale=sigma, size=size) 

    def _logp(self, value, mu, sigma):

        return np.sum(norm.logpdf(value, loc=mu, scale=sigma))


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


class LatentSampler(object):
    
    def __init__(self, V, W, I, Y, mu_V, sigma_V, mu_W, sigma_W):
        
        self.assigned = {V, W}
        
        self.V = V
        self.W = W
        self.I = I
        self.Y = Y
        self.mu_V = mu_V
        self.sigma_V = sigma_V
        self.mu_W = mu_W
        self.sigma_W = sigma_W
    
    def sample(self):
        
        # Currently assigned values
        I = self.I.value
        Y = self.Y.value
        mu_V = self.mu_V.value
        mu_W = self.mu_W.value
        sigma_V = self.sigma_V.value
        sigma_W = self.sigma_W.value
        
        # Indices and lengths
        idx_done = I == 2
        idx_traded_away = I == 1
        idx_not_traded = I == 0
        
        n_done = sum(idx_done)
        n_traded_away = sum(idx_traded_away)
        n_not_traded = sum(idx_not_traded)
        
        # Sampling
        # Done case: Y < min(V, W) <=> V < Y & W < Y
        W_lower = (Y[idx_done] - mu_W) / sigma_W
        W_done = truncated_normal(
                W_lower, np.inf, loc=mu_W, scale=sigma_W, size=n_done)
        
        V_lower = (Y[idx_done] - mu_V) / sigma_V
        V_done = truncated_normal(
                V_lower, np.inf, loc=mu_V, scale=sigma_V, size=n_done)
        
        # Traded away case: W < min(Y, V) = m
        m = np.minimum(Y[idx_traded_away], self.V.value[idx_traded_away])
        W_upper = (m - mu_W) / sigma_W
        W_traded_away = truncated_normal(
                -np.inf, W_upper, loc=mu_W, scale=sigma_W, size=n_traded_away)
        
        V_lower = (W_traded_away - mu_V) / sigma_V
        V_traded_away = truncated_normal(
                V_lower, np.inf, loc=mu_V, scale=sigma_V, size=n_traded_away)
        
        # Not traded case: V < min(W, Y) <=> W > V & V < Y
        W_lower = (self.V.value[idx_not_traded] - mu_W) / sigma_W
        W_not_traded = truncated_normal(
                W_lower, np.inf, loc=mu_W, scale=sigma_W, size=n_not_traded)
        
        m = np.minimum(Y[idx_not_traded], W_not_traded)
        V_upper = (m - mu_V) / sigma_V
        V_not_traded = truncated_normal(
                -np.inf, V_upper, loc=mu_V, scale=sigma_V, size=n_not_traded)
        
        # Finally, assignment.
        V = np.empty(self.V.value.shape)
        W = np.empty(self.W.value.shape)
        
        W[idx_done] = W_done
        W[idx_traded_away] = W_traded_away
        W[idx_not_traded] = W_not_traded
        
        V[idx_done] = V_done
        V[idx_traded_away] = V_traded_away
        V[idx_not_traded] = V_not_traded
        
        self.V.value = V
        self.W.value = W


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
            n += child.value.shape[0]
        
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
            n += child.value.shape[0]
        
        scale = scale_0 + m / 2
        shape = shape_0 + n / 2
        self.sigma.value = np.sqrt(invgamma.rvs(shape, scale=scale))
        self.history['sigma'].append(self.sigma.value)


def truncated_normal(lower, upper, loc, scale, size):
    
    a = np.empty(size)
    
    if np.isinf(lower).any():
        for i in xrange(size):
            a[i] = truncnorm.rvs(lower, upper[i], loc=loc, scale=scale)
    elif np.isinf(upper).any():
        for i in xrange(size):
            a[i] = truncnorm.rvs(lower[i], upper, loc=loc, scale=scale)
    else:
        for i in xrange(size):
            a[i] = truncnorm.rvs(lower[i], upper[i], loc=loc, scale=scale)
    
    return a

class Model(object):
    
    def __init__(self, variables, samplers=None): 

        self.lik = []
        
        # Variables to sample according to their posterior.
        self.variables = variables
        
        # Assigned samplers
        self.samplers = dict()
        if samplers is not None:
            for sampler in samplers:
                name = ' & '.join(var.name for var in sampler.assigned)
                self.samplers[name] = sampler
        
        # If no samplers have been provided, automatically assign a default
        # sampler to unobserved, stochastic variables.
        assigned = set()
        for sampler in self.samplers.itervalues():
            assigned |= sampler.assigned
        
        unassigned = {var for var in variables if var not in assigned}
        for var in unassigned:
            if not var._deterministic and not var._observed:
                self.samplers[var.name] = NormalMHSampler(var)

    
    def logp(self):
        """ Complete log-likelihood of stochastic variables."""
        
        return sum(v.logp() for v in self.variables if not v._deterministic)

    def estimate(self, n_iter):
        
        for _ in xrange(n_iter):
            for sampler in self.samplers.values():
                sampler.sample()
            self.lik.append(self.logp())
        self.logp_history = self.lik
     
if __name__ == "__main__":
    
    w_real = np.random.randn(5)
    X_real = np.random.randn(1000, 5)
    y_real = X_real.dot(w_real)
    
    mu_0, sigma_0 = Value(0), Value(3)
    X = Value(X_real)
    w = Normal(mu_0, sigma_0, value=w_real, size=5, name='w')
    
    @Function
    def Xdotw(w=w, X=X):
        return X.dot(w)
    
    y = Normal(Xdotw, Value(.5), value=y_real, observed=True, name='y')
    
    model = Model([y, Xdotw, w])
    model.estimate(1000)
    