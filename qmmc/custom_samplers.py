"""Samplers specific to the RFQ structural model."""

__author__ = "arnaud.rachez@gmail.com"


import numpy as np
from scipy.stats import invgamma, norm


class WSampler(object):
    
    def __init__(self, W):
        
        self.assigned = {W}
        S = list(W.children)[0]
        self.I = list(S.children)[0]
        self.Y = S.parents['Y']
        self.V = S.parents['V']
        self.W = W
        self.mu = W.parents['mu']
        self.sigma = W.parents['sigma']
    
    def sample(self):
        
        # Currently assigned values
        I = self.I.value
        Y = self.Y.value
        V = self.V.value
        W = self.W.value
        mu_W = self.mu.value
        sigma_W = self.sigma.value
        
        k, l = W.shape
        w = _sample_w2(I, V, Y, mu_W, sigma_W, k, l)
        
        self.W.value = w
        

def _sample_w(I, V, Y, mu_W, sigma_W, k, l):
    
    # Indices and lengths
        idx_done = I == 2
        idx_traded_away = I == 1
        idx_not_traded = I == 0
        
        # Sampling
        # Done case: Y < min(V, W) <=> V < Y & W < Y
        W_lower = (Y[idx_done] - mu_W) / sigma_W
        W_done = truncated_normal(
                W_lower, np.inf, loc=mu_W, scale=sigma_W,
                shape=(sum(idx_done), l))
        
        # Traded away case: min(W) < min(Y, V) = m
        W_traded_away = norm.rvs(
                loc=mu_W, scale=sigma_W, size=(sum(idx_traded_away), l))
        
        m = np.minimum(Y[idx_traded_away], V[idx_traded_away])
        idx_retry = np.min(W_traded_away, axis=1) > m
        while sum(idx_retry) > 0:
            W_traded_away[idx_retry] = norm.rvs(
                loc=mu_W, scale=sigma_W, size=(sum(idx_retry), l))
            idx_retry = np.min(W_traded_away[idx_retry], axis=1) > m[idx_retry]
            
#         idx_min_W = np.argmin(W_traded_away[idx_retry], axis=1)
#         idx_min_W = np.random.randint(0, l, size=sum(idx_retry))
# 
#         W_upper = (m[idx_retry] - mu_W) / sigma_W
#         W_traded_away[(idx_retry, idx_min_W)] = truncated_normal(
#                 -np.inf, W_upper, loc=mu_W, scale=sigma_W,
#                 shape=(sum(idx_retry), ))

        # Not traded case: min(W) > V (and Y > V)
        W_lower = (V[idx_not_traded] - mu_W) / sigma_W
        W_not_traded = truncated_normal(
                W_lower, np.inf, loc=mu_W, scale=sigma_W,
                shape=(sum(idx_not_traded), l))
        
        # Finally, assignment.
        W = np.empty((k, l))
        
        W[idx_done] = W_done
        W[idx_traded_away] = W_traded_away
        W[idx_not_traded] = W_not_traded
        
        return W

def _consistent(Y, V, W):
    d = np.empty(Y.shape, dtype=int)
    
    C = np.min(W, axis=1)
    idx_done = Y <= np.minimum(C, V)
    idx_traded_away = C <= np.minimum(Y, V)
    idx_not_traded = V < np.minimum(C, Y)
    
    d[idx_not_traded] = 0
    d[idx_traded_away] = 1
    d[idx_done] = 2
    return d

def _sample_w2(I, V, Y, mu_W, sigma_W, k, l):
    
    W = norm.rvs(loc=mu_W, scale=sigma_W, size=(k, l))
    
    idx_retry = np.array([True] * k)
    while sum(idx_retry) > 0:
        W[idx_retry] = norm.rvs(
                loc=mu_W, scale=sigma_W, size=(sum(idx_retry), l))
        S = _consistent(Y, V, W)
        idx_retry = S != I
    
    return W
        
    
    

class VSampler(object):
    
    def __init__(self, V):
        
        self.assigned = {V}
        S = list(V.children)[0]
        self.I = list(S.children)[0]
        self.Y = S.parents['Y']
        self.V = V
        self.W = S.parents['W']
        self.mu = V.parents['mu']
        self.sigma = V.parents['sigma']
    
    
    def sample(self):
        
        # Currently assigned values
        I = self.I.value
        Y = self.Y.value
        W = self.W.value
        V = self.V.value
        mu_V = self.mu.value
        sigma_V = self.sigma.value
        
        # Indices and lengths
        idx_done = I == 2
        idx_traded_away = I == 1
        idx_not_traded = I == 0
        
        # Sampling
        # Done case: V < Y (and W > V)       
        V_lower = (Y[idx_done] - mu_V) / sigma_V
        V_done = truncated_normal(
                V_lower, np.inf, loc=mu_V, scale=sigma_V,
                shape=V[idx_done].shape)
        
        # Traded away case: V > min(W) (and min(W) < Y)
        C = np.min(W[idx_traded_away], axis=1)
        V_lower = (C - mu_V) / sigma_V
        V_traded_away = truncated_normal(
                V_lower, np.inf, loc=mu_V, scale=sigma_V,
                shape=V[idx_traded_away].shape)
        
        # Not traded case: V < min(min(W), Y)
        C = np.min(W[idx_not_traded], axis=1)
        m = np.minimum(Y[idx_not_traded], C)
        V_upper = (m - mu_V) / sigma_V
        V_not_traded = truncated_normal(
                -np.inf, V_upper, loc=mu_V, scale=sigma_V,
                shape=V[idx_not_traded].shape)
        
        # Finally, assignment.
        V = np.empty(V.shape)
        
        V[idx_done] = V_done
        V[idx_traded_away] = V_traded_away
        V[idx_not_traded] = V_not_traded
        
        self.V.value = V


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


def truncated_normal(lower, upper, loc, scale, shape):
    
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