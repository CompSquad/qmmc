""" Default sampler for the structural model.

This sampler is applicable to any V and W distribution. It samples from the 
prior and rejects until the consistency conditions are met.

This sampler can be very slow.
"""

__author__ = "arnaud.rachez@gmail.com"

import numpy as np
from scipy.stats import truncnorm

def _sample_v_single_buy(I, W, Y, mu_V, sigma_V):
    
    # Done case: V < Y (and W > V) 
    if I == 2:      
        V_lower = (Y - mu_V) / sigma_V
        V = truncnorm.rvs(V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    # Traded away case: V > min(W) (and min(W) < Y)
    elif I == 1:
        C = np.min(W)
        V_lower = (C - mu_V) / sigma_V
        V = truncnorm.rvs(V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    # Not traded case: V < min(min(W), Y)
    elif I == 0:
        C = np.min(W) if len(W) > 0 else np.inf
        m = np.minimum(Y, C)
        V_upper = (m - mu_V) / sigma_V
        V = truncnorm.rvs(-np.inf, V_upper, loc=mu_V, scale=sigma_V)
    
    else:
        raise ValueError("Unknown value (%s) for I" % I)
    
    return V

def _sample_v_single_sell(I, W, Y, mu_V, sigma_V):
    
    # Done case: Y > max(V, W) 
    if I == 2:      
        V_upper = (Y - mu_V) / sigma_V
        V = truncnorm.rvs(-np.inf, V_upper, loc=mu_V, scale=sigma_V)
    
    # Traded away case: max(W) > max(V, Y)
    elif I == 1:
        C = np.max(W)
        V_upper = (C - mu_V) / sigma_V
        V = truncnorm.rvs(-np.inf, V_upper, loc=mu_V, scale=sigma_V)
    
    # Not traded case: V > max(Y, W)
    elif I == 0:
        C = np.max(W) if len(W) > 0 else -np.inf
        m = np.maximum(Y, C)
        V_lower = (m - mu_V) / sigma_V
        V = truncnorm.rvs(V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    else:
        raise ValueError("Unknown value (%s) for I" % I)
    
    return V


class VSampler():
    
    def __init__(self, V, BuySell="Buy"):
        
        self.assigned = {V}
        S = list(V.children)[0]
        self.I = list(S.children)[0]
        self.Y = S.parents['Y']
        self.V = V
        self.W = S.parents['W']
        self.mu = V.parents['mu']
        self.sigma = V.parents['sigma']
        self.BuySell = BuySell
    
    def sample(self):

        I = self.I.value
        W = self.W.value
        Y = self.Y.value
        mu_V = self.mu.value
        sigma_V = self.sigma.value

        if self.BuySell == "Buy":
            self.V.value = _sample_v_single_buy(I, W, Y, mu_V, sigma_V)
        else:
            self.V.value = _sample_v_single_sell(I, W, Y, mu_V, sigma_V)

# TODO: Implement Sell _decide function as well.
def _decide_buy(V, W, Y):
    
    if len(W) > 0:
        C = np.min(W)
    else:
        C = np.inf
    
    if Y <= np.minimum(C, V):return 2
    if C <= np.minimum(Y, V): return 1
    if V < np.minimum(C, Y): return 0

def _decide_sell(V, W, Y):
    
    if len(W) > 0:
        C = np.max(W)
    else:
        C = -np.inf

    if Y >= np.maximum(C, V): return 2 # Done
    if C >= np.maximum(Y, V): return 1 # Traded away
    if V > np.maximum(C, Y): return 0 # Not traded


def _sample_vw_from_prior(I, V, W, Y, BuySell):
    
    S = -1
    I = I.value
    while S != I:
        W.sample()
        V.sample()
    
        v = V.value
        w = W.value
        y = Y.value
    
        if BuySell == "Buy":
            S = _decide_buy(v, w, y)
        else:
            S = _decide_sell(v, w, y)


class VWSampler(object):
    
    def __init__(self, k, V, W, Y, I, BuySell="Buy"):
        
        self.assigned = {k, V, W}
        
        self.k = k
        self.V = V
        self.W = W
        self.Y = Y
        self.I = I
        self.BuySell = BuySell
    
    def sample(self):

        _sample_vw_from_prior(self.I, self.V, self.W, self.Y, self.BuySell)


def _sample_kvw_from_prior(I, k, V, W, Y, BuySell):
    
    S = -1
    I = I.value
    while S != I:
        k.sample()
        W.sample()
        V.sample()
    
        v = V.value
        w = W.value
        # W = W.last_value
        y = Y.value
        
        if BuySell == "Buy":
            S = _decide_buy(v, w, y)
        else:
            S = _decide_sell(v, w, y)


class KVWSampler(object):
    
    def __init__(self, k, V, W, Y, I, BuySell="Buy"):
        
        self.assigned = {k, V, W}
        
        self.k = k
        self.V = V
        self.W = W
        self.Y = Y
        self.I = I
        self.BuySell = BuySell
    
    def sample(self):

        _sample_kvw_from_prior(self.I, self.k, self.V, self.W, self.Y, self.BuySell)

