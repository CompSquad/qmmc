"""Samplers specific to the RFQ structural model."""

__author__ = "arnaud.rachez@gmail.com"


import numpy as np
from scipy.stats import norm, truncnorm

from minipgm.distributions import truncnorm_rvs, truncnorm_logpdf
from minipgm.distributions import mintruncnorm_rvs, mintruncnorm_logpdf


def _sample_k(k, I):
    
    kv = k.sample()
    if I == 1:
        while kv == 0:
            kv = k.sample()
    return kv


def _sample_w_single(I, V, Y, mu_W, sigma_W, l):
    
    if l == 0:
        return norm.rvs(0, 1, size=0)

    # Done case: Y < min(V, W) <=> V < Y & W < Y
    if I == 2:
        W_lower = (Y - mu_W) / sigma_W
        W = truncnorm.rvs(W_lower, np.inf, loc=mu_W, scale=sigma_W, size=l)
    
    # Traded away case: min(W) < min(Y, V) = m (Working)
    elif I == 1:
        m = np.minimum(Y, V)
        W = mintruncnorm_rvs(m, mu_W, sigma_W, shape=(1, l))[0, :]
    
    # Not traded case: min(W) > V (and Y > V)
    elif I == 0:
        W_lower = (V - mu_W) / sigma_W
        W = truncnorm.rvs(W_lower, np.inf, loc=mu_W, scale=sigma_W, size=l)

    return W


def _logp_w_single(I, V, W, Y, mu_W, sigma_W):
    
    # Done case: Y < min(V, W) => min(W) > Y
    if I == 2:
        W_lower = (Y - mu_W) / sigma_W
        logp = truncnorm.logpdf(W, W_lower, np.inf, loc=mu_W, scale=sigma_W)
    
    # Traded away case: min(W) < min(Y, V) = m (Working)
    elif I == 1:
        m = np.minimum(Y, V)
        logp = mintruncnorm_logpdf(W, m, mu_W, sigma_W)
    
    # Not traded case: min(W) > V (and Y > V)
    elif I == 0:
        W_lower = (V - mu_W) / sigma_W
        logp = truncnorm.logpdf(W, W_lower, np.inf, loc=mu_W, scale=sigma_W)
    
    return np.sum(logp)


def _sample_v_single(I, W, Y, mu_V, sigma_V):
    
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
    
    return V


def _logp_v_single(I, V, W, Y, mu_V, sigma_V):
    
    # Done case: V < Y (and W > V)       
    if I == 2:
        V_lower = (Y - mu_V) / sigma_V
        logp = truncnorm.logpdf(V, V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    # Traded away case: V > min(W) (and min(W) < Y)
    elif I == 1:
        C = np.min(W)
        V_lower = (C - mu_V) / sigma_V
        logp = truncnorm.logpdf(V, V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    # Not traded case: V < min(min(W), Y)
    elif I == 0:
        C = np.min(W) if W.shape[0] > 0 else np.inf
        m = np.minimum(Y, C)
        V_upper = (m - mu_V) / sigma_V
        logp = truncnorm.logpdf(V, -np.inf, V_upper, loc=mu_V, scale=sigma_V)
    
    return logp


def _sample_kvw_from_conditional(I, k, V, W, Y):
    
    I = I.value
    Y = Y.value
    V = V.value
     
    mu_V = V.parents['mu'].value
    sigma_V = V.parents['sigma'].value
    mu_W = W.parents['mu'].value
    sigma_W = W.parents['sigma'].value
     
    k = _sample_k(k, I)
     
    W = _sample_w_single(I, V, Y, mu_W, sigma_W, k)
    W.value = W
     
    V = _sample_v_single(I, W, Y, mu_V, sigma_V)
    V.value = V


class KVWSampler(object):
    
    def __init__(self, k, V, W, Y, I):
        
        self.assigned = {k, V, W}
        
        self.k = k
        self.V = V
        self.W = W
        self.Y = Y
        self.I = I
    
    def sample(self):
        
        _sample_kvw_from_prior(self.I, self.k, self.V, self.W, self.Y)


    def logp(self):
        
        I = self.I.value
        V = self.V.value
        W = self.W.value
        Y = self.Y.value
        
        mu_W = self.W.parents['mu'].value
        sigma_W = self.W.parents['sigma'].value
        mu_V = self.V.parents['mu'].value
        sigma_V = self.V.parents['sigma'].value
        
        logp_w = _logp_w_single(I, V, W, Y, mu_W, sigma_W)
        logp_v = _logp_v_single(I, V, W, Y, mu_V, sigma_V)
        
        return logp_v + logp_w

