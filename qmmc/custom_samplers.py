"""Samplers specific to the RFQ structural model."""

__author__ = "arnaud.rachez@gmail.com"


import numpy as np
from scipy.stats import norm, truncnorm

from distrib import truncnorm_rvs, truncnorm_logpdf
from distrib import mintruncnorm_rvs, mintruncnorm_logpdf

def _sample_w(I, V, Y, mu_W, sigma_W, k, l):
    
    # Indices and lengths
    idx_done = I == 2
    idx_traded_away = I == 1
    idx_not_traded = I == 0
    
    # Sampling
    # Done case: Y < min(V, W) <=> V < Y & W < Y
    W_lower = (Y[idx_done] - mu_W) / sigma_W
    W_done = truncnorm_rvs(
            W_lower, np.inf, loc=mu_W, scale=sigma_W,
            shape=(sum(idx_done), l))
    
    # Traded away case: min(W) < min(Y, V) = m (Working)
    m = np.minimum(Y[idx_traded_away], V[idx_traded_away])
    W_traded_away = mintruncnorm_rvs(
            m, mu_W, sigma_W, shape=(sum(idx_traded_away), l))
    

    # Not traded case: min(W) > V (and Y > V)
    W_lower = (V[idx_not_traded] - mu_W) / sigma_W
    W_not_traded = truncnorm_rvs(
            W_lower, np.inf, loc=mu_W, scale=sigma_W,
            shape=(sum(idx_not_traded), l))
    
    # Finally, assignment.
    W = np.empty((k, l))
    
    W[idx_done] = W_done
    W[idx_traded_away] = W_traded_away
    W[idx_not_traded] = W_not_traded
    
    return W


def _logp_w(I, V, W, Y, mu_W, sigma_W):
    
    # Indices and lengths
    idx_done = I == 2
    idx_traded_away = I == 1
    idx_not_traded = I == 0
    _, l = W.shape
    
    # Done case: Y < min(V, W) => min(W) > Y
    W_lower = (Y[idx_done] - mu_W) / sigma_W
    logp_done = truncnorm_logpdf(
            W[idx_done], W_lower, np.inf, loc=mu_W, scale=sigma_W)
    
    # Traded away case: min(W) < min(Y, V) = m (Working)
    m = np.minimum(Y[idx_traded_away], V[idx_traded_away])
    F_0_m = 1 - (1 - norm.cdf(m, mu_W, sigma_W))
    logp_traded_away = np.sum(np.log(1. / F_0_m * l)
            + norm.logpdf(W[idx_traded_away, 0], loc=mu_W, scale=sigma_W)
            + np.log((1 - norm.cdf(W[idx_traded_away, 0], loc=mu_W, scale=sigma_W))**(l-1)))
    

    # Not traded case: min(W) > V (and Y > V)
    W_lower = (V[idx_not_traded] - mu_W) / sigma_W
    logp_not_traded = truncnorm_logpdf(
            W[idx_not_traded], W_lower, np.inf, loc=mu_W, scale=sigma_W)
    
    return logp_done + logp_traded_away + logp_not_traded


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
        w = _sample_w(I, V, Y, mu_W, sigma_W, k, l)
        
        self.W.value = w

    
    def logp(self):
        
        # Currently assigned values
        I = self.I.value
        Y = self.Y.value
        V = self.V.value
        W = self.W.value
        mu_W = self.mu.value
        sigma_W = self.sigma.value
        
        return _logp_w(I, V, W, Y, mu_W, sigma_W)


def _sample_v(I, W, Y, mu_V, sigma_V):
    
    # Indices and lengths
    idx_done = I == 2
    idx_traded_away = I == 1
    idx_not_traded = I == 0
    
    k_done = sum(idx_done)
    k_traded_away = sum(idx_traded_away)
    k_not_traded = sum(idx_not_traded)
    
    # Sampling
    # Done case: V < Y (and W > V)       
    V_lower = (Y[idx_done] - mu_V) / sigma_V
    V_done = truncnorm_rvs(
            V_lower, np.inf, loc=mu_V, scale=sigma_V, shape=(k_done, ))
    
    # Traded away case: V > min(W) (and min(W) < Y)
    C = np.min(W[idx_traded_away], axis=1)
    V_lower = (C - mu_V) / sigma_V
    V_traded_away = truncnorm_rvs(
            V_lower, np.inf, loc=mu_V, scale=sigma_V, shape=(k_traded_away, ))
    
    # Not traded case: V < min(min(W), Y)
    C = np.min(W[idx_not_traded], axis=1)
    m = np.minimum(Y[idx_not_traded], C)
    V_upper = (m - mu_V) / sigma_V
    V_not_traded = truncnorm_rvs(
            -np.inf, V_upper, loc=mu_V, scale=sigma_V, shape=(k_not_traded, ))
    
    # Finally, assignment.
    V = np.empty(k_done + k_traded_away + k_not_traded)
    
    V[idx_done] = V_done
    V[idx_traded_away] = V_traded_away
    V[idx_not_traded] = V_not_traded
    
    return V


def _logp_v(I, V, W, Y, mu_V, sigma_V):
    
    # Indices and lengths
    idx_done = I == 2
    idx_traded_away = I == 1
    idx_not_traded = I == 0
    
    # Done case: V < Y (and W > V)       
    V_lower = (Y[idx_done] - mu_V) / sigma_V
    logp_done = truncnorm_logpdf(
            V[idx_done], V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    # Traded away case: V > min(W) (and min(W) < Y)
    C = np.min(W[idx_traded_away], axis=1)
    V_lower = (C - mu_V) / sigma_V
    logp_traded_away = truncnorm_logpdf(
            V[idx_traded_away], V_lower, np.inf, loc=mu_V, scale=sigma_V)
    
    # Not traded case: V < min(min(W), Y)
    C = np.min(W[idx_not_traded], axis=1)
    m = np.minimum(Y[idx_not_traded], C)
    V_upper = (m - mu_V) / sigma_V
    logp_not_traded = truncnorm_logpdf(
            V[idx_not_traded], -np.inf, V_upper, loc=mu_V, scale=sigma_V)
    
    return logp_done + logp_traded_away + logp_not_traded


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
        mu_V = self.mu.value
        sigma_V = self.sigma.value
        
        self.V.value = _sample_v(I, W, Y, mu_V, sigma_V)
        
        
    def logp(self):
        
        # Currently assigned values
        I = self.I.value
        Y = self.Y.value
        W = self.W.value
        V = self.V.value
        mu_V = self.mu.value
        sigma_V = self.sigma.value
        
        return _logp_v(I, V, W, Y, mu_V, sigma_V)


def _decide(V, W, Y):
    
    if len(W) > 0:
        C = np.min(W)
    else:
        C = np.inf
    
    if Y <= np.minimum(C, V):return 2
    if C <= np.minimum(Y, V): return 1
    if V < np.minimum(C, Y): return 0


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


def _sample_kvw_from_prior(I, k, V, W, Y):
    
    S = -1
    I = I.value
    while S != I:
        k.sample()
        V.sample()
        W.sample()
    
        v = V.value
        w = W.value
        y = Y.value
    
        S = _decide(v, w, y)


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

