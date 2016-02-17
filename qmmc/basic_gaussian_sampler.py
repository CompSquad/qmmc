import numpy as np
from scipy.stats import norm, truncnorm

from minipgm.distributions import truncnorm_rvs, truncnorm_logpdf
from minipgm.distributions import mintruncnorm_rvs, mintruncnorm_logpdf

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
