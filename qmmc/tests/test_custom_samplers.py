import unittest

import numpy as np
from scipy.stats import norm

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


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()