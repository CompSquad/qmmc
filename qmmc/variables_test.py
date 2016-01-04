import unittest

import numpy as np
from numpy.testing import assert_array_less

from qmmc.variables import Value, Normal, BernoulliNormal, Beta, Binomial
from qmmc.distrib import truncnorm_rvs

class TestVariables(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testNormal(self):
        mu = Value(0.)
        sigma = Value(2.)
        x = Normal(mu, sigma, size=1000, name='x')
        
        self.assertAlmostEqual(
            x.value.mean(), mu.value, delta=3 * sigma.value / np.sqrt(1000))
        
        x.sample()
        self.assertEqual(x.value.shape, (1000, ))
        self.assertEqual(x._last_value.shape, (1000, ))
        
        self.assertAlmostEqual(
            x.value.mean(), mu.value, delta=3 * sigma.value / np.sqrt(1000))
        
        x.logp()
    
    def test_truncated_normal(self):
        
        n = 10
        lower = np.array(range(n))
        upper = np.inf
        x = truncnorm_rvs(lower, upper, loc=0, scale=1, shape=(n, ))
        assert_array_less(lower, x)
        
        lower = -np.inf
        upper = np.array(range(n))
        x = truncnorm_rvs(lower, upper, loc=0, scale=1, shape=(n, ))
        np.testing.assert_array_less(x, upper)
    
    
    def test_bernoulli_normal(self):
        
        mu, sigma = Value(0), Value(1)
        p = Beta(Value(2), Value(2), name='p')
        k = Binomial(p, k=Value(5), name='k')
        x = BernoulliNormal(mu, sigma, k, name='x')
        print k.value
        print x.value
        
        print k.sample()
        print x.sample()
        
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()