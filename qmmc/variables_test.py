import unittest

import numpy as np

from .variables import *

class TestVariables(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testNormal(self):
        mu = Value(0.)
        sigma = Value(2.)
        x = Normal(mu, sigma, size=1000)
        
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
        x = _truncated_normal(lower, upper, loc=0, scale=1, size=n)
        np.testing.assert_array_less(lower, x)
        
        lower = -np.inf
        upper = np.array(range(n))
        x = _truncated_normal(lower, upper, loc=0, scale=1, size=n)
        np.testing.assert_array_less(x, upper)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()