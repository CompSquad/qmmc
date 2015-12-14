import unittest

import matplotlib.pyplot as plt

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm, laplace

from qmmc.distrib import dEP, dEP2, dSEP
from qmmc.distrib import rEP, rEP2, rSEP
from qmmc.distrib import likelihoodEP, likelihoodEP2, likelihoodSEP
from qmmc.distrib import MetropolisHastings


class DistribTest(unittest.TestCase):

    def setUp(self):
            pass 

    def tearDown(self):
            pass

    def test_special_cases(self):
        X = np.linspace(-1, 1)
        for x in X:
            # The normal parameterization is exact.
            self.assertAlmostEqual(dEP2(x, 0, 1, 2), norm.pdf(x))
            # There should be a sqrt(2) factor for the chosen EP parameterization.
            self.assertAlmostEqual(dEP2(x, 0, 1, 1), laplace.pdf(x))
    
    def test_log(self): 
        self.log_helper(0, 1, 1)
        self.log_helper(0, 2, 1)
        self.log_helper(1, 1, 1)
        self.log_helper(1, 1, 2)
        self.log_helper(0, 1.3, 1.6)
        self.log_helper(.3, .7, 3.6)
        
        
    def log_helper(self, mu=0, sigma=1, beta=1):
        """ Test that the `log` keyword in distrib_cython.dEP is working properly."""
        X = np.linspace(-1, 1)
        for x in X:
            self.assertAlmostEqual(
                    dEP(x, mu, sigma, beta, log=True),
                    np.log(dEP(x, mu, sigma, beta)), delta=.001)
            

    def test_loglikEP(self):
        p = np.array([0, 1.3, 1.5])
        y = rEP(*p, size=100000)
        ll_real = likelihoodEP(y, *p, log=True)
        for _ in range(100):
            params = p + .1 * np.random.randn(3)
            ll = likelihoodEP(y, *params, log=True)
            self.assertGreaterEqual(ll_real, ll)

    def test_MH_EP(self):
        
        n = 10000
        real_params = np.array([0.05, 2.5, 3.])
        x = rEP(*real_params, size=n)
        
        n_samples = 1000
        p_init = np.array([1, 2, 2])
        samples = MetropolisHastings(n_samples, x, likelihoodEP, p_init)
        
#         plot_helper_EP(x, real_params, samples)
        
        est = np.mean(samples[-100:, :], axis=0)
        assert_almost_equal(est, real_params, decimal=1)


    def test_MH_SEP(self):
        
        n = 10000
        real_params = np.array([-0.8, 4, 1.5, 1.3])
        x = rSEP(*real_params, size=n)
        
        n_samples = 1000
        p_init = np.array([0, 1, .5, 1.6])
        samples = MetropolisHastings(n=n_samples,
                                     x=x,
                                     lik=likelihoodSEP,
                                     params_init=p_init,
                                     log_prior=None)
        
#         plot_helper_SEP(x, real_params, samples)
        
        est = np.mean(samples[-100:, :], axis=0)
        assert_almost_equal(est, real_params, decimal=1)


def plot_helper_EP(x, real_params, samples):
    
    _, bins, _ = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.75)
    plt.plot(bins, dEP(bins, *real_params))
    plt.plot(bins, dEP(bins, *samples[-1, :]), '--')
    plt.legend(["real", "estimated"])
    plt.title("Metropolis-Hastings for EP.")
    plt.show()
    
    plt.plot(samples)
    plt.plot(samples)
    plt.axhline(real_params[0], ls='--', color='black')
    plt.axhline(real_params[1], ls='--', color='black')
    plt.axhline(real_params[2], ls='--', color='black')
    plt.title("Convergence of samples (EP)")
    plt.show()


def plot_helper_SEP(x, real_params, samples):
    
    _, bins, _ = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.75)
    
    plt.plot(bins, dSEP(bins, *real_params))
    plt.plot(bins, dSEP(bins, *samples[-1, :]), '--')
    plt.legend(["real", "estimated"])
    plt.title("Metropolis-Hastings for SEP.")
    plt.show()
 
    plt.plot(samples)
    plt.axhline(real_params[0], ls='--', color='black')
    plt.axhline(real_params[1], ls='--', color='black')
    plt.axhline(real_params[2], ls='--', color='black')
    plt.axhline(real_params[3], ls='--', color='black')
    plt.title("Convergence of samples (SEP)")
    plt.show()


if __name__ == "__main__":
        #import sys;sys.argv = ['', 'Test.testName']
        unittest.main()