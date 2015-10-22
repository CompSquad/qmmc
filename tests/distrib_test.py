import unittest

from scipy.stats import norm, laplace
import numpy as np

from qmmc.distrib import *

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
      
  @unittest.skip("Deprecated.")   
  def test_loglikEP(self):
    p = np.array([0, 1.3, 1.5])
    y = rEP(*p, size=100000)
    ll_real = likelihoodEP(y, *p, log=True)
    for _ in range(100):
      params = p + .1 * np.random.randn(3)
      ll = likelihoodEP(y, *params, log=True)
      self.assertGreaterEqual(ll_real, ll)

  @unittest.skip("Deprecated.") 
  def test_MH_EP(self):
    import matplotlib.pyplot as plt
    
    n = 10000
    mu, alpha, beta = 0.05, 3.12, 3.96
    x = rEP(mu=mu, alpha=alpha, beta=beta, size=n)
    n, bins, _ = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.75)
    plt.plot(bins, dEP(bins, mu, alpha, beta))
    #plt.plot(bins, dEPa(bins, 0, 3, 3.5))
    
    n_samples = 1000
    p_init = [1, 2, 2]
    samples = MetropolisHastings(n_samples, x, likelihoodEP, p_init)
    
    plt.plot(bins, dEP(bins, *samples[-1, :]), '--')
    plt.legend(["real", "estimated"])
    plt.title("Metropolis-Hastings for EP.")
    plt.show()
    #print samples[-1, :]
   
    plt.plot(samples)
    plt.plot(samples)
    plt.axhline(mu, ls='--', color='black')
    plt.axhline(beta, ls='--', color='black')
    plt.axhline(alpha, ls='--', color='black')
    plt.title("Convergence of samples (EP)")
    plt.show()

  @unittest.skip("Graphical debugging.") 
  def test_MH_SEP(self):
    import matplotlib.pyplot as plt
    
    n = 10000
    mu, sigma, beta, alpha = -0.8, 4, 1.5, 1.3
    x = rSEP(mu=mu, sigma=sigma, beta=beta, alpha=alpha, size=n)
    _, bins, _ = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.75)
    plt.plot(bins, dSEP(bins, mu, sigma, beta, alpha))
    
    n_samples = 1000
    p_init = [0, 1, .5, 1.6]
    samples = MetropolisHastings(n = n_samples,
                                 x = x, 
                                 lik = likelihoodSEP,
                                 params_init = p_init, 
                                 log_prior=None)
    
    plt.plot(bins, dSEP(bins, *samples[-1, :]), '--')
    plt.legend(["real", "estimated"])
    plt.title("Metropolis-Hastings for SEP.")
    plt.show()
    #print "Real = %s" % str([mu, sigma, beta, alpha])
    #print "Est = %s" % str(samples[-1, :])
   
    plt.plot(samples)
    plt.axhline(mu, ls='--', color='black')
    plt.axhline(sigma, ls='--', color='black')
    plt.axhline(beta, ls='--', color='black')
    plt.axhline(alpha, ls='--', color='black')
    plt.title("Convergence of samples (SEP)")
    plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()