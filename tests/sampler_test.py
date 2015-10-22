""" Testing the sampler. """

import os
import logging
mlogger = logging.getLogger('bnpp')
#mlogger.setLevel(logging.INFO)

import mock
import unittest

import scipy.stats

from qmmc.structure import Beta, Quote
from qmmc.market import GaussianQuoteMarket
from qmmc.sampler import BNPPGibbsSampler


class SlowTestGaussian(unittest.TestCase):
 
  def setUp(self):
    alpha=(0, 2.)
    beta=[(-2, 2.), (-1, 2.), (-.5, 2.), (0, 2.), (.5, 2.), (1, 2.), (2, 2.)]
    market = GaussianQuoteMarket(alpha, beta, nb_covar=1)
    market.simulate(nb_trade=2000)
    alpha, beta = market.get_params()
    sampler = BNPPGibbsSampler(alpha, beta)
     
    self.sampler = sampler
    self.market = market
 
  def tearDown(self):
      pass

  def learning(self, n_samples=300, savedir='./results/'):
    savedir = savedir
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    # Randomize a bit more initial state of latent variables
    alpha, beta = self.market.get_params()
    for _ in range(10):
      for quote in alpha.quotes:
        quote.sample_W()
        quote.sample_V()
    # Learn
    sampler = BNPPGibbsSampler(alpha, beta)
    sampler.sample(n_samples)
    sampler.save(savedir=savedir)
  
  @unittest.skip("Slow.")
  def test_learning_all(self):
    savedir = "./res_learning_all/"
    alpha_init = (0, 3)
    beta_init = [(0, 3)] * 7
    self.market.set_params(alpha_init, beta_init)
    self.learning(n_samples=500, savedir=savedir)
 
  @mock.patch.object(Beta, 'sample_sigma')
  @mock.patch.object(Quote, 'sample_W')
  @unittest.skip("Cython.")
  def test_learn_mu(self, msample_sigma, msample_W):
    savedir = "./res_learning_mu/"
    # Initialize with right sigma
    alpha_init = (0, 2)
    beta_init = [(3, 2), (-3, 2), (2, 2), (2, 2), (-1, 2), (3, 2), (-1, 2)]
    self.market.set_params(alpha_init, beta_init)
    self.learning(n_samples=200, savedir=savedir)
     
  @mock.patch.object(Beta, 'sample_mu')
  @mock.patch.object(Quote, 'sample_W')
  @unittest.skip("Slow.")
  def test_learn_sigma(self, msample_sigma, msample_W):
    savedir = "./res_learning_sigma/"
    # Initialize with right mu 
    alpha_init=(0, 2)
    beta_init=[(-2, 3), (-1, 1), (-.5, 3), (0, 4), (.5, 2.5), (1, 3), (2, 1.5)]
    self.market.set_params(alpha_init, beta_init)
    self.learning(n_samples=200, savedir=savedir)
    
  def stability(self, savedir):
    """ Sample all variables from fixed point."""
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    self.market.save(savedir)
    sampler = self.sampler
    n_samples = 100
    sampler.sample(n_samples)
    sampler.save(savedir=savedir)
  
  @unittest.skip("Slow.")
  def test_stability(self):
    savedir = "./res_joint_stability/"
    self.stability(savedir)
   
  @mock.patch.object(Beta, 'sample_sigma')
  @mock.patch.object(Quote, 'sample_W')
  @unittest.skip("Cython.")
  def test_mu_stability(self, msample_sigma, msample_W):
    savedir = "./res_mu_stability/"
    self.stability(savedir)
     
  @mock.patch.object(Beta, 'sample_mu')
  @mock.patch.object(Quote, 'sample_W')
  @unittest.skip("Cython.")
  def test_sigma_stability(self, msample_mu, msample_W):
    savedir = "./res_sigma_stability/"
    self.stability(savedir)



class FastTest(unittest.TestCase):
    
  def setUp(self):
    alpha=(-1, 1.5)
    beta=[(-2, 2), (-1, 2), (-.5, 2), (0, 2), (.5, 2), (1, 2), (4, 2)]
    market = GaussianQuoteMarket(alpha, beta)
    market.alpha.lr.set_params([1, 0, 1])
    market.beta[1].lr.set_params([-1, 1, 1])
    market.simulate(nb_trade=7000)
    alpha, beta = market.get_params()
    sampler = BNPPGibbsSampler(alpha, beta)
      
    self.sampler = sampler
    self.market = market
    
  def test_surface(self):
    import matplotlib.pyplot as plt
    sampler = self.sampler
    beta = sampler.beta[7]
    mul = []
    sigmal = []
    for _ in xrange(1000):
      beta.sample_mu()
      mul.append(beta.mu)
      beta.sample_sigma()
      sigmal.append(beta.sigma)
    plt.scatter(mul, sigmal)
    plt.show()
    plt.close()
    
  def test_sample_V(self):
    import matplotlib.pyplot as plt
    import numpy as np
    sampler = self.sampler
    for _ in xrange(10):
      c = 7  # Chosen arbitrarily
      for quote in sampler.beta[c].quotes:
        quote.sample_V()
    Vl = []
    for quote in sampler.beta[c].quotes:
      Vl.append((quote.sample_V() - quote.CBBT) / 
                quote.bid2mid - sampler.beta[c].lr.predict(quote.z))
    delta = 3 * np.sqrt(np.var(Vl)) / np.sqrt(len(sampler.beta[c].quotes))
    self.assertAlmostEqual(np.mean(Vl),
                           sampler.beta[c].mu,
                           delta=delta)
    self.assertAlmostEqual(np.sqrt(np.var(Vl)),
                           sampler.beta[c].sigma,
                           delta=delta)
    _, bins, _ = plt.hist(Vl, 50, normed=1)
    plt.plot(bins, scipy.stats.norm.pdf(bins, sampler.beta[c].mu,
                                        sampler.beta[c].sigma), 'r')
    plt.title("Sample V")
    plt.show()
    plt.close()
    
  def test_sample_W(self):
    import matplotlib.pyplot as plt
    import numpy as np
    sampler = self.sampler
    for _ in xrange(10):
      for quote in sampler.alpha.quotes:
        quote.sample_W()
    Wl = []
    for quote in sampler.alpha.quotes:
      Wl.append((quote.sample_W()[0] - quote.CBBT) / 
                quote.bid2mid - sampler.alpha.lr.predict(quote.z))  
    delta = 3 * np.sqrt(np.var(Wl)) / np.sqrt(len(sampler.alpha.quotes))
    self.assertAlmostEqual(np.mean(Wl),
                           sampler.alpha.mu,
                           delta=delta)
    self.assertAlmostEqual(np.sqrt(np.var(Wl)),
                           sampler.alpha.sigma,
                           delta=delta)
    _, bins, _ = plt.hist(Wl, 50, normed=1)
    plt.plot(bins, scipy.stats.norm.pdf(bins, sampler.alpha.mu,
                                        sampler.alpha.sigma), 'r')
    plt.title("Sample W")
    plt.show()
    plt.close()
    

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()
  