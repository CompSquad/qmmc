""" Test market class. """

import unittest
import mock

import numpy as np

from qmmc.market import GaussianQuoteMarket

class Test(unittest.TestCase):

  def setUp(self):
    self.alpha=(0, 1.)
    self.beta=[(-2, .2), (-1, .2), (-.5, .2), (0, .2), (.5, .2), (1, .2), (2, .2)]
    self.market = GaussianQuoteMarket(self.alpha, self.beta)

  def tearDown(self):
    pass

  def test_init_gaussian(self):
    alpha=(0, 1.)
    beta=[(-2, 2), (-1, 2), (-.5, 1), (0, 3), (.5, 1), (1, 2), (2, .5)]
    market = GaussianQuoteMarket(alpha, beta)
    self.assertEqual(len(beta), market.nb_customers)
    malpha = (market.alpha.mu, market.alpha.sigma)
    self.assertAlmostEqual(alpha, malpha)
    mbeta = [tuple(b.get_params().values()) for b in market.beta.itervalues()]
    self.assertAlmostEqual(beta, mbeta)

    
  def test_simulate(self):
    market = self.market
    market.nb_covar = 3
    n_samples = 20000
    market.simulate(nb_trade=n_samples)
    quotes = market.quotes
    self.assertEqual(len(quotes), n_samples)
    dW = []
    for quote in quotes:
      dW.extend(quote.dW() - market.alpha.lr.predict(quote.z))
    mu = np.mean(dW)
    sigma = np.sqrt(np.var(dW))
    delta = np.var(dW) / np.sqrt(n_samples)
    self.assertAlmostEqual(mu, self.alpha[0], delta=3*delta)
    self.assertAlmostEqual(sigma, self.alpha[1], delta=3*delta)
     
    dY = [(quote.Y - quote.CBBT) / quote.bid2mid for quote in quotes]
    self.assertAlmostEqual(np.mean(dY), np.mean(dW), delta=6*delta)
    self.assertAlmostEqual(np.var(dY), np.var(dW), delta=6*delta)
     
    for b in market.beta.itervalues():
      quotes = b.quotes
      dV = [quote.dV() - market.beta[1].lr.predict(quote.z) for quote in quotes]
      mu = np.mean(dV)
      sigma = np.sqrt(np.var(dV))
      self.assertAlmostEqual(sigma, b.sigma, delta=.01)
      self.assertAlmostEqual(mu, b.mu, delta=.01)
      
    z = [quote.z for quote in market.quotes]
    delta = 3. / np.sqrt(len(quotes))
    self.assertAlmostEqual(np.mean(z, axis=0)[0], 1., delta=delta)

   
  @mock.patch.object(GaussianQuoteMarket, 'nb_dealers', return_value=4)   
  def test_dealers_competition(self, nb_dealers):
    self.market.simulate(nb_trade=20000)
    dn = len([quote for quote in self.market.quotes if quote.I == "Done"])
    ta = len([quote for quote in self.market.quotes if quote.I == "TradedAway"])
    self.assertAlmostEqual(float(ta) / dn, self.market.nb_dealers(), delta=.2)
    
  def test_alignement(self):
    alpha = (0, 1)
    beta = [(5, 1)]
    market = GaussianQuoteMarket(alpha, beta)
    market.simulate(nb_trade=20000)
    nt = len([quote for quote in market.quotes if quote.I == "NotTraded"])
    self.assertAlmostEqual(nt, 0, 10)
     
    alpha = (0, 1)
    beta = [(-6, 1)]
    market = GaussianQuoteMarket(alpha, beta)
    market.simulate(nb_trade=20000)
    dn = len([quote for quote in market.quotes if quote.I == "Done"])
    self.assertAlmostEqual(dn, 0, delta=10)

  def test_save_load(self):
    alpha_init = (0, 2)
    beta_init = [(-1, 2), (1, 2)]
    market = GaussianQuoteMarket(alpha_init, beta_init, nb_covar=3)
    market.simulate(nb_trade=4000)
    market.save()
    
    covar = {"z0": {'type': "num"}, "z1": {'type': "num"}, "z2": {'type':"num"}}
    lmarket = GaussianQuoteMarket.load('fake-market.csv', covar=covar)
    for quote, lquote in zip(market.quotes, lmarket.quotes):
      self.assertAlmostEqual(quote.Y, lquote.Y)
      np.testing.assert_array_almost_equal(quote.z, lquote.z)
      self.assertAlmostEqual(quote.C, lquote.C)
      self.assertAlmostEqual(quote.nb_dealers, lquote.nb_dealers)
  
if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testName']
  unittest.main()