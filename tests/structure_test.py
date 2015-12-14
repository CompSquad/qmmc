import unittest

from scipy.stats import norm, laplace
import numpy as np

from qmmc.structure import Alpha, Beta, Quote

class Test(unittest.TestCase):

    def setUp(self):
        self.alpha = Alpha(0, 2)
        self.beta = Beta(0, 2)
        self.quote = Quote(qtype="Buy",
                           I="TradedAway",
                           J="CustAcceptQuote",
                           Y=115,
                           C=110,
                           CBBT=110.3,
                           bid2mid=.5,
                           customer=1,
                           nb_dealers=2,
                           alpha=self.alpha,
                           beta=self.beta)
    
    
    def tearDown(self):
            pass
    
    def test_init(self):
        alpha = Alpha(0, 1)
        beta = Beta(1, 3, 4)
        self.assertEqual(0, alpha.mu) 
        self.assertEqual(1, alpha.sigma) 
        self.assertEqual(3, beta.mu)
        self.assertEqual(4, beta.sigma)
    
    def test_sample_dV(self):
        beta = Beta(1, 3)
        n = 100000
        s = [beta.rvs() for _ in xrange(n)]
        mu = np.mean(s)
        sigma = np.sqrt(np.var(s))
        delta = np.var(s) / np.sqrt(100000)
        self.assertAlmostEqual(mu, beta.mu, delta=delta) 
        self.assertAlmostEqual(sigma, beta.sigma, delta=delta)
    
    def test_sample_dV_covar(self):
        beta = Beta(0, 3)
        w = np.random.randn(3)
        beta.lr.set_params(w)
        covar = np.random.randn(3)
        s = beta.rvs(size=10000, z=covar)
        delta = 3 * np.sqrt(beta.sigma) / np.sqrt(len(s))
        self.assertAlmostEqual(np.mean(s), beta.mu + np.dot(w, covar), delta=delta)
        
    def test_sample_dW_covar(self):
        alpha = Alpha(0, 3)
        w = np.random.randn(3)
        alpha.lr.set_params(w)
        covar = np.random.randn(3)
        s = alpha.rvs(size=10000, z=covar)
        delta = 3 * alpha.sigma / np.sqrt(len(s))
        self.assertAlmostEqual(np.mean(s), alpha.mu + np.dot(w, covar), delta=delta)
        
    def test_sample_dW(self):
        n = 100000
        s = [self.alpha.rvs() for _ in xrange(n)]
        mu = np.mean(s)
        sigma = np.sqrt(np.var(s))
        delta = np.var(s) / np.sqrt(100000)
        self.assertAlmostEqual(mu, self.alpha.mu, delta=delta) 
        self.assertAlmostEqual(sigma, self.alpha.sigma, delta=delta)
        
    def test_fg(self):
        quote = self.quote
        delta = (quote.Y - quote.CBBT) / quote.bid2mid
        self.alpha.pdf(delta)
        self.beta.pdf(delta)
        #print "f(%.2f) = %.2f" % (delta, self.alpha.f(delta))
        #print "g(%.2f) = %.2f" % (delta, self.beta.g(delta))
        
    def test_FG(self):
        int1 = self.alpha.cdf(np.inf)
        self.assertAlmostEqual(int1[0], 1, delta=int1[1])
        self.assertAlmostEqual(self.alpha.cdf(0)[0], .5, delta=int1[1])
        int2 = self.beta.cdf(np.inf)
        self.assertAlmostEqual(int2[0], 1, delta=int2[1])
    
    def test_predict(self):
        predS = self.quote.predictS(n_iter=100000)
        self.assertAlmostEqual(np.sum(predS.values()), 1.)
        predI = self.quote.predictI()
        self.assertAlmostEqual(np.sum(predI.values()), 1., delta=.01)
        self.assertAlmostEqual(predI["Done"], predS["Done"], delta=.01)
        self.assertAlmostEqual(predI["TradedAway"], predS["TradedAway"], delta=.01)
        self.assertAlmostEqual(predI["NotTraded"], predS["NotTraded"], delta=.01)
        #print "predS = %s" % str(predS), np.sum(predS.values())
        #print "predI = %s" % str(predI), np.sum(predI.values())
    
    def test_loglik(self):
        alpha = Alpha(1, 2)
        mu = alpha.mu
        sigma = alpha.sigma
        x = np.random.rand()
        l = np.exp(-1. / 2 * ((x - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
        ll = - 1. / 2 * ((x - mu) / sigma)**2 - np.log(np.sqrt(2 * np.pi) * sigma)
        self.assertAlmostEqual(np.log(l), ll)
        self.assertAlmostEqual(l, norm.pdf(x, loc=mu, scale=sigma))


if __name__ == "__main__":
        #import sys;sys.argv = ['', 'Test.testName']
        unittest.main()