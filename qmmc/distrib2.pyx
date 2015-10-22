""" Sampling from a SE distributions and its posterior. """

# cython: profile=True

import numpy as np
cimport numpy as np
import scipy.stats 
from scipy.special import gamma, gammaincinv

def dnorm(x, mu, sigma, params):
  return scipy.stats.norm.pdf(x, mu, sigma)

cpdef inline float dEP(
    float x, float mu=0, float alpha=1, float beta=1, bint log=False):
  if log == False:
    return ((beta / (2 * alpha * gamma(1. / beta))) * 
            np.exp(-(np.abs(x - mu) / alpha)**beta))
  else:
    return np.log((beta / (2 * alpha * gamma(1. / beta))) *
                   np.exp(-(np.abs(x - mu) / alpha)**beta))
    
cpdef inline np.ndarray dEPa(
    np.ndarray x, float mu=0, float alpha=1, float beta=1, bint log=False):
  if log == False:
    return ((beta / (2 * alpha * gamma(1. / beta))) * 
            np.exp(-(np.abs(x - mu) / alpha)**beta))
  else:
    return (np.log((beta / (2 * alpha * gamma(1. / beta)))) -
            (np.abs(x - mu) / alpha)**beta)

cpdef inline float rEP(float mu=0, float alpha=1, float beta=1, bint size=1):
  u = np.random.rand()
  z = 2 *  np.abs(u - 1. / 2)
  z = gammaincinv(1. / beta, z)
  y = mu + np.sign(u - 1. / 2) * alpha * z**(1. / beta)
  return y

cpdef inline np.ndarray rEPa(
    float mu=0, float alpha=1, float beta=1, int size=1):
  u = np.random.rand(size)
  z = 2 *  np.abs(u - 1. / 2)
  z = gammaincinv(1. / beta, z)
  y = mu + np.sign(u - 1. / 2) * alpha * z**(1. / beta)
  return y

def rEP2(mu, sigma, alpha, size=1):
  mu, sigma, alpha = float(mu), float(sigma), float(alpha)
  if size == 1:
    u = np.random.rand()
    b = np.random.beta(1. / alpha, 1 - 1. / alpha)
    r = np.sign(np.random.rand() - .5)
  else:
    u = np.random.rand(size)
    b = np.random.beta(1. / alpha, 1 - 1. / alpha, size=size)
    r = np.sign(np.random.rand(size) - .5)
  return r * (-alpha * b * np.log(u))**(1. / alpha)

def dEP2(x, mu=0, sigma=1, alpha=2, log=False):
  z = (x - mu) / sigma
  c = 2 * alpha**(1. / alpha - 1) * gamma(1. / alpha)
  d = np.exp(-np.abs(z)**alpha / alpha) / (sigma * c)
  if log == False:
    return d
  else:
    return np.log(d)

@np.vectorize
def erfcc(x):
  """Complementary error function."""
  z = abs(x)
  t = 1. / (1. + 0.5*z)
  r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
    t*(.09678418+t*(-.18628806+t*(.27886807+
    t*(-1.13520398+t*(1.48851587+t*(-.82215223+
    t*.17087277)))))))))
  if x >= 0.:
    return r
  else:
    return 2. - r

def pnorm(x):
  return 1. - 0.5*erfcc(x/(2**0.5))

def rSEP(mu=0, sigma=1, beta=0, alpha=2, size=1):
  mu, sigma, beta, alpha = float(mu), float(sigma), float(beta), float(alpha)
  y = rEP2(0, 1, alpha, size=size)
  w = np.sign(y) * np.abs(y)**(alpha / 2) * beta * np.sqrt(2. / alpha)
  if size == 1:
    r = - np.sign(np.random.rand() - scipy.stats.norm.cdf(w))
  else:
    r = - np.sign(np.random.rand(size) - scipy.stats.norm.cdf(w))
  z = r * y
  return mu + sigma * z

def dSEP(x, mu=0., sigma=1., beta=0, alpha=2, log=False):
  mu, sigma, beta, alpha = float(mu), float(sigma), float(beta), float(alpha)
  z = (x - mu) / sigma
  w = np.sign(z) * np.abs(z)**(alpha / 2) * beta * np.sqrt(2. / alpha)
  # Note: There is a sigma division in the paper
  x = 2 * scipy.stats.norm.cdf(w) * dEP2(x, mu, sigma, alpha)
  if log == False:
    return x
  else:
    return np.log(x)
   
def likelihoodEP(
    np.ndarray x, float mu, float alpha, float beta, int log=False):
  """ Likelihood of a sample according to an SE dist."""
  if log == False:
    return np.prod(dEPa(x, mu, alpha, beta))
  else:
    return np.sum(dEPa(x, mu, alpha, beta, log=True))

def likelihoodEP2(x, mu, alpha, beta, log=False):
  """ Likelihood of a sample according to an SE dist."""
  if log == False:
    return np.prod(dEP2(x, mu, alpha, beta))
  else:
    return np.sum(dEP2(x, mu, alpha, beta, log=True))

def likelihoodSEP(x, mu, sigma, beta, alpha, log=False):
  """ Likelihood of a sample according to an SEP dist."""
  if log == False:
    return np.prod(dSEP(x, mu, sigma, beta, alpha))
  else:
    return np.sum(dSEP(x, mu, sigma, beta, alpha, log=True))

cpdef float log_laplace_prior(np.ndarray params, float scale=3):
  cdef float x
  cdef float ll = 0
  for x in params:
    ll += x / scale - np.log(2 * scale)
  return ll

cpdef np.ndarray MetropolisHastings(
    int n, np.ndarray x, object lik, list params_init, object log_prior=None):
  """ Estimate posterior using Metropolis-Hastings."""
  cdef int ndim = len(params_init)
  cdef np.ndarray samples = np.zeros((n, ndim))
  cdef np.ndarray ratios = np.zeros((n, 1))
  cdef float rate = 0.
  
  par_prev = np.array(params_init)
  cdef float ll_prev = lik(x, *par_prev, log=True)
  cdef float ll_cur, ratio
  cdef int i
  for i in xrange(n):
    par_cur = par_prev + .01 * np.random.randn(ndim)
    ll_cur = lik(x, *par_cur, log=True)
    if log_prior is not None:
      ll_prev += log_prior(par_prev)
      ll_cur += log_prior(par_cur)
    ratio = ll_cur - ll_prev
    if np.log(np.random.rand()) < ratio:
      par_prev = par_cur
      ll_prev = ll_cur
      rate += 1
    samples[i] = par_prev
    ratios[i] = ratio
  rate /= n
  
  return samples