""" Function for computing the density and sampling from an EP or SEP.

There are two types of EP parameterization implemented. The first one (`dEP`
and `rEP` is consistent with the description on Wikipedia and is used to sample
from an EP. The second one (`dEP2` and `rEP2`) follow the parameterization 
described in ::Ref:: and is used as a subroutine to sample from an SEP. """

__author__ = "arnaud.rachez@gmail.com"

import numpy as np
import scipy.stats
from scipy.special import gamma, gammaincinv


def dEP(x, mu=0, alpha=1, beta=1, log=False):
    r""" Exponential Power (EP) density function.

    See http://en.wikipedia.org/wiki/Generalized_normal_distribution. The density 
    is written :math:`f_{EP} = e^{-\frac{\vert|x-\mu\vert|^{\beta}}{\alpha}}`
    
    Arguments:
        x (numpy.array): A vector of samples.
        mu (float): Mean.
        alpha (float): Scale parameter
        beta (float): Shape
    
    Returns:
        An array of densities.
    """
    mu, alpha, beta = float(mu), float(alpha), float(beta)
    if log == False:
        return ((beta / (2 * alpha * gamma(1. / beta))) * 
                        np.exp(-(np.abs(x - mu) / alpha)**beta))
    else:
        return np.log((beta / (2 * alpha * gamma(1. / beta))) *
                                     np.exp(-(np.abs(x - mu) / alpha)**beta))

def rEP(mu=0, alpha=1, beta=1, size=1):
    r""" Samples from an Exponential Power (EP) distribution.

    The parameterization used here is consistent with the `qmmc.distrib.dEP`
    function.
    
    Arguments:
        mu (float): Mean.
        alpha (float): Scale parameter
        beta (float): Shape
        size (int): The number of samples to return.
        
    Returns:
        An array of samples.
    """
    if size == 1:
        u = np.random.rand()
    else:
        u = np.random.rand(size)
    z = 2 *    np.abs(u - 1. / 2)
    z = gammaincinv(1. / beta, z)
    y = mu + np.sign(u - 1. / 2) * alpha * z**(1. / beta)
    return y

def rEP2(mu, sigma, alpha, size=1):
    r""" Samples from an Exponential Power (EP) distribution.

    The parameterization used here is consistent with the `qmmc.distrib.dEP2`
    function.
    
    Arguments:
        mu (float): Mean.
        alpha (float): Scale parameter
        beta (float): Shape
        size (int): The number of samples to return.
        
    Returns:
        An array of samlpes.
    """
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
    r""" Exponential Power (EP) density function. 
    
    The parameterization chosen here is consistent with ::Ref::.
    
    The form of the density is :math:`f_{EP} = \frac{1}{c\sigma}e^{-\frac{|z|^{\alpha}}{\alpha}}`
    where :math:`z = \frac{(x - \mu)}{\sigma}` and :math:`c = 2\alpha^{1/{\alpha}-1}\Gamma(1/\alpha)`.
    
    Arguments:
        x (numpy.array): A vector of samples.
        mu (float): Mean.
        alpha (float): Scale parameter
        beta (float): Shape
        log (bool): Whether to return the logarithm of the density.
    
    Returns:
        An array of densities.
    """
    z = (x - mu) / sigma
    c = 2 * alpha**(1. / alpha - 1) * gamma(1. / alpha)
    d = np.exp(-np.abs(z)**alpha / alpha) / (sigma * c)
    if log == False:
        return d
    else:
        return np.log(d)

def rSEP(mu=0, sigma=1, beta=0, alpha=2, size=1):
    """ Samples from a Skew Exponentional Power distribution (SEP).
    
    The density of the SEP is :math:`f_{SEP} = 2\Phi(w)f_{EP}(x, \mu, \sigma, \alpha)`
    
    Arguments:
        mu (float): Mode
        sigma (float): Sigma
        alpha (float): Alpha
        beta (float): Beta
        size (int): Number of samples to draw.
    
    Returns:
        An array of samples from an SEP.
    """
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
    """ Density function of a Skew Exponentional Power distribution (SEP).
    
    The density of the SEP is :math:`f_{SEP} = 2\Phi(w)f_{EP}(x, \mu, \sigma, \alpha)`
    
    Arguments:
        x (numpy.array): An array of samples.
        mu (float): Mode
        sigma (float): Sigma
        alpha (float): Alpha
        beta (float): Beta
    
    Returns:
        An array of densities.
    """
    mu, sigma, beta, alpha = float(mu), float(sigma), float(beta), float(alpha)
    z = (x - mu) / sigma
    w = np.sign(z) * np.abs(z)**(alpha / 2) * beta * np.sqrt(2. / alpha)
    # Note: There is a sigma division in the paper
    x = 2 * scipy.stats.norm.cdf(w) * dEP2(x, mu, sigma, alpha)
    if log == False:
        return x
    else:
        return np.log(x)
     

def likelihoodEP(x, mu, alpha, beta, log=False):
    """ Likelihood of a sample according to an SE dist."""
    if log == False:
        return np.prod(dEP(x, mu, alpha, beta))
    else:
        return np.sum(dEP(x, mu, alpha, beta, log=True))
    
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
    
def MetropolisHastings(n, x, lik, params_init, log_prior=None):
    r""" Estimate posterior using Metropolis-Hastings.
    
    Currently, only the likelihood is passed to the MH procedure. If you would 
    like to include prior information on the parameters you should directly 
    include it in the `lik` parameter.
    
    Arguments:
        n (int): The number of iterations (samples) to perform.
        x (numpy.array): An array of samples
        lik (function): A pointer to the likelihood function.
        params_init (list): A list containing the initial parameter values.
    
    Returns:
        An array containing the sampled parameters at each iteration.
    """
    ndim = params_init.shape[0]
    samples = np.zeros((n, ndim))
    ratios = np.zeros((n, 1))
    rate = 0.
    f = lambda x, params, log: lik(x, *params, log=True)
    
    par_prev = params_init
    ll_prev = f(x, par_prev, log=True)
    
    for i in xrange(n):
        par_cur = par_prev + .05 * np.random.randn(ndim)
        ll_cur = f(x, par_cur, log=True)
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
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    mu, sigma, beta, alpha = 0, 15, 2, 1.6
    x = rSEP(mu, sigma, beta, alpha, size=100000)
    print x
    print dSEP(x, mu, sigma, beta, alpha)
    print np.sum(likelihoodSEP(x, mu, sigma, beta, alpha, log=True))
    print np.sum(np.log(dSEP(x, mu, sigma, beta, alpha)))
    _, bins, _ = plt.hist(x, 50, normed=True)
    plt.plot(bins, dSEP(bins, mu, sigma, beta, alpha), 'r--')
    plt.show()
