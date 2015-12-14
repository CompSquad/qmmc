""" Main data abstractions for the structured model. """

__author__ = "arnaud.rachez@gmail.com"

import logging

from math import factorial

import numpy as np
import scipy.stats 
import scipy.integrate

from qmmc.distrib import    dEP, dEP, rEP, dSEP, rSEP
from qmmc.distrib import likelihoodEP, likelihoodSEP
from qmmc.distrib import MetropolisHastings

import warnings

def deprecated(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emmitted
        when the function is used."""
        def newFunc(*args, **kwargs):
                warnings.warn("Call to deprecated function %s." % func.__name__,
                                            category=DeprecationWarning)
                return func(*args, **kwargs)
        newFunc.__name__ = func.__name__
        newFunc.__doc__ = func.__doc__
        newFunc.__dict__.update(func.__dict__)
        return newFunc


class Error(Exception):
    """ Base class for handling Errors. """
    pass


class CaughtInLoop(Error):
    """ Rejection sampling is failing. """
    pass

  
class BadInitialization(Error):
    """ Assignement to initial hidden values is impossible. """
    pass


class Value:
    
    def __init__(self, value):
        
        self.value = value

    def sample(self):
        
        return self.value
    
    def logp(self):
        
        return 0.


class Uniform:
    
    def __init__(self, value=None):
        
        self.value = value
    
    def sample(self):
        
        self.value = np.random.rand()
        return self.value
    
    def logp(self):
        
        return 0


class TruncatedBinomial:
    
    def __init__(self, p, k, value=None):
        
        self.p = p
        self.k = k
        self.value = value
    
    def sample(self):
        
        self.value = self._sample(self.p.value, self.k.value)
        return self.value
    
    def logp(self):
        
        return self._logp(self.value, self.p.value, self.k.value)
    
    def _sample(self, p, k):
        
        return np.random.binomial(k, p)
    
    def _logp(self, value, p, k):
        
        prob = (p ** value) * ((1 - p) ** (k - value))
        prob *= factorial(k) / (factorial(value) * factorial(k - value))
        
        return np.log(prob)


class BayesianLinearRegression:
    """ This is a linear regression learned using Metropolis-Hastings.
    
    Attributes:
        w (numpy.array): Set of parameters of the linear regression.
        params (list): A list of child parameter classes to keep track of the 
            global structure of the model.
        history (numpy.array): An array containing all the sampled parameters.
        """
    
    def __init__(self):
        self.w = None
        self.params = []
        self.history = None
    
    def __str__(self):
        return "input_dim = %d, w = %s" % (self.input_dim, str(self.w))
    
    def post_init(self, input_dim):
        """ This function will be called when the first co-variable is encountered. 
        
        It allows to set the dimensionality of the regression automatically.
        
        Arguments:
            input_dim (int): Dimensionality of the input.
        """
        self.w = np.zeros(input_dim)
        self.history = np.array([self.w])
        
    def predict(self, x):
        """ Compute the scalar product between :math:`x` and :math:`w`. """
        # When there is no need for co-variables simply return 0. to be consistent. 
        # This is a hack allowing to keep the linear regression semantic even in the
        # absence of any real co-variables.
        if x is None:
            return 0.
        
        # We do not initialize the dimensionality a priori and thus must do it when
        # computing the linear regression for the first time. The first 
        # co-variable encountered determines the dimensionality of the regression.
        if self.w is None:
            self.post_init(len(x))
        return np.dot(self.w, x)
    
    def set_params(self, w, reset=False):
        """ Allows to manually set parameters and reset the history of samples.
        
        Arguments:
            w (list): A list of real numbers of length `self.input_dim`
            reset (bool): Whether or not to reset the history of samples.
        """
        self.w = np.array(w)
        if reset == True or self.history is None:
            self.history = np.array([self.w])

    def update_params(self):
        """ Updates the parameters using Metropolis-Hastings.
        
        The update of the parameters must be called when there are actual 
        co-variables in the model. It is left to the user to specify it in the 
        :method:`sampler.BNPPGibbsSampler.sample` method.
        """
        # Since the update of parameters for the linear regression is expensive, the 
        # number of iterations of MH is currently hard coded to 10. Moreover, due to 
        # the signature of the MetropolisHasting function we provide dummy `x` 
        # variables. These will not be used in practice since the likelihood does 
        # depend on them.
        #
        # TODO: Make all this mess a bit clearer.
        niter = 10
        x = np.array([1])
        params_init = np.array(self.w)
        samples = MetropolisHastings(niter, x, self.loglik, params_init=params_init)
        self.w = samples[-1]
        self.history = np.vstack((self.history, self.w))
    
    def loglik(self, x=0, *argv, **argc):
        """ The log-likelihood of the quotes. """        
        # The likelihood is computed given the current set of parameters for
        # `structure.Alpha`, `structure.Beta` and `structure.BayesianLinearRegression`.
        # Since it does not accept any parameters, it forces the MetropolisHastings 
        # call to be subverted.
        # (see `structure.BayesianLogisticRegression.update_params()`)
        ll = 0
        w_prev = self.w if self.w is None else np.array(self.w)
        self.w = np.array(argv) if argv is not None else None
        assert(w_prev is not self.w)
        for param in self.params:
            ll += param.loglik()
        self.w = w_prev
        return ll

class BaseParams(object):
    """ This class is necessary for polymorphism with Cython.
    
    Note:
        Cython does not seem to be compatible with abstract classes in python."""

class Alpha(BaseParams):
    """ Parameters for the :math:`f` distribution on the :math:`W_k`.

    In this version the distribution is a gaussian with mean :math:`\mu`
    and standard deviation :math:`\sigma`. All the dealers share the same set of 
    parameters and thus there should only be one instance of this class.
    
    Parameters:
        mu (float): The mean for the distribution on the :math:`W_k`.
        sigma (float): The standard deviation.
        
    Attributes:
        quotes (list of structure.Quotes): All the quotes on which the instance has 
            an influence.
        lr (structure.BayesianLinearRegression): The associated linear regression on
            the co-variables.
        history (dict): A list of the sample history of parameters.
    """
    
    def __init__(self, mu=0, sigma=3):
        self.mu = mu
        self.sigma = sigma
        self.lr = BayesianLinearRegression()
        self.lr.params.append(self)
        self.quotes = []
        self.history = {"mu": [self.mu], "sigma": [self.sigma]}
        
    def __str__(self):
        params = {"mu": self.mu, "sigma": self.sigma, "nb_quotes": len(self.quotes)}
        return ("Alpha: mu = %(mu).2f, sigma = %(sigma).2f, nb_quotes = %(nb_quotes)s" 
                        % params)
    
    def set_params(self, mu, sigma=2):
        """ Sets a hard value for :math:`\mu`, :math:`\sigma` and :math:`w`."""
        self.mu = mu
        self.sigma = sigma
    
    def get_params(self):
        """ Returns a disctionary containing parameter values. """
        return {"mu": self.mu, "sigma": self.sigma}

    def update_params(self):
        """ Method to update the value of :math:`\mu` and :math:`\sigma`.
        
        The update is realised by sampling new values for the parameters based on
        the full conditionals.
        """
        if len(self.quotes) > 0:
            self.sample_mu()
            self.sample_sigma()
            self.history["mu"].append(self.mu)
            self.history["sigma"].append(self.sigma)
    
    def pdf(self, delta, z=None):
        """ Density function. """
        bias = self.lr.predict(z)
        return scipy.stats.norm.pdf(delta, self.mu + bias, self.sigma)
    
    def cdf(self, delta, z=None):
        """ Cumulative distribution. """
        f = lambda delta: self.pdf(delta, z=z)
        return scipy.integrate.quad(f, -np.inf, delta)
    
    def rvs(self, size=1, z=None):
        """ Samples :math:`\delta` from :math:`f`. """
        bias = self.lr.predict(z)
        # Below is a small hack for optimization since it appears that the 
        # broadcasting of operations in numpy using the keyword size slows down the
        # sampling quite a bit.
        if size == 1:
            return self.sigma * np.random.randn() +    self.mu + bias
        else:
            return self.sigma * np.random.randn(size) +    self.mu + bias
    
    def sample_mu(self):
        """ Sample self.mu. """
        dW = []
        ndW = 0.
        for quote in self.quotes:
            dW.extend(quote.dW() - self.lr.predict(quote.z))
            ndW += len(quote.dW())
        
        mu_prior = 0
        sigma_prior = 10
        mu = ((mu_prior / sigma_prior**2 + np.sum(dW) / self.sigma**2) / 
                    (1. / sigma_prior**2 + ndW / self.sigma**2))
        sigma = np.sqrt(1. / (1. / sigma_prior**2 + ndW / self.sigma**2))
        self.mu = sigma * np.random.randn(1)[0] + mu

    def sample_sigma(self):
        """ Sample self.sigma. """            
        dbar = 0
        ndW = 0.
        for quote in self.quotes:
            a = (quote.dW() - self.lr.predict(quote.z) - self.mu)**2
            dbar += np.sum(a)
            ndW += len(quote.dW())
        
        scale_prior = 3
        shape_prior = 1
        scale = scale_prior + dbar / 2
        shape = shape_prior + ndW / 2
        self.sigma = np.sqrt(
                scipy.stats.invgamma.rvs(shape, scale=scale, size=1)[0])
    
    def loglik(self):
        """ Log-likelihood function. """
        x = []
        for quote in self.quotes:
            bias = self.lr.predict(quote.z)
            for dw in quote.dW():
                x.append(dw - bias)
        x = np.array(x)
        ll = np.sum(- 1. / 2 * ((x - self.mu) / self.sigma)**2 - 
                                np.log(np.sqrt(2 * np.pi) * self.sigma))
        return ll

class Beta(Alpha):
    """ Parameters for the :math:`g` distribution on the :math:`V`.
    
    Parameters:
        customer (int): The client id.
        mu (float): The mean for the distribution on the :math:`V`.
        sigma (float): The standard deviation.
        
    Attributes:
        quotes (list of structure.Quotes): All the quotes on which the instance has 
            an influence.
        lr (structure.BayesianLinearRegression): The associated linear regression on
            the co-variables.
        history (dict): A list of the sample history of parameters.
    
    """
    
    def __init__(self, customer=0, mu=0, sigma=3):
        super(Beta, self).__init__(mu=mu, sigma=sigma)
        self.customer = customer
        
    def __str__(self):
        params = {"mu": self.mu, "sigma": self.sigma, "id": self.customer,
                            "nb_quotes": len(self.quotes)}
        return ("Beta_%(id)d: mu = %(mu).2f, sigma = %(sigma).2f, id = %(id)d, "
                        "nb_quotes = %(nb_quotes)s" % params)
     
    def sample_mu(self):
        """ Sample self.mu. 
        
        Note:
            Uses a randomly chosen prior :math:`(mu, sigma) = (0, 3)`.
        """
        dV = []
        for quote in self.quotes:
            dV.append(quote.dV() - self.lr.predict(quote.z))
        ndV = len(self.quotes)
        
        mu_prior = 0
        sigma_prior = 3
        mu = ((mu_prior / sigma_prior**2 + np.sum(dV) / self.sigma**2) / 
                    (1. / sigma_prior**2 + ndV / self.sigma**2))
        sigma = np.sqrt(1. / (1. / sigma_prior**2 + ndV / self.sigma**2))
        self.mu = sigma * np.random.randn() + mu

    def sample_sigma(self):
        """ Sample self.sigma. 
        
        Note:
            Uses a randomly chosen prior :math:`(scale, shape) = (30, 14)`.
        """
        dbar = 0
        for quote in self.quotes:
            a = (quote.dV() - self.lr.predict(quote.z) - self.mu)**2
            dbar += a
        ndV = len(self.quotes)
        
        scale_prior = 20
        shape_prior = 9
        scale = scale_prior + dbar / 2
        shape = shape_prior + ndV / 2
        self.sigma = np.sqrt(
                scipy.stats.invgamma.rvs(shape, scale=scale, size=1)[0])
    
    def loglik(self):
        """ Likelihood function. """
        x = []
        for quote in self.quotes:
            bias = self.lr.predict(quote.z)
            x.append(quote.dV() - bias)
        x = np.array(x)
        ll = np.sum(- 1. / 2 * ((x - self.mu) / self.sigma)**2 - 
                                np.log(np.sqrt(2 * np.pi) * self.sigma))
        return ll


def log_prior_SEP(x):
        if x[3] > 1 and x[3] <=2:
                return 0
        else:
                return -np.inf

class AlphaSEP(BaseParams):
    
    def __init__(self, mu=0, sigma=2, beta=0, alpha=2):
        
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.lr = BayesianLinearRegression()
        self.lr.params.append(self)
        self.quotes = []
        self.history = {"mu": [self.mu], "sigma": [self.sigma], 
                                        "alpha": [self.alpha], "beta": [self.beta]}

    def __str__(self):
        return ("alphaSEP: mu = %.2f, sigma = %.2f, beta = %.2f, alpha = %.2f, "
                        "nb_quotes = %d" % (self.mu, self.sigma, self.beta, self.alpha,
                                                                len(self.quotes)))
    
    def get_params(self):
        return {"mu": self.mu, "sigma": self.sigma, "alpha": self.alpha,
                        "beta": self.beta}
    
    def set_params(self, mu, sigma, beta, alpha):
        """ Method to set a hard value for :math:`\mu` and :math:`\sigma`."""
        self.mu = mu
        self.sigma =sigma
        self.alpha = alpha
        self.beta = beta

    def update_params(self):
        """ Method to update the value ofthe parameters.
        
        The update is realised by sampling new values for the parameters based on
        the full conditionals. It currently uses a uniform prior.
        """
        if len(self.quotes) > 0:
            self.sample_params()
            self.history["mu"].append(self.mu)
            self.history["sigma"].append(self.sigma)
            self.history["alpha"].append(self.alpha)
            self.history["beta"].append(self.beta)
    
    def sample_params(self, n_iter=100):
        dW = []
        for quote in self.quotes:
            dw = quote.dW() - self.lr.predict(quote.z)
            dW.extend(dw)
        dW = np.array(dW)
        params_init = np.array([self.mu, self.sigma, self.beta, self.alpha])
        samples = MetropolisHastings(n_iter, dW, likelihoodSEP,
                                                                 params_init=params_init,
                                                                 log_prior=log_prior_SEP)
        mu, sigma, beta, alpha = samples[-1]
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
    
    def pdf(self, delta, z=None, log=False):
        bias = self.lr.predict(z)
        return dSEP(delta - bias, self.mu, self.sigma, self.beta, self.alpha, log=log)
    
    def cdf(self, delta, z=None):
        f = lambda delta: self.pdf(delta, z=z)
        return scipy.integrate.quad(f, -np.inf, delta)
    
    def rvs(self, size=1, z=None):
        """ Samples :math:`\delta` from :math:`f`. """
        bias = self.lr.predict(z)
        return rSEP(self.mu, self.sigma, self.beta, self.alpha, size=size) + bias
    
    def loglik(self):
        delta = []
        for quote in self.quotes:
            bias = self.lr.predict(quote.z)
            for dw in quote.dW():
                delta.append(dw - bias)
        delta = np.array(delta)
        return np.sum(dSEP(delta, self.mu, self.sigma, self.beta, self.alpha, log=True))


class Quote:
    """ Basic structure_cython for a quote's information. 
    
    Parameters:
        alpha (structure_cython.Alpha): Parameter class on the :math:`f`.
        beta (structure_cython.Beta): Parameter class on the :math:`g`.
        qtype (str): Either Buy or Sell.
        I (string): Status of the RFQ, either `Done`, `TradedAway` or `NotDone`.
        J (string): Detailed status.
        Y (float): BNPPAnsweredPrice.
        C (float): cover price.
        CBBT (float): RFQCompositePrice.
        bid2mid (float): Bid2Mid.
        customer (int): Integer identifying the customer (starts from 1)
        nb_dealers (int): Number of competitors who replied to the RFQ.
    
    Attributes:
        max_reject (int): maximum number of rejection in the rejection sampling.
        W (np.array): Values attibuted to the quote by competitors
        V (float): Value attributed to the quote by the customer
        z (numpy.array): Vector of co-variates.
    """

    def __init__(self, qtype, I, J, Y, C, CBBT, bid2mid, customer, nb_dealers,
                             alpha, beta, covar=None):
        
        self.logger = logging.getLogger('bnpp.Quote')
        assert(qtype in ["Buy", "Sell"])
        
        self.alpha = alpha
        self.beta = beta
        self.qtype = qtype
        self.I = I
        self.J = J
        self.Y = Y
        self.C = C
        self.CBBT = CBBT
        self.bid2mid = bid2mid
        self.customer = customer
        self.nb_dealers = nb_dealers - 1
        assert(self.nb_dealers > 0)
        self.z = covar
        self.max_reject = 10000000
        self.init_VW()
        self.logger.debug("Loaded %s" % self)

    def __str__(self):
        return ("Quote: type = %s, I = %s, J = %s, Y = %.2f, C = %.2f, V = %.2f,"
                "W = %s, Customer = %d, CBBT = %.2f, W ~ %s, V ~ %s" % 
                (self.qtype,
                 self.I,
                 self.J,
                 self.Y,
                 self.C,
                 self.V,
                 "|".join("%.2f" % w for w in self.W),
                 self.customer,
                 self.CBBT,
                 self.alpha,
                 self.beta))
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('bnpp.Quote')
    
    def rf(self, z=None):
        """ Samples from one of the dealers' distribution on latent values."""
        return self.bid2mid * self.alpha.rvs(size=1, z=z) + self.CBBT
    
    def rg(self, z=None):
        """ Samples from the customers' distribution on latent values."""
        return self.bid2mid * self.beta.rvs(size=1, z=z) + self.CBBT

    def init_VW(self):
        if self.qtype == "Sell":
            self._sellInit()
        if self.qtype == "Buy":
            self._buyInit()

    def _buyInit(self):
        """ Initialise the latent variables of the model consistently with I."""
        if self.I == "Done" and self.C == 0:
            self.V = self.Y + .5
            self.W = np.array([self.Y + .5] * self.nb_dealers)
        elif self.I == "Done" and self.C != 0:
            self.V = self.Y + .5
            self.W = np.array([self.C] + [self.C + .5] * (self.nb_dealers - 1))     
        elif self.I == "TradedAway":
            if self.J == "TiedTradedAway":
                self.C = self.Y
                self.V = self.C
                self.W = np.array([self.C] + [self.C + .5] * (self.nb_dealers - 1))
            elif self.J == "Covered":
                self.C = self.Y - .5
                self.V = self.C
                self.W = np.array([self.C] + [self.Y + .5] * (self.nb_dealers - 1))
            else:
                self.V = self.Y
                self.W = np.array([self.Y - .5] * self.nb_dealers)
        
        elif self.I == "NotTraded":
            self.V = self.Y - .5
            self.W = np.array([self.Y] * self.nb_dealers)
        # Check if initialization is consistent.
        if not self.consistent():
            raise BadInitialization("%s" % self)
    
    def _sellInit(self):
        """ Initialise the latent variables of the model consistently with I."""
        if self.I == "Done" and self.C == 0:
            self.V = self.Y - .5
            self.W = np.array([self.Y - .5] * self.nb_dealers)
        elif self.I == "Done" and self.C != 0:
            self.V = self.Y - .5
            self.W = np.array([self.C] + [self.C - .5] * (self.nb_dealers - 1))     
        elif self.I == "TradedAway":
            if self.J == "TiedTradedAway":
                self.C = self.Y
                self.V = self.C - .5
                self.W = np.array([self.C] + [self.C - .5] * (self.nb_dealers - 1))
            elif self.J == "Covered":
                self.C = self.Y + .5
                self.V = self.Y
                self.W = np.array([self.C] + [self.Y - .5] * (self.nb_dealers - 1))
            else:
                self.V = self.Y - .5
                self.W = np.array([self.Y + .5] * self.nb_dealers)
        elif self.I == "NotTraded":
            self.V = self.Y + .5
            self.W = np.array([self.Y] * self.nb_dealers)
        
        # Check if initialization is consistent.
        if not self.consistent():
            raise BadInitialization("%s" % self)
    
    def consistent(self):
        """ Calculates whether the current state of the quote is consistent. """
        if self.qtype == "Buy":
            return self._buyConsistent()
        elif self.qtype == "Sell":
            return self._sellConsistent()
        else:
            raise("Type %s not in {Buy, Sell}" % self.qtype)
        
    def _buyConsistent(self):
        if self.I == "Done":
            if self.C == 0 and (np.min(self.W) >= self.Y) and (self.Y <= self.V):
                return 1
            elif self.C != 0 and (np.min(self.W) >= self.C) and (self.Y <= min(self.C, self.V)):
                return 1
        elif self.I == "TradedAway":
            if self.J == "TiedTradedAway":
                if self.Y == self.C and np.min(self.W) == self.C and self.C <= self.V:
                    return 1
            elif self.J == "Covered":
                W_ranked = sorted(self.W)
                ll = len(W_ranked)
                if W_ranked[0] < self.Y and W_ranked[0] <= self.V:
                    if ll > 1:
                        if np.min(W_ranked[1:]) > self.Y:
                            return 1
                    else:
                        return 1
            else:
                if np.min(self.W) <= min(self.Y, self.V):
                    return 1
        elif self.I == "NotTraded" and (np.min(self.W) > self.V) and (self.Y > self.V):
            return 1
        else:
            return 0
        
    def _sellConsistent(self):
        if self.I == "Done":
            if self.C == 0 and (np.max(self.W) <= self.Y) and (self.Y >= self.V):
                return 1
            elif self.C != 0 and (np.max(self.W) <= self.C) and (self.Y >= max(self.C, self.V)):
                return 1
        elif self.I == "TradedAway":
            if self.J == "TiedTradedAway":
                if self.Y == self.C and np.max(self.W) == self.C and self.C >= self.V:
                    return 1
            elif self.J == "Covered":
                W_ranked = sorted(self.W, key=lambda x: -x)
                ll = len(W_ranked)
                if W_ranked[0] > self.Y and W_ranked[0] >= self.V:
                    if ll > 1:
                        if np.max(W_ranked[1:]) < self.Y:
                            return 1
                    else:
                        return 1
            else:
                if np.max(self.W) >= max(self.Y, self.V):
                    return 1
        elif self.I == "NotTraded" and (np.max(self.W) < self.V) and (self.Y < self.V):
            return 1
        else:
            return 0
    
    @deprecated
    def predictI(self):
        """ Probability of the RFQ status evaluated doing numerical integration.
        
        Returns:
            (float) The probability that the quote be in the "Done", "TradedAway" or 
            "NotTraded" state.
        """
        self.logger.debug("Predicting RFQ status by Integrating...")
        I = {"Done": 0, "TradedAway": 0, "NotTraded": 0}
        Y = self.Y
        CBBT = self.CBBT
        bid2mid = self.bid2mid
        deltaY = (Y - CBBT) / bid2mid
        n = self.nb_dealers
        F = self.alpha.cdf
        g = self.beta.pdf
        G = self.beta.cdf
        # Case "Done"
        I["Done"] = (1 - F(deltaY)[0])**n * (1 - G(deltaY)[0])
        # Case "TradedAway"
        int1 = lambda dv: (1 - (1 - F(min(dv, deltaY))[0])**n) * g(dv)
        I["TradedAway"] = scipy.integrate.quad(int1, -np.inf, deltaY)[0]
        # Case "NotTraded"
        int3 = lambda v: ((1 - F(v)[0])**n) * g(v)
        I["NotTraded"] = scipy.integrate.quad(int3, -np.inf, deltaY)[0]
        
        return I
    
    @deprecated
    def predictS(self, n_iter=100):
        """ Probability of the RFQ status evaluated from repeated sampling.
        
        Returns:
            (float) The probability that the quote be in the "Done", "TradedAway" or 
            "NotTraded" state.
        """
        self.logger.debug("Predicting RFQ status by sampling...")
        I = {"Done": 0, "TradedAway": 0, "NotTraded": 0}
        for _ in range(n_iter):
            dW = self.alpha.rvs(size=self.nb_dealers, z=self.z)
            dV = self.beta.rvs(z=self.z)
            W = self.bid2mid * dW + self.CBBT
            V = self.bid2mid * dV + self.CBBT
            
            if np.min(W) > self.Y and self.Y < V:
                I["Done"] += 1. / n_iter
            elif np.min(W) < min(self.Y, V):
                I["TradedAway"] += 1. / n_iter
            elif np.min(W) > V and self.Y > V:
                I["NotTraded"] += 1. / n_iter
        return I
     
    def sample_W(self):
        """ Use rejection-sampling for the W. 
        
        Returns:
            (float) The sampled :math:`W`
        
        Raises:
            CaughtInLoop: The rejection sampling procedure failed for more than 
                `max_reject` iterations.
        """
        self.logger.debug(self)
        self.logger.debug("Sampling W from %s" % self.alpha)
        for i in xrange(self.nb_dealers):
            n_reject = 0
            if i == 0 and self.I == "Done" and self.C != 0:
                self.W[0] = self.C
                continue
            if i == 0 and self.I == "TradedAway" and self.J == "TiedTradedAway":
                self.W[0] = self.Y
                continue
            if self.I == "TradedAway" and self.J == "Covered":
                pass
            for j in xrange(self.max_reject):
                n_reject += 1
                self.W[i] = self.rf(z=self.z)
                if self.consistent():
                    self.logger.debug("Sampled W_%d = %.2f in %d iterations."
                                                        % (i, self.W[i], j))
                    break
                if n_reject >= self.max_reject:
                    self.logger.critical("Sampling W from N(%.2f, %.2f) - (b2m = %s)" % 
                                        (self.bid2mid * self.alpha.mu + self.CBBT, 
                                         self.bid2mid * self.alpha.sigma,
                                         self.bid2mid))
                    self.logger.critical(self)
                    raise CaughtInLoop("W caught in loop: n_reject = %d" % n_reject)
        return self.W

    def sample_V(self):
        """ Use rejection sampling for the V. 
        
        Returns:
            (float) The sampled :math:`V`
        
        Raises:
            CaughtInLoop: The rejection sampling procedure failed for more than 
                `max_reject` iterations.
        """
        self.logger.debug("Sampling V from g_%d = %s" % (self.customer, self.beta))
        self.logger.debug(self)
        for j in xrange(self.max_reject):
            self.V = self.rg(z=self.z)
            if self.consistent():
                self.logger.debug("Sampled V = %.2f in %d iterations - %s" 
                                                    % (self.V, j, self.beta))
                return self.V
            self.logger.debug("V = %.2f was rejected" % self.V)
        raise CaughtInLoop("V caught in loop: n_reject = %d" % j)
         
    def dV(self):
        """ Returns the :math:`\delta` for the :math:`V` variable."""
        return np.array(self.V - self.CBBT) / self.bid2mid

    def dW(self):
        """ Returns the :math:`\delta` for the :math:`W_k` variables."""
        return np.array(self.W - self.CBBT) / self.bid2mid
