""" A fake quote market to simulator to test BNPPGibbsSampler. """

__author__ = "arnaud.rachez@gmail.com"

from abc import ABCMeta, abstractmethod

import logging
import operator

import csv
import numpy as np
import pandas as pd

from sampler import BNPPGibbsSampler
from qmmc.structure import Alpha, Beta, Quote, BadInitialization
from qmmc.structure import AlphaSEP
from qmmc.structure import BayesianLinearRegression


class Error(Exception):
  """ Base class for handling Errors. """
  pass

class PType:
  GAUSSIAN = "Gaussian"
  EP = "EP"
  SEP = "SEP"

class ParamsFactory(object):
  """ A factory to create a set of parameters.
  
  The parameters objects can have a Gaussian, EP or SEP distribution.
  """

  @staticmethod
  def factory(self, alpha_type, beta_type, n):
    if alpha_type == PType.GAUSSIAN:
      alpha = Alpha()
    elif alpha_type == PType.EP:
      raise NotImplementedError("AlphaEP has been removed.")
      #alpha = AlphaEP()
    elif alpha_type == PType.SEP:
      alpha = AlphaSEP()
    else:
      raise NotImplementedError("%s is not implemented")
    
    if beta_type == PType.GAUSSIAN:
      beta = {i + 1: Beta(i + 1) for i in range(n)}
    elif beta_type == PType.EP:
      raise NotImplementedError("BetaEP has been removed.")
      #beta = {i + 1: BetaEP(i + 1) for i in range(n)}
    elif beta_type == PType.SEP:
      raise NotImplementedError("BetaSEP has been removed.")
      #beta = {i + 1: BetaSEP(i + 1) for i in range(n)} 
    else:
      raise NotImplementedError("%s is not implemented")
    
    lr = BayesianLinearRegression()
    for b in beta.itervalues():
      lr.params.append(b)
      b.lr = lr
    
    return alpha, beta
    
class QuoteMarket(object):
  """ Base class for a market of quotes.
  
  Parameters:
    alpha_type (str): Can be 'Gaussian', 'EP' or 'SEP'.
    beta_type (str): Can be 'Gaussian', 'EP' or 'SEP'.
    n (int): The number of customers in the market.
    alpha_init (tuple): Initial values for alpha.
    beta_init (list): List of initial values for beta.
  
  Attributes:
    alpha (structure_cython.BaseParam): The alpha parameters.
    beta (structure_cython.BaseParams): The beta parameters.
    nb_customers (int): The number of customer in the market.
    quotes (list): The collection of quotes.
  """
  __metaclass__ = ABCMeta
  
  def __init__(self, alpha_type, beta_type, n, alpha_init=None, beta_init=None,
               nb_covar=0, alpha_prior=None, beta_prior=None, **kwargs):
    
    self.logger = logging.getLogger('bnpp.QuoteMarket')
    
    alpha, beta = ParamsFactory.factory(self, alpha_type, beta_type, n)
    if alpha_init != None:
      alpha.set_params(*alpha_init)
    if beta_init != None:
      for b, params in zip(beta.itervalues(), beta_init):
        b.set_params(*params)
    self.alpha = alpha
    self.beta = beta
    self.nb_customers = len(beta)
    self.dW = []
    self.dV = []
    self.quotes = []
    self.nb_covar = nb_covar
    if nb_covar != 0:
      self.covnames = ["z%d" % k for k in range(nb_covar)]
    else:
      self.covnames = None
    self.sampler = None

  def __getstate__(self):
    d = dict(self.__dict__)
    del d['logger']
    return d

  def __setstate__(self, d):
    self.__dict__.update(d)
    self.logger = logging.getLogger('bnpp.QuoteMarket')
  
  def nb_dealers(self):
    """ Helper function to return a random number of traders. """
    return np.random.randint(1, 5)

  def sample_rfq(self, qtype):
    """ Generates fake RFQs from the given parameters. """
    
    # Sample all the variables
    # We chose to add a bias of 1. to the randomly generated covariables for 
    # absolutely no good reason, it could have been any other number.
    nb_dealers = self.nb_dealers()
    customer = np.random.randint(self.nb_customers) + 1
    if self.nb_covar > 0:
      covar = 1 + np.random.randn(self.nb_covar)
    else:
      covar = None
    dY = self.alpha.rvs(size=1, z=covar)
    dW = self.alpha.rvs(size=nb_dealers, z=covar)
    dV = self.beta[customer].rvs(size=1, z=covar)
    
    # Save latent variables
    self.dV.append(dV)
    self.dW.append(dW)
    
    # Compute fake observed variables and latent prices
    cbbt = 110
    bid2mid = .5
    Y = bid2mid * dY + cbbt
    W = bid2mid * dW + cbbt
    V = bid2mid * dV + cbbt
    I = self.decide_quote(Y, V, W, qtype)
    J = "None"
    C = 0
    assert(qtype in ["Sell", "Buy"])
    
    # Add quote to the market
    quote = Quote(qtype, I, J, Y, C, cbbt, bid2mid, customer, nb_dealers + 1, 
                  self.alpha, self.beta[customer], covar=covar)
    quote.V = V
    quote.W = W if nb_dealers > 1 else np.array([W])
    self.alpha.quotes.append(quote)
    self.beta[customer].quotes.append(quote)
    self.quotes.append(quote)
    return quote
  
  def predict(self, test_quotes, n_iter=1000, type='probs'):
    """ Compute the probability of the RFQ status for a list of quotes.
    
    Arguments:
      test_quotes (list of structure_cython.Quote): A list of quotes.
      n_iter (int): The number of samples to use to evaluate the probability.
      type (str): Either `probs` to get the probability of each state, or 
        `label` to get the most probable label.
      
    Returns:
      (list of dict) The probability of each quote to be in the "Done",
      "TradedAway" or "NotTraded" state.
    """
    self.logger.info("Predicting RFQ status by sampling...")
    pred = []
    for quote in test_quotes:
      I = quote.predictS(n_iter=n_iter)
      pred.append(I)
    if type != 'probs':
      return [max(p.iteritems(), key=operator.itemgetter(1))[0] for p in pred]
    return pred

  def sample(self, n_iter=100, sample_covar=True):
    """ Create a Gibbs samppler and sample latent variables and parameters. 
    
    Arguments:
      n_iter (int): Number of iterations to run the sampler.
      covar (bool): Whether to fit a bayesian linear regression on the
        covariables or not.
      savedir (str): A path to the folder where the samples and metrics will be 
        saved.
    """
    if self.sampler is None:
      sampler = BNPPGibbsSampler(self.alpha, self.beta)
      self.sampler = sampler
    else:
      sampler = self.sampler
    sampler.sample(n_iter, sample_covar=sample_covar)
   
  def loglik(self, n_iter=1000):
    """ Returns the log-likelihood of the market. """
    self.logger.info("Computing liklihood of the market...")
    status = [quote.I for quote in self.quotes]
    pred = self.predict(self.quotes, n_iter=n_iter)
    ps = zip(pred, status)
    lik = [p[s] for p, s in ps]
    return np.sum([np.log(l + 1e-16) for l in lik])     
    
    
  def decide_quote(self, Y, V, W, qtype="Buy"):
    """ Computes the state of the quote: `Done`, `TradedAway` or `NotTraded`.
    
    Arguments:
      Y (float): BNPPAnsweredQuote
      V (float): Hidden price attributed to the quote by the customer.
      W (np.array): Hidden prices of the dealers.
    
    Returns:
      (str) The status of the trade.
    
    Raises:
      Error: Something was not possible.
    """
    assert(qtype in ["Sell", "Buy"])
    if qtype == "Buy":
      if (np.min(W) >= Y) and (Y <= V):
        return "Done"
      elif np.min(W) <= min(Y, V):
        return "TradedAway"
      elif (np.min(W) > V) and (Y > V):
        return "NotTraded"
      else:
        return Error("Bug!")
    else:
      if (np.max(W) <= Y) and (Y >= V):
        return "Done"
      elif np.max(W) >= max(Y, V):
        return "TradedAway"
      elif (np.max(W) < V) and (Y < V):
        return "NotTraded"
      else:
        return Error("Bug!")
  
  def set_covar_weights(self, alpha_covweights, beta_covweights):
    """ Manually affects values to the linear regression coefficients.
    
    Arguments:
      alpha_covar (list): A list of weights to set the regression on W
      beta_covar (list): A list of weights for the regression on V
    """
    self.alpha.lr.set_params(alpha_covweights)
    self.beta[1].lr.set_params(beta_covweights)
    assert(len(alpha_covweights) == len(beta_covweights))
    self.nb_covar = len(alpha_covweights)
  
  def simulate(self, qtype="Buy", nb_trade=2000):
    """ Fake a market.
    
    Arguments:
      nb_trade (int): The number of RFQs done one the market.
    """
    for _ in xrange(nb_trade):
      self.sample_rfq(qtype)
  
  def __str__(self):
    return ("Y ~ %s,\n"
            "W ~ %s, \n"
            "V ~ %s" 
            % (self.alpha, self.alpha,
               "\n    ".join("%s" % b for b in self.beta.itervalues())))
    
  def show(self):
    print self
  
  @staticmethod 
  def load(dforpath, alpha_type, beta_type, covar=None, **kwargs):
    """ Loads observed variables from a dataframe.
    
    Arguments:
      dforpath (pandas.dataframe): A dataframe with observed variables. A string
        can also be provided corresponding to the location of the csv file to 
        load in df.
      covar (dict): A dictionary of variable names and their format. The format 
        can be `num` (numerical) or `cat` (categorical).
      alpha_type (str): Can be 'normal', 'EP' or 'SEP'.
      beta_type (str): Can be 'normal', 'EP' or 'SEP'.
      
    Returns:
      (market.QuoteMarket) An instance of the loaded market.
    
    Raises:
      Error: If the quote is inconsistent or the type of the covariable is not 
      known. (TODO: create separate error classes)
    """
    
    if isinstance(dforpath, str):
      df = pd.read_csv(dforpath)
    else:
      df = dforpath
    nb_customers = df["Customer"].max()
    market = QuoteMarket(alpha_type, beta_type, nb_customers, **kwargs)
    market.nb_covar = len(covar) if covar is not None else 0
    market.covnames = []
    alpha, beta = market.get_params()
    
    # Covariables can be specified and need to be preprocessed a bit if they are
    # numerical or categorical. If no covariable are specified then covar should
    # be `None`.
    if covar is not None:
      for var, attrib in covar.iteritems():
        t = attrib['type']
        if t not in ["num", "cat"]:
          raise Error("Wrong type %s provided for covariable" % type)
        elif t == "cat":
          cvar = pd.get_dummies(df[var])
          if var == "RiskCaptain":
            cvar = cvar / np.tile(df["Bid2Mid"].values[:, None], cvar.shape[1])
          r = attrib.get('ref')
          if r not in cvar.columns:
            raise Error("Reference level %s does not exist" % r)
          cvar.drop(attrib.get('ref'), axis=1, inplace=True)
          market.covnames.extend(cvar.columns)
          cvar = cvar.values
        elif t == "num":
          cvar = np.array([df[var].values], dtype=float).T
          market.covnames.append(var)
        try:
          covar = np.hstack((covar, cvar))
        except:
          covar = cvar
    
    # Iteratively create quote instances and throw an error with those who are
    # inconsistent in the database.
    for i, rfq in df.iterrows():
      try:
        covari = covar[i, :] if covar is not None else None
        quote = Quote(rfq["BuySell"],
                      rfq["TradeStatus"],
                      rfq["StatusDetails"],
                      rfq["BNPPAnsweredQuote"],
                      rfq["CoverPrice"],
                      rfq["RFQCompositePrice"],
                      rfq["Bid2Mid"],
                      rfq["Customer"],
                      rfq["NbDealers"],
                      alpha,
                      beta[rfq["Customer"]],
                      covar=covari)
        alpha.quotes.append(quote)
        beta[rfq["Customer"]].quotes.append(quote)
        market.quotes.append(quote)
      except BadInitialization as e:
        market.logger.info("Skipped %s" % e)
    return market
  
  def get_params(self):
    """ Returns the alpha and beta objects. """
    return self.alpha, self.beta
  
  def set_params(self, alpha_params, beta_params):
    """ Initialise alpha and beta with given values. """
    self.alpha.set_params(*alpha_params)
    for beta, params in zip(self.beta.itervalues(), beta_params):
      beta.set_params(*params)
  
  def save(self, path=''):
    """ Saves the fake market to a text file.
    
    Arguments:
      path (str): The path to the folder where the files will be output.
    """
    if self.covnames is None and self.nb_covar > 0:
      self.covnames = ["z%d" % k for k in range(self.nb_covar)]
    
    filepath = path +"fake-market.csv"
    with open(filepath, 'w') as csvfile:
      fieldnames = ["TradeStatus", "StatusDetails", "BNPPAnsweredQuote",
                    "CoverPrice", "RFQCompositePrice", "Bid2Mid", "Customer",
                    "NbDealers", "delta", "BuySell"]
      if self.covnames is not None:
        fieldnames += self.covnames
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      
      for quote in self.quotes:
        row = {"BuySell": quote.qtype,
               "BNPPAnsweredQuote": quote.Y,
               "TradeStatus": quote.I,
               "StatusDetails": quote.J,
               "CoverPrice": quote.C,
               "RFQCompositePrice": quote.CBBT,
               "Bid2Mid": quote.bid2mid,
               "Customer": quote.customer,
               "NbDealers": quote.nb_dealers + 1,
               "delta": (quote.Y - quote.CBBT) / quote.bid2mid}
        if self.covnames is not None and quote.z is not None:
          for k, z in zip(self.covnames, quote.z):
            row[k] = z
        writer.writerow(row)
    
    filepath = path + "fake-market-params.txt"
    with open(filepath, 'w') as f:
      f.write("%s" % self)
    try:
      self.sampler.save(savedir=path, covnames=self.covnames)
    except:
      pass
  
class GaussianQuoteMarket(QuoteMarket):
  """ A class to generate RFQs from a set of known parameters.
  
  Parameters:
    alpha (tuple): :math:`(\mu, \sigma)` for :class:`structure_cython.Alpha`
      initialization.
    beta (list of tuple): :math:`(\mu_m, \sigma_m)_{m=1..M}` for
      :class:`structure_cython.Beta` initialization.
    
  Attributes:
    quotes (list): A list of dict containing the quote's state.
  """
  
  def __init__(self, alpha=(0, 2, 0), beta=[(-1, 1, 0), (0, 1, 0), (1, 1, 0)],
               **kwargs):
    super(GaussianQuoteMarket, self).__init__(
        PType.GAUSSIAN, PType.GAUSSIAN, len(beta), alpha, beta, **kwargs)
    
  @staticmethod
  def load(df, covar=None, **kwargs):
    """ Statically loads a market from a dataframe.
    
    Arguments:
      df (pandas.dataframe): A data frame.
    """
    return QuoteMarket.load(df, PType.GAUSSIAN, PType.GAUSSIAN, covar=covar,
                            **kwargs)


class SEPQuoteMarket(QuoteMarket):
  """ QuoteMarket with SEP traders and Gaussian clients."""
  
  def __init__(self, alpha=(0, 2, 0, 2), 
               beta=[(-1, 1, 0), (0, 1, 0), (1, 1, 0)], **kwargs):
    super(SEPQuoteMarket, self).__init__(
        PType.SEP, PType.GAUSSIAN, len(beta), alpha, beta, **kwargs)
  
  @staticmethod
  def load(df, covar=None, **kwargs):
    """ Statically loads a market from a dataframe.
    
    Arguments:
      df (pandas.dataframe): A data frame.
    """
    return QuoteMarket.load(df, PType.SEP, PType.GAUSSIAN, covar=covar,
                            **kwargs)
  