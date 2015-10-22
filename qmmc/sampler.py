""" Main script to perform Gibbs sampling on the structured model. """

__author__ = "arnaud.rachez@gmail.com"

import os
import logging

from copy import deepcopy

import numpy as np
import pandas as pd


class BNPPGibbsSampler():
  """ This class govern the Gibbs sampling procedure.
  
  Parameters:
    alpha (bnpp.Alpha): The instance for parameters on :math:`f`.
    beta (list(bnpp.Beta)): List of instances for the parameters on :math:`g`.
  
  Attributes:
    iteration (int): The current number of iterations done.
    llhist (np.array): List of sampled parameters for Alpha and Beta.
  """
  
  def __init__(self, alpha, beta):
    
    self.logger = logging.getLogger()
    self.alpha = alpha
    self.beta = beta
    self.llhist = {"alpha": [], "beta": [], "diffa": [], "diffb": []}
    self.iteration = 0
  
  def __getstate__(self):
    d = dict(self.__dict__)
    del d['logger']
    return d
  
  def __setstate__(self, d):
    self.__dict__.update(d)
    self.logger = logging.getLogger('bnpp.sampler')
  
  def get_history(self):
    params = set(self.alpha.get_params().keys() + self.beta[1].get_params().keys())
    history = deepcopy(self.alpha.history)
    history["id"] = ["alpha"] * len(history["mu"])
    for k in params:
      if not k in history.keys():
        history[k] = [0] * len(history["id"])
    for b in self.beta.itervalues():
      if len(b.quotes) != 0:
        h = b.history
        for param, hist in h.iteritems():
          try:
            history[param].extend(list(hist))
          except Exception as e:
            history[param] = list(hist)
        history["id"].extend(["beta_%d" % b.customer] * len(hist))
        for v in history.itervalues():
          if len(v) < len(history["id"]):
            v.extend([0] * (self.iteration + 1))
    history = pd.DataFrame(history)
    self.history = history
    return history
  
  def save(self, savedir='./results/', covnames=None):
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    self.history.to_csv(savedir + "history.csv")
    ll_hist = pd.DataFrame(self.llhist)
    ll_hist.to_csv(savedir + "ll_history.csv")
    
    if covnames is not None:
      np.savetxt(savedir + "alpha-covar.txt", self.alpha.lr.history, delimiter=",")
      np.savetxt(savedir + "beta-covar.txt", self.beta[1].lr.history, delimiter=",")
      covar_hist = pd.DataFrame(self.alpha.lr.history, columns=covnames)
      covar_hist.to_csv(savedir + "alpha_covar.csv")
      covar_hist = pd.DataFrame(self.beta[1].lr.history, columns=covnames)
      covar_hist.to_csv(savedir + "beta_covar.csv")
  
  def sample(self, n_iter=100, sample_covar=False):
    """ The main sampling function.
    
    Alternate between sampling the latent variables (first the :math:`V` and
    then the :math:`W_k`) and then the parameters :math:`(mu, sigma)` and 
    :math:`(\dot{\mu_m}, \dot{\sigma_m})` independently. 
    
    Arguments:
      n_iter (int): Number of samping iterations to perform.
      covar (bool): Whether to sample the covariables parameters or not.
    
    Raises:
      BadInitialization: The sampling messed up consistency of the states.
    """
    lla_prev = self.alpha.loglik()
    llb_prev = np.sum(b.loglik() for b in self.beta.itervalues())
    for _ in xrange(n_iter):
      self.logger.info("Iteration %d" % self.iteration)
      self.iteration += 1
      
      # Sample latent variables
      for _, quote in enumerate(self.alpha.quotes):
        quote.sample_V()
        quote.sample_W()
      
      # Sample parameters
      self.alpha.update_params()
      self.logger.info("new_sample_%s" % self.alpha)
      for b in self.beta.itervalues():
        b.update_params()
        self.logger.info("new_sample_%s" % b)
      
      # Keep track of the likelihood over latent variables for debugging
      lla_cur = self.alpha.loglik()
      llb_cur = np.sum(b.loglik() for b in self.beta.itervalues())
      self.llhist["alpha"].append(lla_cur)
      self.llhist["beta"].append(llb_cur)
      self.llhist["diffa"].append(lla_prev - lla_cur)
      self.llhist["diffb"].append(llb_prev - llb_cur)
      lla_prev = lla_cur
      llb_prev = llb_cur
      
      # Only sample covariable parameters when explicitly told to do so.
      if sample_covar == True:
        self.alpha.lr.update_params()
        self.beta[1].lr.update_params()
        self.logger.info(self.alpha.lr.w)
        self.logger.info(self.beta[1].lr.w)
    
    self.get_history()
                              