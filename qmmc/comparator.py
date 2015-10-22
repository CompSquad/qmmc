""" Selecting the best model """

import os
import logging
logging.basicConfig()
mlogger = logging.getLogger('bnpp')
mlogger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from market import GaussianQuoteMarket, EPQuoteMarket, SEPQuoteMarket

class MType:
  GAUSSIAN = "Gaussian"
  EP = "EP"
  SEP = "SEP"

class ModelComparator(object):
  
  def __init__(self, parameter_list, csv_file):
    
    self.logger = logging.getLogger('bnpp.ModelSelector')
    models = []
    for params in parameter_list:
      if params["model_type"] == MType.GAUSSIAN:
        model = GaussianQuoteMarket.load(csv_file, **params)
      elif params["model_type"] == MType.EP:
        model = EPQuoteMarket.load(csv_file, **params)
      elif params["model_type"] == MType.SEP:
        model = SEPQuoteMarket.load(csv_file, **params)
      else:
        raise NotImplementedError("%s is not implemented." % 
                                  params["model_type"])
      models.append((model, params))
      self.logger.info("Created %s" % model)
    self.models = models
    self.loglik = None
  
#   def set_mean_params(self, model, burn_in):
#     d = pd.DataFrame(model.alpha.history).values
#     p = np.mean(d[burn_in:, :], axis=0)
#     model.alpha.set_params(*p)
#     for b in model.beta.itervalues():
#       d = pd.DataFrame(b.history).values
#       p = np.mean(d[burn_in:, :], axis=0)
#       b.set_params(*p)
#     try:
#       p = np.mean(model.alpha.lr.history[burn_in:, :], axis=0)
#       model.alpha.lr.set_params(p)
#       p = np.mean(model.beta[1].lr.history[burn_in:, :], axis=0)
#       model.alpha.lr.set_params(p)
#     except:
#       pass  # There is no covariables in the model.
#   
#   def compare(self, n_iter, burn_in=0):
#     loglik = []
#     for model, params in self.models:
#       try:
#         covar = params["covar"]
#         use_covar = True
#       except:
#         use_covar = False
#       if params["covar"] == None:
#         use_covar = False
#       self.logger.info("Sampling %s with params %s" % (model, params))
#       model.sample(n_iter, covar=use_covar)
#       self.set_mean_params(model, burn_in=burn_in)
#       ll = model.loglik(n_iter=2)
#       loglik.append((model, ll))
#     self.loglik = loglik
#     return loglik 

def set_mean_params(model, burn_in):
  d = pd.DataFrame(model.alpha.history).values
  p = np.mean(d[burn_in:, :], axis=0)
  model.alpha.set_params(*p)
  for b in model.beta.itervalues():
    d = pd.DataFrame(b.history).values
    p = np.mean(d[burn_in:, :], axis=0)
    b.set_params(*p)
  try:
    p = np.mean(model.alpha.lr.history[burn_in:, :], axis=0)
    model.alpha.lr.set_params(p)
    p = np.mean(model.beta[1].lr.history[burn_in:, :], axis=0)
    model.alpha.lr.set_params(p)
  except:
    pass

if __name__ == "__main__":
  
  experiments = ([
    {
      "model_type": "Gaussian",
      "covar": {"NbDealers": "num"},
      "alpha_bias": (0, 3),
      "beta_bias": (30, 14)
    },
    {
      "model_type": "Gaussian",
      "covar": None,
      "alpha_bias": (0, 3),
      "beta_bias": (30, 14)
    }])
    
  datafile = "/Users/arnaud/cellule/data/bnpp/bnpp-python-onecust.csv"
  comparator = ModelComparator(experiments, datafile)
  ll = comparator.compare(n_iter=2, burn_in=0)
  print ll
