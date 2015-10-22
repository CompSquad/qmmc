""" Demonstrate the API on an example."""

__author__ = "arnaud.rachez@gmail.com"

import os
import logging
logging.basicConfig(level=logging.INFO)

from qmmc.sampler import BNPPGibbsSampler
from qmmc.market import GaussianQuoteMarket, SEPQuoteMarket
  
  
def main():
  
  # Create a directory to hold the results of the experiment.
  savedir = os.path.join(os.path.dirname(__file__), 'sim_market')
  if not os.path.exists(savedir):
    os.makedirs(savedir)
  
  # Initialize the market with chosen parameters
  alpha=(-1, 2, 1, 2)
  beta=[(-1, 1.5), (1, 1.5)]
  market = SEPQuoteMarket(alpha, beta, nb_covar=0)
  
  
  # Manually assign weights for regression
#   alpha_covweights = [1, 0, -1]
#   beta_covweights = [1, 0, -1]
#   market.set_covar_weights(alpha_covweights, beta_covweights)
  
  # Simulate a few RFQs and save them to a csv file
  # When the market is saved to a csv file, only observed variables and 
  # covariables are saved. The state of latent variables V and W is lost.
  market.simulate("Buy", 20000)
  market.save(savedir)
  
  # Reload the saved market
  covar = {"z0": {'type': "num"},
           "z1": {'type': "num"},
           "z2": {'type': "num"}}
  market = SEPQuoteMarket.load(savedir + "fake-market.csv", covar=None)
  
  # Sample latent variables and parameters
  n_samples = 100
  print market
  for _ in range(100):
    market.sample(n_samples, sample_covar=False)
    market.save(savedir)
#   print "Real:", alpha_covweights, beta_covweights
  
  
if __name__ == "__main__":
  main()
  
  