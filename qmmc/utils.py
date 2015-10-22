""" Utils. """

def scatter_hist(x, y):
  import matplotlib.pyplot as plt
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  
  fig, axScatter = plt.subplots(figsize=(10,10))
  
  # the scatter plot:
  axScatter.scatter(x, y, alpha=.5)
  axScatter.set_aspect(1.)
  
  # create new axes on the right and on the top of the current axes
  # The first argument of the new_vertical(new_horizontal) method is
  # the height (width) of the axes to be created in inches.
  divider = make_axes_locatable(axScatter)
  axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
  axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)
  
  # make some labels invisible
  plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
           visible=False)
  axHistx.hist(x, bins=50)
  axHisty.hist(y, bins=50, orientation='horizontal')
  
  # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
  # thus there is no need to manually adjust the xlim and ylim of these
  # axis.
  
  #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
  for tl in axHistx.get_xticklabels():
      tl.set_visible(False)
  
  #axHisty.axis["left"].major_ticklabels.set_visible(False)
  for tl in axHisty.get_yticklabels():
      tl.set_visible(False)
  
  return fig

import matplotlib.pyplot as plt
import numpy as np
from pylab import *

NUM_COLORS = 5

cm = get_cmap('gist_rainbow')
color = []
for i in range(NUM_COLORS):
    color.append(cm(1.*i/NUM_COLORS))

def plot_market_hist(market, ll=False):
    rcParams['figure.figsize'] = (16.0, 16.0)
    history = market.sampler.history
    for i, col in enumerate(["mu", "sigma", "beta", "alpha"]):
	if not hasattr(market.alpha, col):
	    break
        d = history.ix[:, ["id", col]]
        legend = history["id"].unique()
        for p in legend:
            e = history.ix[d["id"] == p, col]
            plt.subplot(4, 1, i + 1)
            plt.plot(e)
            plt.title(col)
        plt.legend(legend, loc=3)
        plt.show()
    
    rcParams['figure.figsize'] = (16.0, 4.0)
    if market.covnames != []:
        chist = market.alpha.lr.history
        plt.plot(chist)
        plt.title("Alpha covariables")
        plt.legend(market.covnames)
        plt.show()

        chist = market.beta[1].lr.history
        plt.plot(chist)
        plt.title("Beta covariables")
        plt.legend(market.covnames)
        plt.show()
    
    if ll == True:
        llhist = market.sampler.llhist
        llhist = np.array(llhist['alpha']) + np.array(llhist['beta'])
        plt.plot(llhist)
        plt.title("Hidden variables likelihood")
        plt.show()


import pandas as pd

def set_mean_params(model, burn_in):
  d = pd.DataFrame(model.alpha.history)
  p = dict(np.mean(d.ix[burn_in:, :], axis=0))
  model.alpha.set_params(**p)
  for b in model.beta.itervalues():
    d = pd.DataFrame(b.history)
    p = dict(np.mean(d.ix[burn_in:, :], axis=0))
    b.set_params(**p)
  try:
    p = np.mean(model.alpha.lr.history[burn_in:, :], axis=0)
    model.alpha.lr.set_params(p)
    p = np.mean(model.beta[1].lr.history[burn_in:, :], axis=0)
    model.beta[1].lr.set_params(p)
  except:
    pass 

def gaussian_market_scatter_hist(market, param='alpha'):
    history = market.sampler.history
    mu = np.array(history.ix[history.ix[:, "id"] == param, "mu"])
    sigma = np.array(history.ix[history.ix[:, "id"] == param, "sigma"])
    fig = scatter_hist(mu, sigma)
