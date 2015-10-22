import numpy as np
import scipy.stats
import scipy.integrate

f = lambda delta: scipy.stats.norm.pdf(delta, loc=0, scale=2)
F = lambda delta: scipy.stats.norm.cdf(delta, loc=0, scale=2)

g = lambda delta, mu, sigma: scipy.stats.norm.pdf(delta, loc=mu, scale=sigma)
G = lambda delta, mu, sigma: scipy.stats.norm.cdf(delta, loc=mu, scale=sigma)

def lik(quote, mu, sigma):
  I = quote.I
  Y = quote.Y
  CBBT = quote.CBBT
  bid2mid = quote.bid2mid
  deltaY = (Y - CBBT) / bid2mid
  n = quote.nb_dealers
  
  if I == "Done":
    return ((1 - F(deltaY))**n) * (1 - G(deltaY, mu, sigma))
  elif I == "TradedAway":
    int1 = lambda dv: (1 - (1 - F(min(dv, deltaY)))**n) * g(dv, mu, sigma)
    return scipy.integrate.quad(int1, -np.inf, deltaY)[0]
  elif I == "NotTraded":
    int3 = lambda v: ((1 - F(v))**n) * g(v, mu, sigma)
    return scipy.integrate.quad(int3, -np.inf, deltaY)[0]
  
def pred(quote, mu, sigma):
  pred = {"Done": 0, "TradedAway": 0, "NotTraded": 0}
  Y = quote.Y
  CBBT = quote.CBBT
  bid2mid = quote.bid2mid
  deltaY = (Y - CBBT) / bid2mid
  n = quote.nb_dealers
  pred["Done"] = (1 - F(deltaY)**n) * (1 - G(deltaY, mu, sigma))
  int1 = lambda dv: (1 - (1 - F(min(dv, deltaY)))**n) * g(dv, mu, sigma)
  pred["TradedAway"] = scipy.integrate.quad(int1, -np.inf, deltaY)[0]
  int3 = lambda v: ((1 - F(v))**n) * g(v, mu, sigma)
  pred["NotTraded"] = scipy.integrate.quad(int3, -np.inf, deltaY)[0]
  return pred
  
if __name__ == "__main__":
  from market import QuoteMarket
  from scipy.optimize import minimize
  
  market = QuoteMarket(alpha=(0, 2), beta=[(1, 2)])
  market.simulate(1000)
  
  def nll(x):
    l = np.array([lik(quote, x[0], x[1]) for quote in market.quotes])
    return -np.sum(np.log(l))
  
  print nll([1, 2])
  
  x0 = [4, 3]
  def plop(xk):
    print xk
  res = minimize(nll, x0, callback=plop)
  print "Result = %s" % str(res.x)
  print "Succes = %s" % str(res.success)
  print res.message
  
  print nll(res.x)
             
  quote = market.quotes[0]
  p = pred(quote, 0, 1)
  print p, np.sum(p.values())
  
  
  

  