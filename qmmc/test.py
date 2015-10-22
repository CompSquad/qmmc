import numpy as np
import scipy.stats
from qmmc.distrib2 import dEP, dSEP
import matplotlib.pyplot as plt

# import plotly.plotly as py
 
# py.sign_in('arnaud.rachez', '4zvzni8k1v')

x = np.linspace(-8, 5, 50)
# mu, alpha, beta = 1.12, 2.37, 1.29
# plt.plot(x, dEP(x, mu, alpha, beta))
# 
# mu, sigma, beta, alpha = -1, 2, 1.5, 2
# plt.plot(x, dSEP(x, mu, sigma, beta, alpha), '--')

mu, sigma, beta, alpha = -1, 1.82, 1.5, 3.5
plt.plot(x, dSEP(x, mu, sigma, beta, alpha))

mu, sigma, beta, alpha = -1, 1.82, -1.5, 3.5
plt.plot(x, dSEP(x, mu, sigma, beta, alpha), '--')

# mu, sigma = 1.25, 1.82
# plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma))

# mu, sigma = .7, 2.5
# plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma), '--')
plt.show()

# fig = plt.gcf()
# plot_url = py.plot_mpl(fig)
# print plot_url