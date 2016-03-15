""" A collections of structural models."""

__author__ = "arnaud.rachez@gmail.com"

import numpy as np

from minipgm.variables import Value, Function
from minipgm.variables import Binomial, Normal, InvGamma, Uniform, Beta
from minipgm.models import Model

from .variables import BernoulliSEP, BernoulliFlip
from .samplers.sampler_default import KVWSampler, VSampler


class ExtendedModel(object):
    
    def __init__(self, k_real, n_real, Y_real, C_real, I_real):
        """ Build the model graph.
        
        Parameters
        ----------
        k_real: int
            Real number of clients that replied
        n_real: int
            Raximum number of clients that could have replied
        Y_real: float
            Quote
        C_real: ndarray
            Quote of the other dealers
        I_real: int [0, 1, 2]
            Status of the RFQ
        """
        
        # Map status to number
        def IMap(v):
            if v == 'Done': return 2
            elif v == 'TradedAway': return 1
            elif v == 'NotTraded': return 0
            else:
                print v
                raise ValueError()
        I_real = np.array([IMap(v) for v in I_real])
        print I_real == 2
        
        # Priors
        mu_0, sigma_0 = Value(0), Value(10)
        scale_0, shape_0 = Value(3), Value(3)
        a_0, b_0 = Value(1), Value(1)
        lower_0, upper_0 = Value(1), Value(3)
        
        # Parameters
        mu_V = Normal(mu_0, sigma_0, value=0, name='mu_V')
        sigma_V = InvGamma(scale_0, shape_0, value=0.5, name='sigma_V')
        
        mu_W = Normal(mu_0, sigma_0, value=0, name='mu_W')
        sigma_W = InvGamma(scale_0, shape_0, value=3, name='sigma_W')
        nu_W = Normal(mu_0, sigma_0, value=1, name='nu_W')
        tau_W = Uniform(lower_0, upper_0, value=1.5, name='tau_W')
        
        p = Beta(a_0, b_0, value=.5, name='p')
        
        # Variables
        m = len(Y_real)
        k = np.empty(m, dtype=object)
        V = np.empty(m, dtype=object)
        W = np.empty(m, dtype=object)
        Y = np.empty(m, dtype=object)
        I = np.empty(m, dtype=object)
        S = np.empty(m, dtype=object)
        for i in xrange(m):
            V[i] = Normal(mu_V, sigma_V, name='V_%d' % i)
            if I_real[i] == 2:
                k[i] = Binomial(p, Value(n_real[i]), value=k_real[i],
                                observed=True, name='k_%d' % i)
                W[i] = BernoulliSEP(mu_W, sigma_W, nu_W, tau_W, k[i],
                                    value=C_real[i, :k_real[i]],
                                    observed=True, name='W_%d' % i)
            else:
                k[i] = Binomial(p, Value(n_real[i]), name='k_%d' % i)
                W[i] = BernoulliSEP(
                        mu_W, sigma_W, nu_W, tau_W, k[i], name='W_%d' % i)
                
            Y[i] = Value(Y_real[i])
            
            @Function
            def S_i(Y=Y[i], V=V[i], W=W[i]):
                
                if len(W) > 0:
                    C = np.min(W)
                else:
                    C = np.inf
                
                if Y <= np.minimum(C, V): return 2
                if C <= np.minimum(Y, V): return 1
                if V < np.minimum(C, Y): return 0
            
            S[i] = S_i
            I[i] = BernoulliFlip(Value(0.0), S[i], k=3, value=I_real[i],
                                 observed=True, name='I_%d' % i)
        
            
        variables = [mu_V, sigma_V, mu_W, sigma_W, nu_W, tau_W, p]
        variables.extend(V)
        variables.extend(W)
        
        v_samplers = [VSampler(V[i]) for i in xrange(m) if W[i]._observed]
        kvw_samplers = [KVWSampler(k[i], V[i], W[i], Y[i], I[i]) for i in range(m) if not W[i]._observed]
        
        samplers = v_samplers + kvw_samplers
        
        self.mu_V = mu_V
        self.sigma_V = sigma_V
        self.mu_W = mu_W
        self.sigma_W = sigma_W
        self.nu_W = nu_W
        self.tau_W = tau_W
        self.p = p
        self.V = V
        self.W = W
        
        self.model = Model(variables=variables, samplers=samplers)
        self.samplers = self.model.samplers
        self.variables = self.model.variables


            

