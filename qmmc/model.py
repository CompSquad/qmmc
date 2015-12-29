"""Models agglomerate variables to be sampled. """

__author__ = "arnaud.rachez@gmail.com"

from .samplers import NormalMHSampler

class Model(object):

    def __init__(self, variables, samplers=None): 
        
        # Variables to sample according to their posterior.
        self.variables = variables
        
        # Assigned samplers
        self.samplers = dict()
        if samplers is not None:
            for sampler in samplers:
                name = ' & '.join(var.name for var in sampler.assigned)
                self.samplers[name] = sampler
        
        # If no samplers have been provided, automatically assign a default
        # sampler to unobserved, stochastic variables.
        assigned = set()
        for sampler in self.samplers.itervalues():
            assigned |= sampler.assigned
        
        unassigned = {var for var in variables if var not in assigned}
        for var in unassigned:
            if not var._deterministic and not var._observed:
                self.samplers[var.name] = NormalMHSampler(var)

    def logp(self):
        """ Complete log-likelihood of stochastic variables.
        """
        return sum(v.logp() for v in self.variables if not v._deterministic)
    
    def logp_custom(self):
        
        logp = 0
        for v in self.variables:
            if not v._deterministic:
                if v.name not in {'V', 'W'}:
                    logp += v.logp()

        for s in self.samplers.itervalues():
            if hasattr(s, 'W') or hasattr(s, 'V'):
                logp += s.logp()
        
        return logp

    def estimate(self, n_iter):
        
        self.logp_hist = []
        self.logp_custom_hist = []
        
        for i in xrange(n_iter):
            if i % 10 == 0:
                print "{}%".format(int(i / float(n_iter) * 100)),
            for sampler in self.samplers.values():
                sampler.sample()
            self.logp_hist.append(self.logp())
            self.logp_custom_hist.append(self.logp_custom())
        print "100%"
