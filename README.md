Quote market Monte Carlo.
=========================

This module can be used to estimate the latent pricing strategies of actors 
in a bond market. It relies on Gibbs sampling and Metropolis-Hastings 
for parameter estimation. Pricing strategies can be standard Gaussian densities 
or less common but (expectedly) more realistic Exponential Power (EP) and Skewed 
Exponential Power (SEP) distributions.

installing
----------

Before installing the package, please check that you have all the required
libraries installed:

```shell
pip install -r requirements.txt
```

If everything is OK, you can proceed with the installation.

```shell
python setup.py install
```


Documentation
-------------

The documentation is available at http://compmath.fr/doc.

Alternatively, you can build it yourself with the following commands:

```shell
cd doc/
make html
```

The documentation index will be output in `doc/build/html/index.html`.