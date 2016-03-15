[![Build Status](https://travis-ci.org/CompSquad/qmmc.svg?branch=master)](https://travis-ci.org/CompSquad/qmmc)

Quote market Monte Carlo.
=========================

This module can be used to estimate the latent pricing strategies of actors 
in a bond market. It relies on Gibbs sampling and Metropolis-Hastings 
for parameter estimation.

Installing
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

Testing
-------

```shell
python -m unittest discover -v
```

Examples
--------

Please see folder `/examples`.

