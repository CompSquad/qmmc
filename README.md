Quote market Monte Carlo.
=========================

This module can be used to estimate the latent pricing strategies of actors 
in a bond market. It relies on Gibbs sampling and Metropolis-Hastings 
for parameter estimation.

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

Cython
------

Cython support has been removed by default. Soon to be re-enabled.

Testing
-------

```shell
python -m unittest discover -v
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
