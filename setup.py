import qmmc
from distutils.core import setup
import setuptools

from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("qmmc/*.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    name='qmmc',
    description='A module for estimating bond pricing strategies.',
    long_description=open('README.md').read(),
    version='0.0.1dev',
    author='Arnaud Rachez',
    author_email='arnaud.rachez@gmail.com',
    url='http://compmath.fr/doc',
    packages=['qmmc'],
    license='???'
)