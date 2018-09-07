#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import re
import ast
import os
from glob import glob
from setuptools import setup


classes = """
    Development Status :: 1 - Pre-Alpha
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('Deep Multimodal Autoencoder')

with open('README.md') as f:
    long_description = f.read()
extensions = [
]
version='0.0.1'
setup(name='maestro',
      version=version,
      license='BSD',
      description=description,
      long_description=long_description,
      author="gneiss development team",
      author_email="jamietmorton@gmail.com",
      maintainer="gneiss development team",
      maintainer_email="jamietmorton@gmail.com",
      packages=['maestro'],
      scripts=glob('scripts/maestro'),
      install_requires=[
          'biom-format',
          'numpy >= 1.9.2',
          'pandas >= 0.18.0',
          'scipy >= 0.15.1',
          'nose >= 1.3.7',
          'scikit-bio >= 0.5.1',
      ],
      classifiers=classifiers,
      package_data={}
)
