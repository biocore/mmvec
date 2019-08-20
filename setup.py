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
from glob import glob
from setuptools import find_packages, setup

classes = """
    Development Status :: 3 - Alpha
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

description = ('Microbe-metabolite interactions through neural networks')

# version parsing from __init__ pulled from Flask's setup.py
# https://github.com/mitsuhiko/flask/blob/master/setup.py
_version_re = re.compile(r'__version__\s+=\s+(.*)')
with open('mmvec/__init__.py', 'rb') as f:
    hit = _version_re.search(f.read().decode('utf-8')).group(1)
    version = str(ast.literal_eval(hit))


setup(name='mmvec',
      version=version,
      license='BSD-3-Clause',
      description=description,
      long_description=description,
      author="gneiss development team",
      author_email="jamietmorton@gmail.com",
      maintainer="gneiss development team",
      maintainer_email="jamietmorton@gmail.com",
      packages=find_packages(),
      scripts=glob('scripts/mmvec'),
      install_requires=[
          'biom-format',
          'numpy >= 1.9.2',
          'pandas >= 0.18.0',
          'scipy >= 0.15.1',
          'nose >= 1.3.7',
          'scikit-bio >= 0.5.1',
      ],
      classifiers=classifiers,
      entry_points={
          'qiime2.plugins': ['q2-mmvec=mmvec.q2.plugin_setup:plugin']
      },
      package_data={},
      zip_safe=False)
