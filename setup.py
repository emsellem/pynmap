#!/usr/bin/env python
"""pynmap: mapping N body snapshots

    pynmap is a small module to project N body snapshots using input Velocities
    and other attached quantities (age, metallicity) to create projected maps.
"""

import sys

# Version
version = {}
with open("pynmap/version.py") as fp:
    exec(fp.read(), version)

# simple hack to allow use of "python setup.py develop".  Should not affect
# users, only developers.
if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup
else:
    from distutils.core import setup

import os

if os.path.exists('MANIFEST'): 
    os.remove('MANIFEST')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='pynmap',
      description='Python N-body mapping',
      version = version['__version__'],
      author='Eric Emsellem',
      author_email='eric.emsellem@eso.org',
      maintainer='Eric Emsellem',
      url='http://',
      packages=['pynmap', 'pynmap.utils'],
     )
