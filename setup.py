"""This module contains the packaging routine for oddsgym"""

from setuptools import setup, find_packages

setup(packages=find_packages(),
      install_requires=['gym', 'more-itertools', 'pandas'])
