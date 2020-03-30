"""This module contains the packaging routine for oddsgym"""

from setuptools import setup, find_packages
from bumpversion.cli import _load_configuration

setup(packages=find_packages(),
      install_requires=['gym', 'more-itertools', 'pandas'],
      version=_load_configuration('setup.cfg', None, {})[0]['bumpversion']['current_version'])
