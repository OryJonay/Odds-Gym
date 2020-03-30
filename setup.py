"""This module contains the packaging routine for oddsgym"""
from os import path
from setuptools import setup, find_packages
from configparser import ConfigParser

def _get_version():
    with open(path.join(path.dirname(__file__), 'setup.cfg')) as setup_configuration_file:
        parser = ConfigParser()
        parser.read_string(setup_configuration_file.read())
        return parser['bumpversion']['current_version']

setup(packages=find_packages(),
      install_requires=['gym', 'more-itertools', 'pandas'],
      version=_get_version())
