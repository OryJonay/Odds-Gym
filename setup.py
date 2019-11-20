from setuptools import setup, find_packages

setup(name='oddsgym',
      version='0.0.1',
      description="A sports betting environment for OpenAI Gym.",
      author="Ory Jonay",
      author_email='kriegsmeister@gmail.com',
      url='https://github.com/OryJonay/Odds-Gym',
      packages=find_packages(),
      install_requires=['gym', 'more_itertools'])
