from pytest import fixture
from numpy import array
from oddsgym.envs.base import BaseOddsEnv

@fixture()
def basic_env(request):
    return BaseOddsEnv(array([[1, 2]]), ['w', 'l'], [1, 1])
