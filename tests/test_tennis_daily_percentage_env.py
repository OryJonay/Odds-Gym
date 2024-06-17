import pytest
import numpy
from gymnasium.spaces import Box
from numpy import array


def _(action):
    return numpy.linspace(-1, 1, 2 ** 2)[action]


def test_attributes(tennis_daily_percentage_env):
    assert tennis_daily_percentage_env.action_space == Box(low=array([[-1] * 2] * 3).reshape(6),
                                                           high=array([[1] * 2] * 3).reshape(6))
    assert tennis_daily_percentage_env.observation_space == Box(low=0., high=float('Inf'), shape=(3, 2))
    assert tennis_daily_percentage_env.starting_bank == 10
    assert tennis_daily_percentage_env.balance == tennis_daily_percentage_env.starting_bank
    assert tennis_daily_percentage_env.current_step == 0
    assert numpy.array_equal(tennis_daily_percentage_env.bet_size_matrix, numpy.ones(shape=(3, 2)))
    assert numpy.array_equal(tennis_daily_percentage_env.players.values,
                             numpy.array([['Berrettini M.', 'Harris A.'],
                                          ['Berankis R.', 'Carballes Baena R.'],
                                          ['Cilic M.', 'Moutet C.'],
                                          ['Davidovich Fokina A.', 'Gombos N.'],
                                          ['Hurkacz H.', 'Novak D.'],
                                          ['Querrey S.', 'Berankis R.']]))


@pytest.mark.parametrize("action,expected_reward", [(array([0, 0] * 3).reshape(6), 0),
                                                    (array([0, 0.1] * 3).reshape(6), -2),
                                                    (array([0.1, 0] * 3).reshape(6), 0.84),
                                                    (array([0.1, 0.1] * 3).reshape(6), -1.16)])
def test_step(tennis_daily_percentage_env, action, expected_reward):
    odds, reward, done, *_ = tennis_daily_percentage_env.step(action)
    numpy.testing.assert_almost_equal(reward, expected_reward, 2)
    assert not done
    assert tennis_daily_percentage_env.current_step == 1


def test_multiple_steps(tennis_daily_percentage_env):
    odds, reward, done, truncated, info = tennis_daily_percentage_env.step(array([0, 0.1] * 3).reshape(6))
    assert reward == -2
    assert tennis_daily_percentage_env.balance == tennis_daily_percentage_env.starting_bank - 2
    assert not done
    assert tennis_daily_percentage_env.current_step == 1
    bet_size = 1 / tennis_daily_percentage_env.balance
    odds, reward, done, truncated, info = tennis_daily_percentage_env.step(array([bet_size, 0] * 3).reshape(6))
    numpy.testing.assert_almost_equal(reward, 1.64, 2)
    assert tennis_daily_percentage_env.balance == tennis_daily_percentage_env.starting_bank - 2 + 1.64
    assert not done
    assert tennis_daily_percentage_env.current_step == 2
    bet_size = 1 / tennis_daily_percentage_env.balance
    odds, reward, done, truncated, info = tennis_daily_percentage_env.step(array([0, bet_size] * 3).reshape(6))
    assert reward == -1
    assert tennis_daily_percentage_env.balance == tennis_daily_percentage_env.starting_bank - 2 + 1.64 - 1
    assert done
