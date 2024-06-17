import pytest
import numpy
import io

from gymnasium.spaces import Box
from unittest import mock


def _(action):
    return numpy.linspace(-1, 1, 2 ** 2)[action]


def test_attributes(tennis_daily_env):
    assert tennis_daily_env.action_space == Box(low=-1, high=1, shape=(3,))
    assert tennis_daily_env.observation_space == Box(low=0., high=float('Inf'), shape=(3, 2))
    assert tennis_daily_env.starting_bank == 10
    assert tennis_daily_env.balance == tennis_daily_env.starting_bank
    assert tennis_daily_env.current_step == 0
    assert numpy.array_equal(tennis_daily_env.bet_size_matrix, numpy.ones(shape=(3, 2)))
    assert numpy.array_equal(tennis_daily_env.players.values,
                             numpy.array([['Berrettini M.', 'Harris A.'],
                                          ['Berankis R.', 'Carballes Baena R.'],
                                          ['Cilic M.', 'Moutet C.'],
                                          ['Davidovich Fokina A.', 'Gombos N.'],
                                          ['Hurkacz H.', 'Novak D.'],
                                          ['Querrey S.', 'Berankis R.']]))


@pytest.mark.parametrize("action,expected_reward", [((_(0),), 0),
                                                    ((_(1), _(1)), -2),
                                                    ((_(3), _(3)), -1.16),
                                                    ((_(2), _(2)), 0.84)])
def test_step(tennis_daily_env, action, expected_reward):
    odds, reward, done, *_ = tennis_daily_env.step(action)
    numpy.testing.assert_almost_equal(reward, expected_reward, 2)
    assert not done
    assert tennis_daily_env.current_step == 1


def test_multiple_steps(tennis_daily_env):
    odds, reward, done, truncated, info = tennis_daily_env.step((_(1), _(1)))
    assert reward == -2
    assert tennis_daily_env.balance == tennis_daily_env.starting_bank - 2
    assert not done
    assert tennis_daily_env.current_step == 1
    odds, reward, done, truncated, info = tennis_daily_env.step((_(2), _(2)))
    assert reward == 1.25
    assert tennis_daily_env.balance == tennis_daily_env.starting_bank - 2 + 1.25
    assert not done
    assert tennis_daily_env.current_step == 2
    odds, reward, done, truncated, info = tennis_daily_env.step((_(3), _(3)))
    numpy.testing.assert_almost_equal(-0.66, reward, 2)
    numpy.testing.assert_almost_equal(tennis_daily_env.balance, tennis_daily_env.starting_bank - 2 + 1.25 - 0.66)
    assert done


def test_render(tennis_daily_env):
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        tennis_daily_env.render()
    assert fake_stdout.getvalue() == ('Player Berrettini M. VS Player Harris A., '
                                      'Player Berankis R. VS Player Carballes Baena R..'
                                      '\nCurrent balance at step 0: 10\n')
