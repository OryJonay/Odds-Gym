import pytest
import numpy
import io

from gymnasium.spaces import Box
from unittest import mock


def test_attributes(tennis_env):
    assert tennis_env.action_space.n == 2 ** 2
    assert tennis_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 2))
    assert tennis_env.starting_bank == 10
    assert tennis_env.balance == tennis_env.starting_bank
    assert tennis_env.current_step == 0
    assert numpy.array_equal(tennis_env.bet_size_matrix, numpy.ones(shape=(1, 2)))
    assert numpy.array_equal(tennis_env.players, numpy.array([['Berrettini M.', 'Harris A.'],
                                                              ['Berankis R.', 'Carballes Baena R.']]))


@pytest.mark.parametrize("action,expected_reward", [(0, 0),
                                                    (1, -1),
                                                    (2, 0.11),
                                                    (3, -0.89)])
def test_step(tennis_env, action, expected_reward):
    odds, reward, done, *_ = tennis_env.step(action)
    numpy.testing.assert_almost_equal(reward, expected_reward, 2)
    assert not done
    assert tennis_env.current_step == 1


def test_multiple_steps(tennis_env):
    odds, reward, done, *_ = tennis_env.step(1)
    assert reward == -1
    assert tennis_env.balance == tennis_env.starting_bank - 1
    assert not done
    assert tennis_env.current_step == 1
    odds, reward, done, *_ = tennis_env.step(2)
    numpy.testing.assert_almost_equal(reward, 0.73, 2)
    numpy.testing.assert_almost_equal(tennis_env.balance, tennis_env.starting_bank - 1 + 0.73, 2)
    assert done


def test_render(tennis_env):
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        tennis_env.render()
    assert fake_stdout.getvalue() == 'Player Berrettini M. VS Player Harris A..\nCurrent balance at step 0: 10\n'
