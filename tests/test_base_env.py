import io
import pytest
import numpy

from unittest import mock
from gymnasium.spaces import Box


def test_attributes(basic_env):
    assert basic_env.action_space.n == 2 ** 2
    assert basic_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 2))
    assert basic_env.starting_bank == 10
    assert basic_env.balance == basic_env.starting_bank
    assert basic_env.current_step == 0
    assert numpy.array_equal(basic_env.bet_size_matrix, numpy.ones(shape=(1, 2)))


@pytest.mark.parametrize("action,expected_reward", [numpy.array((0, 0)),
                                                    numpy.array((1, 1)),
                                                    numpy.array((2, -1)),
                                                    numpy.array((3, 0))])
def test_step(basic_env, action, expected_reward):
    odds, reward, done, *_ = basic_env.step(action)
    assert reward == expected_reward
    assert not done
    assert basic_env.current_step == 1


def test_reset(basic_env):
    odds, reward, done, truncated, info = basic_env.step(1)
    assert reward == 1
    assert basic_env.balance == basic_env.starting_bank + 1
    assert not done
    assert basic_env.current_step == 1
    assert info['legal_bet']
    assert info['results'] == 1
    assert info['reward'] == 1
    assert not info['done']
    odds, reward, done, *_ = basic_env.step(2)
    assert reward == 2
    assert done
    basic_env.reset()
    assert basic_env.balance == basic_env.starting_bank


def test_info(basic_env):
    info = basic_env.create_info(1)
    assert info['current_step'] == 0
    numpy.testing.assert_array_equal(info['odds'], numpy.array([[1, 2]]))
    assert info['verbose_action'] == [['l']]
    assert info['action'] == 1
    assert info['balance'] == 10
    assert info['reward'] == 0
    assert not info['legal_bet']
    assert info['results'] is None
    assert not info['done']
    basic_env.pretty_print_info(info)


def test_render(basic_env):
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        basic_env.render()
    assert fake_stdout.getvalue() == 'Current balance at step 0: 10\n'


@pytest.mark.parametrize("action", range(4))
def test_step_when_balance_is_0(basic_env, action):
    basic_env.balance = 0
    odds, reward, done, *_ = basic_env.step(action)
    assert reward == 0
    assert done
    assert basic_env.current_step == 0


def test_step_illegal_action(basic_env):
    basic_env.balance = 1
    odds, reward, done, *_ = basic_env.step(3)  # illegal - making a double when when the balance is 1
    assert reward == -2
    assert not done
    assert basic_env.current_step == 1


@pytest.mark.parametrize("current_step,expected_results", [(0, numpy.array([[0, 1]], dtype=numpy.float64)),
                                                           (1, numpy.array([[1, 0]], dtype=numpy.float64))])
def test_get_results(basic_env, current_step, expected_results):
    basic_env.current_step = current_step
    results = basic_env.get_results()
    assert numpy.array_equal(results, expected_results)
