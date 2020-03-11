import pytest
from gym.spaces import Box
import numpy


def test_attributes(basic_env):
    assert basic_env.action_space.n == 2 ** 2
    assert basic_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 2))
    assert basic_env.STARTING_BANK == 10
    assert basic_env.balance == basic_env.STARTING_BANK
    assert basic_env.current_step == 0
    assert basic_env.single_bet_size == 1


@pytest.mark.parametrize("action,expected_reward", [(0, 0), (1, -1), (2, 1), (3, 0)])
def test_step(basic_env, action, expected_reward):
    odds, reward, done, _ = basic_env.step(action)
    assert reward == expected_reward
    assert not done
    assert basic_env.current_step == 1


def test_reset(basic_env):
    odds, reward, done, _ = basic_env.step(2)
    assert reward == 1
    assert basic_env.balance == basic_env.STARTING_BANK + 1
    assert not done
    assert basic_env.current_step == 1
    odds, reward, done, _ = basic_env.step(1)
    assert reward == 2
    assert done
    basic_env.reset()
    assert basic_env.balance == basic_env.STARTING_BANK


def test_render(basic_env):
    assert basic_env.render() == "Current balance at step {}: {}".format(basic_env.current_step, basic_env.balance)


@pytest.mark.parametrize("action", range(4))
def test_step_when_balance_is_0(basic_env, action):
    basic_env.balance = 0
    odds, reward, done, _ = basic_env.step(action)
    assert reward == 0
    assert done
    assert basic_env.current_step == 0


def test_step_illegal_action(basic_env):
    basic_env.balance = 1
    odds, reward, done, _ = basic_env.step(3)  # illegal - making a double when when the balance is 1
    assert reward == -float('inf')
    assert not done
    assert basic_env.current_step == 0


@pytest.mark.parametrize("current_step_value,excpected_results", [(0, numpy.array([[0, 1]], dtype=numpy.float64)),
                                                                  (1, numpy.array([[1, 0]], dtype=numpy.float64))])
def test_get_results(basic_env, current_step_value, excpected_results):
    basic_env.current_step = current_step_value
    results = basic_env.get_results()
    assert numpy.array_equal(results, excpected_results)
