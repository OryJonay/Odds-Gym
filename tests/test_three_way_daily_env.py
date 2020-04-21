import pytest
import numpy
from gym.spaces import Box
from numpy import array


def test_attributes(three_way_daily_env):
    assert three_way_daily_env.action_space == Box(low=0, high=2 ** 3 - 0.01, shape=(2,))
    assert three_way_daily_env.observation_space == Box(low=0., high=float('Inf'), shape=(2, 3))
    assert three_way_daily_env.STARTING_BANK == 10
    assert three_way_daily_env.balance == three_way_daily_env.STARTING_BANK
    assert three_way_daily_env.current_step == 0
    assert numpy.array_equal(three_way_daily_env.bet_size_matrix, numpy.ones(shape=(2, 3)))
    assert numpy.array_equal(three_way_daily_env.teams.values, numpy.array([['FCB', 'PSG'], ['MCB', 'MTA'],
                                                                            ['PSG', 'MCB'], ['FCB', 'MTA'],
                                                                            ['PSG', 'FCB'], ['MTA', 'MCB']]))


@pytest.mark.parametrize("action,expected_reward", [((0, 0), 0), ((1, 1), 2), ((2, 2), 0), ((3, 3), -2), ((4, 4), 2),
                                                    ((5, 5), 0), ((6, 6), -2), ((7, 7), 0)])
def test_step(three_way_daily_env, action, expected_reward):
    odds, reward, done, _ = three_way_daily_env.step(action)
    assert reward == expected_reward
    assert not done
    assert three_way_daily_env.current_step == 1


@pytest.mark.parametrize("current_step,action,expected_reward", [(0, array([0, 0]), 0), (0, array([5, 4, 7]), 0),
                                                                 (0, array([2, 7]), 2), (1, array([1, 7]), -1),
                                                                 (1, array([1]), -1), (1, array([3, 6, 7]), 2),
                                                                 (2, array([2]), 2), (2, array([2, 2]), 1),
                                                                 (2, array([2, 1, 3]), 3), (2, array([6, 0, 7]), 0)])
def test_step_non_uniform(three_way_daily_env_non_uniform, current_step, action, expected_reward):
    three_way_daily_env_non_uniform.current_step = current_step
    odds, reward, done, _ = three_way_daily_env_non_uniform.step(action)
    assert reward == expected_reward


def test_multiple_steps(three_way_daily_env):
    odds, reward, done, _ = three_way_daily_env.step((3, 3))
    assert reward == -2
    assert three_way_daily_env.balance == three_way_daily_env.STARTING_BANK - 2
    assert not done
    assert three_way_daily_env.current_step == 1
    odds, reward, done, _ = three_way_daily_env.step((1, 1))
    assert reward == -2
    assert three_way_daily_env.balance == three_way_daily_env.STARTING_BANK - 2 - 2
    assert not done
    assert three_way_daily_env.current_step == 2
    odds, reward, done, _ = three_way_daily_env.step((2, 2))
    assert reward == -2
    assert three_way_daily_env.balance == three_way_daily_env.STARTING_BANK - 2 - 2 - 2
    assert done


def test_multiple_steps_non_uniform(three_way_daily_env_non_uniform):
    current_bank = three_way_daily_env_non_uniform.STARTING_BANK
    odds, reward, done, _ = three_way_daily_env_non_uniform.step(array([2, 7]))
    assert three_way_daily_env_non_uniform.balance == current_bank + 2
    assert not done
    assert three_way_daily_env_non_uniform.current_step == 1
    current_bank += 2
    odds, reward, done, _ = three_way_daily_env_non_uniform.step(array([3, 6, 7]))
    assert three_way_daily_env_non_uniform.balance == current_bank + 2
    assert not done
    assert three_way_daily_env_non_uniform.current_step == 2
    current_bank += 2
    odds, reward, done, _ = three_way_daily_env_non_uniform.step(array([2, 1, 3]))
    assert three_way_daily_env_non_uniform.balance == current_bank + 3
    assert done
