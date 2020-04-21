import pytest
import numpy
from gym.spaces import Box
from numpy import array


def test_attributes(three_way_daily_percentage_env):
    assert three_way_daily_percentage_env.action_space == Box(low=array([[0] * 4] * 2).reshape(8),
                                                              high=array([[2 ** 3 - 0.01] + [1] * 3] * 2).reshape(8))
    assert three_way_daily_percentage_env.observation_space == Box(low=1., high=float('Inf'), shape=(2, 3))
    assert three_way_daily_percentage_env.STARTING_BANK == 10
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK
    assert three_way_daily_percentage_env.current_step == 0
    assert numpy.array_equal(three_way_daily_percentage_env.bet_size_matrix, numpy.ones(shape=(2, 3)))
    assert numpy.array_equal(three_way_daily_percentage_env.teams.values, array([['FCB', 'PSG'], ['MCB', 'MTA'],
                                                                                 ['PSG', 'MCB'], ['FCB', 'MTA'],
                                                                                 ['PSG', 'FCB'], ['MTA', 'MCB']]))


@pytest.mark.parametrize("action,expected_reward", [(array([[0] * 4] * 2).reshape(8), 0),
                                                    (array([[1] + [0.1] * 3] * 2).reshape(8), 2),
                                                    (array([[2] + [0.1] * 3] * 2).reshape(8), 0),
                                                    (array([[3] + [0.1] * 3] * 2).reshape(8), -2),
                                                    (array([[4] + [0.1] * 3] * 2).reshape(8), 2),
                                                    (array([[5] + [0.1] * 3] * 2).reshape(8), 0),
                                                    (array([[6] + [0.1] * 3] * 2).reshape(8), -2),
                                                    (array([[7] + [0.1] * 3] * 2).reshape(8), 0)])
def test_step(three_way_daily_percentage_env, action, expected_reward):
    odds, reward, done, _ = three_way_daily_percentage_env.step(action)
    assert reward == expected_reward
    assert not done
    assert three_way_daily_percentage_env.current_step == 1


@pytest.mark.parametrize("current_step,action,expected_reward",
                         [(0, array([0, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1]), 0),
                          (0, array([0, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1]), 0),
                          (0, array([2, 0.0, 0.25, 0.0, 7, 0.2, 0.1, 0.1]), 6.5),
                          # Reward = (0.25*10*2)+(0.2*10*4)-0.25*10-0.2*10-0.1*10-0.1*10 = 6.5
                          (1, array([6, 0.0, 0.0, 0.0]), 0),
                          (1, array([6, 0.0, 0.4, 0.5]), 6),
                          # Reward = (0.5*10*3)-0.4*10-0.5*10 = 6
                          (2, array([2, 0.0, 0.15, 0.0, 1, 0.135, 0.135, 0.1, 7, 0.2, 0.2, 0.2]), 1),
                          # Reward = (0.15*10*3)+(0.135*10*1)+(0.2*10*2)-0.15*10-0.135*10-3*(0.2*10) = 1
                          (2, array([2, 0.0, 0.15, 0.0, 1, 0.135, 0.135, 0.1, 7, 0.4, 0.2, 0.2]), -10.85)
                          # Illegal bet
                          ])
def test_step_non_uniform(three_way_daily_percentage_env_non_uniform, current_step, action, expected_reward):
    three_way_daily_percentage_env_non_uniform.current_step = current_step
    odds, reward, done, _ = three_way_daily_percentage_env_non_uniform.step(action)
    assert reward == expected_reward


def test_multiple_steps(three_way_daily_percentage_env):
    odds, reward, done, _ = three_way_daily_percentage_env.step(array([[3] + [0.1] * 3] * 2).reshape(8))
    assert reward == -2
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK - 2
    assert not done
    assert three_way_daily_percentage_env.current_step == 1
    bet_matrix = [1 / three_way_daily_percentage_env.balance] * 3
    odds, reward, done, _ = three_way_daily_percentage_env.step(array([[1] + bet_matrix] * 2).reshape(8))
    assert reward == -2
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK - 2 - 2
    assert not done
    assert three_way_daily_percentage_env.current_step == 2
    bet_matrix = [1 / three_way_daily_percentage_env.balance] * 3
    odds, reward, done, _ = three_way_daily_percentage_env.step(array([[2] + bet_matrix] * 2).reshape(8))
    assert reward == -2
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK - 2 - 2 - 2
    assert done


def test_multiple_steps_non_uniform(three_way_daily_percentage_env_non_uniform):
    current_bank = three_way_daily_percentage_env_non_uniform.STARTING_BANK
    odds, reward, done, _ = three_way_daily_percentage_env_non_uniform.step(array([2, 0.0, 0.25, 0.0,
                                                                                   7, 0.2, 0.1, 0.1]))
    assert reward == 6.5
    assert three_way_daily_percentage_env_non_uniform.balance == current_bank + 6.5
    assert not done
    assert three_way_daily_percentage_env_non_uniform.current_step == 1
    current_bank += reward
    odds, reward, done, _ = three_way_daily_percentage_env_non_uniform.step(array([2, 0.0, 0.4, 0.0]))
    assert reward == -current_bank * 0.4
    assert three_way_daily_percentage_env_non_uniform.balance == current_bank - current_bank * 0.4
    assert not done
    assert three_way_daily_percentage_env_non_uniform.current_step == 2
    current_bank += reward
    odds, reward, done, _ = three_way_daily_percentage_env_non_uniform.step(array([2, 0.0, 0.15, 0.0,
                                                                                   1, 0.135, 0.135, 0.1,
                                                                                   7, 0.2, 0.2, 0.2]))
    assert round(reward, 2) == 0.99
    assert done
