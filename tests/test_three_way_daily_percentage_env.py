import pytest
import numpy
from gym.spaces import Box


def test_attributes(three_way_daily_percentage_env):
    assert three_way_daily_percentage_env.action_space == Box(low=numpy.array([[0] * 4] * 2),
                                                              high=numpy.array([[2 ** 3 - 0.01] + [1] * 3] * 2))
    assert three_way_daily_percentage_env.observation_space == Box(low=1., high=float('Inf'), shape=(2, 3))
    assert three_way_daily_percentage_env.STARTING_BANK == 10
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK
    assert three_way_daily_percentage_env.current_step == 0
    assert numpy.array_equal(three_way_daily_percentage_env.bet_size_matrix, numpy.ones(shape=(2, 3)))
    assert numpy.array_equal(three_way_daily_percentage_env.teams.values, numpy.array([['FCB', 'PSG'], ['MCB', 'MTA'],
                                                                                       ['PSG', 'MCB'], ['FCB', 'MTA'],
                                                                                       ['PSG', 'FCB'], ['MTA', 'MCB']]))


@pytest.mark.parametrize("action,expected_reward", [(numpy.array([[0] * 4] * 2), 0),
                                                    (numpy.array([[1] + [0.1] * 3] * 2), 2),
                                                    (numpy.array([[2] + [0.1] * 3] * 2), 0),
                                                    (numpy.array([[3] + [0.1] * 3] * 2), -2),
                                                    (numpy.array([[4] + [0.1] * 3] * 2), 2),
                                                    (numpy.array([[5] + [0.1] * 3] * 2), 0),
                                                    (numpy.array([[6] + [0.1] * 3] * 2), -2),
                                                    (numpy.array([[7] + [0.1] * 3] * 2), 0)])
def test_step(three_way_daily_percentage_env, action, expected_reward):
    odds, reward, done, _ = three_way_daily_percentage_env.step(action)
    assert reward == expected_reward
    assert not done
    assert three_way_daily_percentage_env.current_step == 1


def test_multiple_steps(three_way_daily_percentage_env):
    odds, reward, done, _ = three_way_daily_percentage_env.step(numpy.array([[3] + [0.1] * 3] * 2))
    assert reward == -2
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK - 2
    assert not done
    assert three_way_daily_percentage_env.current_step == 1
    bet_matrix = [1 / three_way_daily_percentage_env.balance] * 3
    odds, reward, done, _ = three_way_daily_percentage_env.step(numpy.array([[1] + bet_matrix] * 2))
    assert reward == -2
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK - 2 - 2
    assert not done
    assert three_way_daily_percentage_env.current_step == 2
    bet_matrix = [1 / three_way_daily_percentage_env.balance] * 3
    odds, reward, done, _ = three_way_daily_percentage_env.step(numpy.array([[2] + bet_matrix] * 2))
    assert reward == -2
    assert three_way_daily_percentage_env.balance == three_way_daily_percentage_env.STARTING_BANK - 2 - 2 - 2
    assert done
