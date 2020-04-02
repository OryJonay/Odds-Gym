import pytest
import numpy
from gym.spaces import Box


def test_attributes(three_way_env):
    assert three_way_env.action_space.n == 2 ** 3
    assert three_way_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 3))
    assert three_way_env.STARTING_BANK == 10
    assert three_way_env.balance == three_way_env.STARTING_BANK
    assert three_way_env.current_step == 0
    assert numpy.array_equal(three_way_env.bet_size_matrix, numpy.ones(shape=(1, 3)))
    assert numpy.array_equal(three_way_env.teams, numpy.array([['FCB', 'PSG'], ['MCB', 'MTA']]))


@pytest.mark.parametrize("action,expected_reward", [(0, 0), (1, -1), (2, 1), (3, -1), (4, 0),
                                                    (5, -2), (6, 0), (7, -1)])
def test_step(three_way_env, action, expected_reward):
    odds, reward, done, _ = three_way_env.step(action)
    assert reward == expected_reward
    assert not done
    assert three_way_env.current_step == 1


def test_multiple_steps(three_way_env):
    odds, reward, done, _ = three_way_env.step(3)
    assert reward == -1
    assert three_way_env.balance == three_way_env.STARTING_BANK - 1
    assert not done
    assert three_way_env.current_step == 1
    odds, reward, done, _ = three_way_env.step(1)
    assert reward == 3
    assert three_way_env.balance == three_way_env.STARTING_BANK - 1 + 3
    assert done
