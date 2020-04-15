import pytest
import numpy
from gym.spaces import Box


def test_attributes(three_way_percentage_env):
    assert three_way_percentage_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 3))
    assert three_way_percentage_env.action_space == Box(low=numpy.array([0.] * 4),
                                                        high=numpy.array([2 ** 3 - 0.01] + [1.] * 3))
    assert numpy.array_equal(three_way_percentage_env.teams, numpy.array([['FCB', 'PSG'], ['MCB', 'MTA']]))


@pytest.mark.parametrize("action,expected_reward", [((0, 0.1, 0.1, 0.1), 0),
                                                    ((1, 0.1, 0.1, 0.1), -1),
                                                    ((2, 0.1, 0.1, 0.1), 1),
                                                    ((3, 0.1, 0.1, 0.1), -1)])
def test_step(three_way_percentage_env, action, expected_reward):
    odds, reward, done, _ = three_way_percentage_env.step(action)
    assert reward == expected_reward
    assert not done
    assert three_way_percentage_env.current_step == 1
