import pytest
import numpy
from gymnasium.spaces import Box


def test_attributes(tennis_percentage_env):
    assert tennis_percentage_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 2))
    assert tennis_percentage_env.action_space == Box(low=numpy.array([-1] * 2),
                                                     high=numpy.array([1.] * 2))
    assert numpy.array_equal(tennis_percentage_env.players, numpy.array([['Berrettini M.', 'Harris A.'],
                                                                         ['Berankis R.', 'Carballes Baena R.']]))


@pytest.mark.parametrize("action,expected_reward", [(numpy.array((0, 0)), 0),
                                                    (numpy.array((0.1, 0)), 0.11),
                                                    (numpy.array((0, 0.1)), -1),
                                                    (numpy.array((0.1, 0.1)), -0.89)])
def test_step(tennis_percentage_env, action, expected_reward):
    odds, reward, done, *_ = tennis_percentage_env.step(action)
    numpy.testing.assert_almost_equal(reward, expected_reward, 2)
    assert not done
    assert tennis_percentage_env.current_step == 1
