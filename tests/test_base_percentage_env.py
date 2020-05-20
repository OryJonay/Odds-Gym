import pytest
import numpy
from gym.spaces import Box


def _(action):
    return numpy.linspace(-1, 1, 2 ** 2)[action]


def test_attributes(basic_percentage_env):
    assert basic_percentage_env.action_space == Box(low=numpy.array([-1] * 3),
                                                    high=numpy.array([1.] * 3))
    assert basic_percentage_env.observation_space == Box(low=1., high=float('Inf'), shape=(1, 2))
    assert basic_percentage_env.STARTING_BANK == 10
    assert basic_percentage_env.balance == basic_percentage_env.STARTING_BANK
    assert basic_percentage_env.current_step == 0
    assert basic_percentage_env.bet_size_matrix is None


@pytest.mark.parametrize("action,expected_reward", [((_(0), 0., 0.), 0),
                                                    ((_(1), 0.25, 0.), 2.5),
                                                    ((_(2), 0., 0.25), -2.5),
                                                    ((_(3), 0.25, 0.25), 0)])
def test_step(basic_percentage_env, action, expected_reward):
    odds, reward, done, _ = basic_percentage_env.step(action)
    assert reward == expected_reward
    assert done
    assert basic_percentage_env.current_step == 0


@pytest.mark.parametrize("action, expected_reward", [((_(0), 0.1, 0.0), 0),
                                                     ((_(1), 0.1, 0.1), -1),
                                                     ((_(2), 0.1, 0.1), -1),
                                                     ((_(3), 0.1, 0), -1),
                                                     ((_(3), 0.3, 0.8), -11),
                                                     ((_(2), 0, 0), 0)])
def test_legal_bet(basic_percentage_env, action, expected_reward):
    _, reward, _, _ = basic_percentage_env.step(action)
    assert expected_reward == reward
