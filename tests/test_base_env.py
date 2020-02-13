import pytest
from gym.spaces import Box

def test_attributes(basic_env):
    assert basic_env.action_space.n == 2 ** 2
    assert basic_env.observation_space == Box(low=1., high=float('Inf'), shape=(2,))
    assert basic_env.STARTING_BANK == 10
    assert basic_env.balance == basic_env.STARTING_BANK
    assert basic_env.current_step == 0

@pytest.mark.parametrize("action,expected_reward", [(0, 0), (1, -1), (2, 1), (3, 0)])
def test_step(basic_env, action, expected_reward):
    odds, reward, done, _ = basic_env.step(action)
    assert reward == expected_reward
    assert done
    assert basic_env.current_step == 0
