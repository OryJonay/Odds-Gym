import pytest
import numpy
import io

from gymnasium.spaces import Box
from numpy import array
from unittest import mock


def _(action):
    return numpy.linspace(-1, 1, 2 ** 3)[action]


def test_attributes(three_way_daily_env):
    assert three_way_daily_env.action_space == Box(low=-1, high=1, shape=(2,))
    assert three_way_daily_env.observation_space == Box(low=0., high=float('Inf'), shape=(2, 3))
    assert three_way_daily_env.starting_bank == 10
    assert three_way_daily_env.balance == three_way_daily_env.starting_bank
    assert three_way_daily_env.current_step == 0
    assert numpy.array_equal(three_way_daily_env.bet_size_matrix, numpy.ones(shape=(2, 3)))
    assert numpy.array_equal(three_way_daily_env.teams.values, numpy.array([['FCB', 'PSG'], ['MCB', 'MTA'],
                                                                            ['PSG', 'MCB'], ['FCB', 'MTA'],
                                                                            ['PSG', 'FCB'], ['MTA', 'MCB']]))


@pytest.mark.parametrize("action,expected_reward", [((_(0), _(0)), 0),
                                                    ((_(1), _(1)), -2),
                                                    ((_(3), _(3)), -2),
                                                    ((_(2), _(2)), 0),
                                                    ((_(4), _(4)), 2),
                                                    ((_(5), _(5)), 0),
                                                    ((_(6), _(6)), 2),
                                                    ((_(7), _(7)), 0)])
def test_step(three_way_daily_env, action, expected_reward):
    odds, reward, done, *_ = three_way_daily_env.step(action)
    assert reward == expected_reward
    assert not done
    assert three_way_daily_env.current_step == 1


@pytest.mark.parametrize("current_step,action,expected_reward", [(0, array([_(0), _(0)]), 0),
                                                                 (0, array([_(5), _(6), _(7)]), 0),
                                                                 (0, array([_(2), _(7)]), 2),
                                                                 (1, array([_(4), _(7)]), -1),
                                                                 (1, array([_(4)]), -1),
                                                                 (1, array([_(1), _(3), _(7)]), 2),
                                                                 (2, array([_(2)]), 2),
                                                                 (2, array([_(2), _(2)]), 1),
                                                                 (2, array([_(2), _(4), _(1)]), 3),
                                                                 (2, array([_(3), _(0), _(7)]), 0)])
def test_step_non_uniform(three_way_daily_env_non_uniform, current_step, action, expected_reward):
    three_way_daily_env_non_uniform.current_step = current_step
    odds, reward, done, truncated, info = three_way_daily_env_non_uniform.step(action)
    assert reward == expected_reward


def test_multiple_steps(three_way_daily_env):
    odds, reward, done, truncated, info = three_way_daily_env.step((_(1), _(1)))
    assert reward == -2
    assert three_way_daily_env.balance == three_way_daily_env.starting_bank - 2
    assert not done
    assert three_way_daily_env.current_step == 1
    odds, reward, done, truncated, info = three_way_daily_env.step((_(4), _(4)))
    assert reward == -2
    assert three_way_daily_env.balance == three_way_daily_env.starting_bank - 2 - 2
    assert not done
    assert three_way_daily_env.current_step == 2
    odds, reward, done, truncated, info = three_way_daily_env.step((_(2), _(2)))
    assert reward == -2
    assert three_way_daily_env.balance == three_way_daily_env.starting_bank - 2 - 2 - 2
    assert done


def test_multiple_steps_non_uniform(three_way_daily_env_non_uniform):
    current_bank = three_way_daily_env_non_uniform.starting_bank
    odds, reward, done, truncated, info = three_way_daily_env_non_uniform.step(array([_(2), _(7)]))
    assert three_way_daily_env_non_uniform.balance == current_bank + 2
    assert not done
    assert three_way_daily_env_non_uniform.current_step == 1
    current_bank += 2
    odds, reward, done, truncated, info = three_way_daily_env_non_uniform.step(array([_(1), _(3), _(7)]))
    assert three_way_daily_env_non_uniform.balance == current_bank + 2
    assert not done
    assert three_way_daily_env_non_uniform.current_step == 2
    current_bank += 2
    odds, reward, done, truncated, info = three_way_daily_env_non_uniform.step(array([_(2), _(4), _(1)]))
    assert three_way_daily_env_non_uniform.balance == current_bank + 3
    assert done


def test_render(three_way_daily_env_non_uniform):
    with mock.patch('sys.stdout', new=io.StringIO()) as fake_stdout:
        three_way_daily_env_non_uniform.render()
    assert fake_stdout.getvalue() == 'Home Team FCB VS Away Team PSG, ' \
                                     'Home Team MCB VS Away Team MTA.\nCurrent balance at step 0: 10\n'
