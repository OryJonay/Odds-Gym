import pytest
from gym.spaces import Box
from pandas import DataFrame
from datetime import datetime, timedelta
import numpy
from numpy import array, zeros


def test_attributes(daily_bets_env):
    assert daily_bets_env.action_space == Box(low=0, high=2 ** 2 - 0.01, shape=(2,))
    assert daily_bets_env.observation_space == Box(low=1., high=float('Inf'), shape=(2, 2))
    assert daily_bets_env.STARTING_BANK == 10
    assert daily_bets_env.balance == daily_bets_env.STARTING_BANK
    assert daily_bets_env.current_step == 0
    assert daily_bets_env.single_bet_size == 1
    assert len(daily_bets_env.days) == 2
    assert daily_bets_env.days[0] == datetime.today().date() - timedelta(days=1)
    assert daily_bets_env.days[1] == datetime.today().date()


def test_get_odds(daily_bets_env):
    odds = daily_bets_env.get_odds()
    assert odds.equals(DataFrame([{'w': 1, 'l': 2}, {'w': 4, 'l': 3}], dtype=numpy.float64))


@pytest.mark.parametrize("actions,bets", [(array([0, 0]), array([[0, 0], [0, 0]])),
                                          (array([0, 1]), array([[0, 0], [1, 0]])),
                                          (array([0, 2]), array([[0, 0], [0, 1]])),
                                          (array([0, 3]), array([[0, 0], [1, 1]])),
                                          (array([1, 0]), array([[1, 0], [0, 0]])),
                                          (array([1, 1]), array([[1, 0], [1, 0]])),
                                          (array([1, 2]), array([[1, 0], [0, 1]])),
                                          (array([1, 3]), array([[1, 0], [1, 1]])),
                                          (array([2, 0]), array([[0, 1], [0, 0]])),
                                          (array([2, 1]), array([[0, 1], [1, 0]])),
                                          (array([2, 2]), array([[0, 1], [0, 1]])),
                                          (array([2, 3]), array([[0, 1], [1, 1]])),
                                          (array([3, 0]), array([[1, 1], [0, 0]])),
                                          (array([3, 1]), array([[1, 1], [1, 0]])),
                                          (array([3, 2]), array([[1, 1], [0, 1]])),
                                          (array([3, 3]), array([[1, 1], [1, 1]])),
                                          ])
def test_get_bet(daily_bets_env, actions, bets):
    assert numpy.array_equal(daily_bets_env.get_bet(actions), bets)


def test_attributes_of_non_uniform(daily_bets_env_non_uniform):
    assert daily_bets_env_non_uniform.action_space == Box(low=0, high=2 ** 2 - 0.01, shape=(3,))
    assert daily_bets_env_non_uniform.observation_space == Box(low=1., high=float('Inf'), shape=(3, 2))
    assert len(daily_bets_env_non_uniform.days) == 3
    assert daily_bets_env_non_uniform.days[0] == datetime.today().date() - timedelta(days=2)
    assert daily_bets_env_non_uniform.days[1] == datetime.today().date() - timedelta(days=1)
    assert daily_bets_env_non_uniform.days[2] == datetime.today().date()


@pytest.mark.parametrize("current_step_value,excpected_odds",
                         [(0, DataFrame([{'w': 1, 'l': 2}, {'w': 0, 'l': 0}, {'w': 0, 'l': 0}],
                                        dtype=numpy.float64)),
                          (1, DataFrame([{'w': 4, 'l': 3}, {'w': 0, 'l': 0}, {'w': 0, 'l': 0}],
                                        dtype=numpy.float64)),
                          (2, DataFrame([{'w': 4, 'l': 3}, {'w': 4, 'l': 3}, {'w': 5, 'l': 4}],
                                        dtype=numpy.float64))])
def test_get_odds_non_uniform(daily_bets_env_non_uniform, current_step_value, excpected_odds):
    daily_bets_env_non_uniform.current_step = current_step_value
    odds = daily_bets_env_non_uniform.get_odds()
    assert odds.equals(excpected_odds)


@pytest.mark.parametrize("actions,bets", [(array([0]), zeros([3, 2])),
                                          (array([0] * 2), zeros([3, 2])),
                                          (array([0] * 3), zeros([3, 2])),
                                          (array([1]), array([[1, 0], [0, 0], [0, 0]])),
                                          (array([1] * 2), array([[1, 0], [1, 0], [0, 0]])),
                                          (array([1] * 3), array([[1, 0], [1, 0], [1, 0]])),
                                          (array([2]), array([[0, 1], [0, 0], [0, 0]])),
                                          (array([2] * 2), array([[0, 1], [0, 1], [0, 0]])),
                                          (array([2] * 3), array([[0, 1], [0, 1], [0, 1]])),
                                          (array([3]), array([[1, 1], [0, 0], [0, 0]])),
                                          (array([3] * 2), array([[1, 1], [1, 1], [0, 0]])),
                                          (array([3] * 3), array([[1, 1], [1, 1], [1, 1]])),
                                          (array([0, 1]), array([[0, 0], [1, 0], [0, 0]])),
                                          (array([0, 2]), array([[0, 0], [0, 1], [0, 0]])),
                                          (array([0, 3]), array([[0, 0], [1, 1], [0, 0]])),
                                          (array([1, 0]), array([[1, 0], [0, 0], [0, 0]])),
                                          (array([1, 2]), array([[1, 0], [0, 1], [0, 0]])),
                                          (array([1, 3]), array([[1, 0], [1, 1], [0, 0]])),
                                          (array([2, 0]), array([[0, 1], [0, 0], [0, 0]])),
                                          (array([2, 1]), array([[0, 1], [1, 0], [0, 0]])),
                                          (array([2, 3]), array([[0, 1], [1, 1], [0, 0]])),
                                          (array([3, 0]), array([[1, 1], [0, 0], [0, 0]])),
                                          (array([3, 1]), array([[1, 1], [1, 0], [0, 0]])),
                                          (array([3, 2]), array([[1, 1], [0, 1], [0, 0]])),
                                          (array([0, 0, 1]), array([[0, 0], [0, 0], [1, 0]])),
                                          (array([0, 0, 2]), array([[0, 0], [0, 0], [0, 1]])),
                                          (array([0, 0, 3]), array([[0, 0], [0, 0], [1, 1]])),
                                          (array([0, 1, 0]), array([[0, 0], [1, 0], [0, 0]])),
                                          (array([0, 2, 0]), array([[0, 0], [0, 1], [0, 0]])),
                                          (array([0, 3, 0]), array([[0, 0], [1, 1], [0, 0]])),
                                          (array([1, 0, 0]), array([[1, 0], [0, 0], [0, 0]])),
                                          (array([2, 0, 0]), array([[0, 1], [0, 0], [0, 0]])),
                                          (array([3, 0, 0]), array([[1, 1], [0, 0], [0, 0]])),
                                          (array([0, 1, 1]), array([[0, 0], [1, 0], [1, 0]])),
                                          (array([0, 1, 2]), array([[0, 0], [1, 0], [0, 1]])),
                                          (array([0, 1, 3]), array([[0, 0], [1, 0], [1, 1]])),
                                          (array([0, 2, 1]), array([[0, 0], [0, 1], [1, 0]])),
                                          (array([0, 2, 2]), array([[0, 0], [0, 1], [0, 1]])),
                                          (array([0, 2, 3]), array([[0, 0], [0, 1], [1, 1]])),
                                          (array([0, 3, 1]), array([[0, 0], [1, 1], [1, 0]])),
                                          (array([0, 3, 2]), array([[0, 0], [1, 1], [0, 1]])),
                                          (array([0, 3, 3]), array([[0, 0], [1, 1], [1, 1]])),
                                          (array([1, 0, 1]), array([[1, 0], [0, 0], [1, 0]])),
                                          (array([1, 0, 2]), array([[1, 0], [0, 0], [0, 1]])),
                                          (array([1, 0, 3]), array([[1, 0], [0, 0], [1, 1]])),
                                          (array([1, 1, 0]), array([[1, 0], [1, 0], [0, 0]])),
                                          (array([1, 1, 2]), array([[1, 0], [1, 0], [0, 1]])),
                                          (array([1, 1, 3]), array([[1, 0], [1, 0], [1, 1]])),
                                          (array([1, 2, 0]), array([[1, 0], [0, 1], [0, 0]])),
                                          (array([1, 2, 1]), array([[1, 0], [0, 1], [1, 0]])),
                                          (array([1, 2, 2]), array([[1, 0], [0, 1], [0, 1]])),
                                          (array([1, 2, 3]), array([[1, 0], [0, 1], [1, 1]])),
                                          ])
def test_get_bet_non_uniform(daily_bets_env_non_uniform, actions, bets):
    assert numpy.array_equal(daily_bets_env_non_uniform.get_bet(actions), bets)


@pytest.mark.parametrize("current_step_value,excpected_results",
                         [(0, array([[0, 1], [0, 0], [0, 0]], dtype=numpy.float64)),
                          (1, array([[1, 0], [0, 0], [0, 0]], dtype=numpy.float64)),
                          (2, array([[1, 0], [1, 0], [1, 0]], dtype=numpy.float64))])
def test_get_results_non_uniform(daily_bets_env_non_uniform, current_step_value, excpected_results):
    daily_bets_env_non_uniform.current_step = current_step_value
    results = daily_bets_env_non_uniform.get_results()
    assert numpy.array_equal(results, excpected_results)


@pytest.mark.parametrize("current_step_value,action,excpected_reward,finished",
                         [(0, array([0, 0]), 0, False),
                          (1, array([0, 0]), 0, True),
                          (0, array([1, 1]), 2, False),
                          (1, array([1, 1]), 6, True),
                          (0, array([1, 2]), -2, False),
                          (1, array([1, 2]), 2, True),
                          (0, array([2, 2]), 0, False),
                          (1, array([2, 2]), -2, True),
                          (0, array([3, 3]), 2, False),
                          (1, array([3, 3]), 4, True)])
def test_step(daily_bets_env, current_step_value, action, excpected_reward, finished):
    daily_bets_env.current_step = current_step_value
    odds, reward, done, info = daily_bets_env.step(action)
    assert reward == excpected_reward
    assert done == finished


@pytest.mark.parametrize("current_step_value,action,excpected_reward,finished",
                         [(0, array([0] * 3), 0, False),
                          (1, array([0] * 3), 0, False),
                          (2, array([0] * 3), 0, True),
                          (0, array([1] * 3), -3, False),
                          (1, array([1] * 3), 1, False),
                          (2, array([1] * 3), 10, True),
                          (0, array([3] * 3), -4, False),
                          (1, array([3] * 3), -2, False),
                          (2, array([3] * 3), 7, True),
                          (0, array([1, 1]), -2, False),
                          (1, array([1, 2]), 2, False),
                          (2, array([1, 3]), 5, True),
                          (0, array([2, 1]), 0, False),
                          (1, array([2, 2]), -2, False),
                          (2, array([2, 3]), 1, True)
                          ])
def test_step_non_uniform(daily_bets_env_non_uniform, current_step_value, action, excpected_reward, finished):
    daily_bets_env_non_uniform.current_step = current_step_value
    odds, reward, done, info = daily_bets_env_non_uniform.step(action)
    assert reward == excpected_reward
    assert done == finished