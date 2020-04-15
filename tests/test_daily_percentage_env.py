import pytest
from gym.spaces import Box
from pandas import DataFrame
from datetime import datetime, timedelta
import numpy
from numpy import array, zeros


def test_attributes(daily_bets_percentage_env):
    assert daily_bets_percentage_env.action_space == Box(low=zeros(shape=(2, 3)),
                                                         high=array([2 ** 2 - 0.01, 1, 1] * 2).reshape(2, 3))
    assert daily_bets_percentage_env.observation_space == Box(low=1., high=float('Inf'), shape=(2, 2))
    assert daily_bets_percentage_env.STARTING_BANK == 10
    assert daily_bets_percentage_env.balance == daily_bets_percentage_env.STARTING_BANK
    assert daily_bets_percentage_env.current_step == 0
    assert len(daily_bets_percentage_env.days) == 2
    assert daily_bets_percentage_env.days[0] == datetime.today().date() - timedelta(days=1)
    assert daily_bets_percentage_env.days[1] == datetime.today().date()


def test_get_odds(daily_bets_percentage_env):
    odds = daily_bets_percentage_env.get_odds()
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
def test_get_bet(daily_bets_percentage_env, actions, bets):
    assert numpy.array_equal(daily_bets_percentage_env.get_bet(actions), bets)


@pytest.mark.parametrize("current_step,action,expected_reward,finished",
                         [(0, array([[0, 0.1, 0.1], [0, 0.1, 0.1]]), 0, False),
                          (1, array([[0, 0.1, 0.1], [0, 0.1, 0.1]]), 0, True),
                          (0, array([[1, 0.1, 0.1], [1, 0.1, 0.1]]), 2, False),
                          (1, array([[1, 0.1, 0.1], [1, 0.1, 0.1]]), 6, True),
                          (0, array([[1, 0.1, 0.1], [2, 0.1, 0.1]]), -2, False),
                          (1, array([[1, 0.1, 0.1], [2, 0.1, 0.1]]), 2, True),
                          (0, array([[2, 0.1, 0.1], [2, 0.1, 0.1]]), 0, False),
                          (1, array([[2, 0.1, 0.1], [2, 0.1, 0.1]]), -2, True),
                          (0, array([[3, 0.1, 0.1], [3, 0.1, 0.1]]), 2, False),
                          (1, array([[3, 0.1, 0.1], [3, 0.1, 0.1]]), 4, True),
                          (0, array([[0, 0., 0.], [0, 0., 0.]]), 0, False),
                          (1, array([[0, 0., 0.], [0, 0., 0.]]), 0, True),
                          (0, array([[1, 0., 0.], [1, 0., 0.]]), 0, False),
                          (1, array([[1, 0., 0.], [1, 0., 0.]]), 0, True),
                          (0, array([[1, 0., 0.], [2, 0., 0.]]), 0, False),
                          (1, array([[1, 0., 0.], [2, 0., 0.]]), 0, True),
                          (0, array([[2, 0., 0.], [2, 0., 0.]]), 0, False),
                          (1, array([[2, 0., 0.], [2, 0., 0.]]), 0, True),
                          (0, array([[3, 0., 0.], [3, 0., 0.]]), 0, False),
                          (1, array([[3, 0., 0.], [3, 0., 0.]]), 0, True)])
def test_step(daily_bets_percentage_env, current_step, action, expected_reward, finished):
    daily_bets_percentage_env.current_step = current_step
    odds, reward, done, info = daily_bets_percentage_env.step(action)
    assert reward == expected_reward
    assert done == finished


def test_attributes_of_non_uniform(daily_bets_percentage_env_non_uniform):
    assert daily_bets_percentage_env_non_uniform.action_space == Box(low=zeros(shape=(3, 3)),
                                                                     high=array([2 ** 2 - 0.01, 1., 1.] * 3).reshape(3,
                                                                                                                     3))
    assert daily_bets_percentage_env_non_uniform.observation_space == Box(low=1., high=float('Inf'), shape=(3, 2))
    assert len(daily_bets_percentage_env_non_uniform.days) == 3
    assert daily_bets_percentage_env_non_uniform.days[0] == datetime.today().date() - timedelta(days=2)
    assert daily_bets_percentage_env_non_uniform.days[1] == datetime.today().date() - timedelta(days=1)
    assert daily_bets_percentage_env_non_uniform.days[2] == datetime.today().date()


@pytest.mark.parametrize("current_step,excpected_odds",
                         [(0, DataFrame([{'w': 1, 'l': 2}, {'w': 0, 'l': 0}, {'w': 0, 'l': 0}],
                                        dtype=numpy.float64)),
                          (1, DataFrame([{'w': 4, 'l': 3}, {'w': 0, 'l': 0}, {'w': 0, 'l': 0}],
                                        dtype=numpy.float64)),
                          (2, DataFrame([{'w': 4, 'l': 3}, {'w': 4, 'l': 3}, {'w': 5, 'l': 4}],
                                        dtype=numpy.float64))])
def test_get_odds_non_uniform(daily_bets_percentage_env_non_uniform, current_step, excpected_odds):
    daily_bets_percentage_env_non_uniform.current_step = current_step
    odds = daily_bets_percentage_env_non_uniform.get_odds()
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
def test_get_bet_non_uniform(daily_bets_percentage_env_non_uniform, actions, bets):
    assert numpy.array_equal(daily_bets_percentage_env_non_uniform.get_bet(actions), bets)


@pytest.mark.parametrize("current_step,expected_results",
                         [(0, array([[0, 1], [0, 0], [0, 0]], dtype=numpy.float64)),
                          (1, array([[1, 0], [0, 0], [0, 0]], dtype=numpy.float64)),
                          (2, array([[1, 0], [1, 0], [1, 0]], dtype=numpy.float64))])
def test_get_results_non_uniform(daily_bets_percentage_env_non_uniform, current_step, expected_results):
    daily_bets_percentage_env_non_uniform.current_step = current_step
    results = daily_bets_percentage_env_non_uniform.get_results()
    assert numpy.array_equal(results, expected_results)


@pytest.mark.parametrize("current_step,action,expected_reward,finished",
                         [(0, array([0] * 3 + [0.1] * 6).reshape(3, 3).T, 0, False),
                          (1, array([0] * 3 + [0.1] * 6).reshape(3, 3).T, 0, False),
                          (2, array([0] * 3 + [0.1] * 6).reshape(3, 3).T, 0, True),
                          (0, array([1] * 3 + [0.1] * 6).reshape(3, 3).T, -1, False),
                          (1, array([1] * 3 + [0.1] * 6).reshape(3, 3).T, 3, False),
                          (2, array([1] * 3 + [0.1] * 6).reshape(3, 3).T, 10, True),
                          (0, array([3] * 3 + [0.1] * 6).reshape(3, 3).T, 0, False),
                          (1, array([3] * 3 + [0.1] * 6).reshape(3, 3).T, 2, False),
                          (2, array([3] * 3 + [0.1] * 6).reshape(3, 3).T, 7, True),
                          (0, array([1, 1] + [0] + [0.1] * 6).reshape(3, 3).T, -1, False),
                          (1, array([1, 2] + [0] + [0.1] * 6).reshape(3, 3).T, 3, False),
                          (2, array([1, 3] + [0] + [0.1] * 6).reshape(3, 3).T, 5, True),
                          (0, array([2, 1] + [0] + [0.1] * 6).reshape(3, 3).T, 1, False),
                          (1, array([2, 2] + [0] + [0.1] * 6).reshape(3, 3).T, -1, False),
                          (2, array([2, 3] + [0] + [0.1] * 6).reshape(3, 3).T, 1, True),
                          (0, array([0] * 3 + [0] * 6).reshape(3, 3).T, 0, False),
                          (1, array([0] * 3 + [0] * 6).reshape(3, 3).T, 0, False),
                          (2, array([0] * 3 + [0] * 6).reshape(3, 3).T, 0, True),
                          (0, array([1] * 3 + [0] * 6).reshape(3, 3).T, 0, False),
                          (1, array([1] * 3 + [0] * 6).reshape(3, 3).T, 0, False),
                          (2, array([1] * 3 + [0] * 6).reshape(3, 3).T, 0, True),
                          (0, array([3] * 3 + [0] * 6).reshape(3, 3).T, 0, False),
                          (1, array([3] * 3 + [0] * 6).reshape(3, 3).T, 0, False),
                          (2, array([3] * 3 + [0] * 6).reshape(3, 3).T, 0, True),
                          (0, array([1, 1] + [0] + [0] * 6).reshape(3, 3).T, 0, False),
                          (1, array([1, 2] + [0] + [0] * 6).reshape(3, 3).T, 0, False),
                          (2, array([1, 3] + [0] + [0] * 6).reshape(3, 3).T, 0, True),
                          (0, array([2, 1] + [0] + [0] * 6).reshape(3, 3).T, 0, False),
                          (1, array([2, 2] + [0] + [0] * 6).reshape(3, 3).T, 0, False),
                          (2, array([2, 3] + [0] + [0] * 6).reshape(3, 3).T, 0, True)
                          ])
def test_step_non_uniform(daily_bets_percentage_env_non_uniform, current_step, action, expected_reward, finished):
    daily_bets_percentage_env_non_uniform.current_step = current_step
    odds, reward, done, info = daily_bets_percentage_env_non_uniform.step(action)
    assert reward == expected_reward
    assert done == finished


@pytest.mark.parametrize("current_step,action,expected_reward,finished",  # Initial bank = 10
                         # Bet size = 25% = 2.5$, Reward = -2.5
                         [(0, array([[1, 0.25, 0.0]]), -2.5, False),
                          # Bet size = [[1.5$, 0.75$]], Reward = 4*1.5 - 1.5 - 0.75 = 3.75
                          (1, array([[3, 0.15, 0.075]]), 3.75, False),
                          # Bet size = [[2.2$, 2$], [0, 3$], [2.7$, 0]]
                          # Reward = 4*2.2 + 5*2.7 - 2.2 - 2 - 3 - 2.7 = 12.4
                          (2, array([[3, 0.22, 0.2], [2, 0, 0.3], [1, 0.27, 0]]), 12.4, True),
                          # Validate not using extra odds
                          (0, array([[1, 0.25, 0.5]]), -2.5, False),
                          # Validate not using extra bets
                          (1, array([[3, 0.15, 0.075], [3, 0.2, 0.2]]), 3.75, False),
                          # Validate non legal bet
                          (2, array([[3, 0.22, 0.2], [2, 0, 0.3], [1, 0.5, 0]]), -numpy.inf, False),
                          # Validate legal bet although total percentages exceeds 100
                          (0, array([[1, 0.25, 0.8]]), -2.5, False),
                          ])
def test_step_non_uniform_non_round_percentage_with_balance(daily_bets_percentage_env_non_uniform, current_step,
                                                            action, expected_reward, finished):
    daily_bets_percentage_env_non_uniform.current_step = current_step
    odds, reward, done, info = daily_bets_percentage_env_non_uniform.step(action)
    assert reward == expected_reward
    assert done == finished
