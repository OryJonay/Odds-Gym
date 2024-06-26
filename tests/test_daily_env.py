from datetime import datetime, timedelta

import numpy
import pytest
from gymnasium.spaces import Box
from numpy import array, zeros
from pandas import DataFrame

from oddsgym.envs.daily_bets import DailyOddsEnv


def _(action):
    return numpy.linspace(-1, 1, 2**2)[action]


@pytest.mark.parametrize("max_number_of_games", ["auto", 1])
def test_max_number_of_games_valid(max_number_of_games):
    dataframe = DataFrame(
        [{"w": 4, "l": 3, "date": datetime.today().date(), "result": 0}]
    )
    DailyOddsEnv(
        dataframe.drop("result", axis="columns"),
        ["w", "l"],
        dataframe["result"],
        max_number_of_games,
    )


@pytest.mark.parametrize("max_number_of_games", [0, None, 3.5, "some random string"])
def test_max_number_of_games_invalid(max_number_of_games):
    dataframe = DataFrame(
        [{"w": 4, "l": 3, "date": datetime.today().date(), "result": 0}]
    )
    with pytest.raises(ValueError):
        DailyOddsEnv(
            dataframe.drop("result", axis="columns"),
            ["w", "l"],
            dataframe["result"],
            max_number_of_games,
        )


def test_attributes(daily_bets_env):
    assert daily_bets_env.action_space == Box(low=-1, high=1, shape=(2,))
    assert daily_bets_env.observation_space == Box(
        low=0.0, high=float("Inf"), shape=(2, 2)
    )
    assert daily_bets_env.starting_bank == 10
    assert daily_bets_env.balance == daily_bets_env.starting_bank
    assert daily_bets_env.current_step == 0
    assert numpy.array_equal(daily_bets_env.bet_size_matrix, numpy.ones(shape=(2, 2)))
    assert len(daily_bets_env.days) == 2
    assert daily_bets_env.days[0] == datetime.today().date() - timedelta(days=1)
    assert daily_bets_env.days[1] == datetime.today().date()


def test_get_odds(daily_bets_env):
    odds = daily_bets_env.get_odds()
    assert numpy.array_equal(
        odds,
        DataFrame([{"w": 1, "l": 2}, {"w": 4, "l": 3}], dtype=numpy.float64).values,
    )


@pytest.mark.parametrize(
    "actions,bets",
    [
        (array([0, 0]), array([[0, 0], [0, 0]])),
        (array([0, 1]), array([[0, 0], [0, 1]])),
        (array([0, 2]), array([[0, 0], [1, 0]])),
        (array([0, 3]), array([[0, 0], [1, 1]])),
        (array([1, 0]), array([[0, 1], [0, 0]])),
        (array([1, 1]), array([[0, 1], [0, 1]])),
        (array([1, 2]), array([[0, 1], [1, 0]])),
        (array([1, 3]), array([[0, 1], [1, 1]])),
        (array([2, 0]), array([[1, 0], [0, 0]])),
        (array([2, 1]), array([[1, 0], [0, 1]])),
        (array([2, 2]), array([[1, 0], [1, 0]])),
        (array([2, 3]), array([[1, 0], [1, 1]])),
        (array([3, 0]), array([[1, 1], [0, 0]])),
        (array([3, 1]), array([[1, 1], [0, 1]])),
        (array([3, 2]), array([[1, 1], [1, 0]])),
        (array([3, 3]), array([[1, 1], [1, 1]])),
    ],
)
def test_get_bet(daily_bets_env, actions, bets):
    assert numpy.array_equal(daily_bets_env.get_bet(actions), bets)


def test_attributes_of_non_uniform(daily_bets_env_non_uniform):
    assert daily_bets_env_non_uniform.action_space == Box(low=-1, high=1, shape=(3,))
    assert daily_bets_env_non_uniform.observation_space == Box(
        low=0.0, high=float("Inf"), shape=(3, 2)
    )
    assert len(daily_bets_env_non_uniform.days) == 3
    assert daily_bets_env_non_uniform.days[0] == datetime.today().date() - timedelta(
        days=2
    )
    assert daily_bets_env_non_uniform.days[1] == datetime.today().date() - timedelta(
        days=1
    )
    assert daily_bets_env_non_uniform.days[2] == datetime.today().date()


@pytest.mark.parametrize(
    "current_step,expected_odds",
    [
        (
            0,
            DataFrame(
                [{"w": 1, "l": 2}, {"w": 0, "l": 0}, {"w": 0, "l": 0}],
                dtype=numpy.float64,
            ),
        ),
        (
            1,
            DataFrame(
                [{"w": 4, "l": 3}, {"w": 0, "l": 0}, {"w": 0, "l": 0}],
                dtype=numpy.float64,
            ),
        ),
        (
            2,
            DataFrame(
                [{"w": 4, "l": 3}, {"w": 4, "l": 3}, {"w": 5, "l": 4}],
                dtype=numpy.float64,
            ),
        ),
    ],
)
def test_get_odds_non_uniform(daily_bets_env_non_uniform, current_step, expected_odds):
    daily_bets_env_non_uniform.current_step = current_step
    odds = daily_bets_env_non_uniform.get_odds()
    assert numpy.array_equal(odds, expected_odds.values)


@pytest.mark.parametrize(
    "actions,bets",
    [
        (array([0]), zeros([3, 2])),
        (array([0] * 2), zeros([3, 2])),
        (array([0] * 3), zeros([3, 2])),
        (array([1]), array([[0, 1], [0, 0], [0, 0]])),
        (array([1] * 2), array([[0, 1], [0, 1], [0, 0]])),
        (array([1] * 3), array([[0, 1], [0, 1], [0, 1]])),
        (array([2]), array([[1, 0], [0, 0], [0, 0]])),
        (array([2] * 2), array([[1, 0], [1, 0], [0, 0]])),
        (array([2] * 3), array([[1, 0], [1, 0], [1, 0]])),
        (array([3]), array([[1, 1], [0, 0], [0, 0]])),
        (array([3] * 2), array([[1, 1], [1, 1], [0, 0]])),
        (array([3] * 3), array([[1, 1], [1, 1], [1, 1]])),
        (array([0, 1]), array([[0, 0], [0, 1], [0, 0]])),
        (array([0, 2]), array([[0, 0], [1, 0], [0, 0]])),
        (array([0, 3]), array([[0, 0], [1, 1], [0, 0]])),
        (array([1, 0]), array([[0, 1], [0, 0], [0, 0]])),
        (array([1, 2]), array([[0, 1], [1, 0], [0, 0]])),
        (array([1, 3]), array([[0, 1], [1, 1], [0, 0]])),
        (array([2, 0]), array([[1, 0], [0, 0], [0, 0]])),
        (array([2, 1]), array([[1, 0], [0, 1], [0, 0]])),
        (array([2, 3]), array([[1, 0], [1, 1], [0, 0]])),
        (array([3, 0]), array([[1, 1], [0, 0], [0, 0]])),
        (array([3, 1]), array([[1, 1], [0, 1], [0, 0]])),
        (array([3, 2]), array([[1, 1], [1, 0], [0, 0]])),
        (array([0, 0, 1]), array([[0, 0], [0, 0], [0, 1]])),
        (array([0, 0, 2]), array([[0, 0], [0, 0], [1, 0]])),
        (array([0, 0, 3]), array([[0, 0], [0, 0], [1, 1]])),
        (array([0, 1, 0]), array([[0, 0], [0, 1], [0, 0]])),
        (array([0, 2, 0]), array([[0, 0], [1, 0], [0, 0]])),
        (array([0, 3, 0]), array([[0, 0], [1, 1], [0, 0]])),
        (array([1, 0, 0]), array([[0, 1], [0, 0], [0, 0]])),
        (array([2, 0, 0]), array([[1, 0], [0, 0], [0, 0]])),
        (array([3, 0, 0]), array([[1, 1], [0, 0], [0, 0]])),
        (array([0, 1, 1]), array([[0, 0], [0, 1], [0, 1]])),
        (array([0, 1, 2]), array([[0, 0], [0, 1], [1, 0]])),
        (array([0, 1, 3]), array([[0, 0], [0, 1], [1, 1]])),
        (array([0, 2, 1]), array([[0, 0], [1, 0], [0, 1]])),
        (array([0, 2, 2]), array([[0, 0], [1, 0], [1, 0]])),
        (array([0, 2, 3]), array([[0, 0], [1, 0], [1, 1]])),
        (array([0, 3, 1]), array([[0, 0], [1, 1], [0, 1]])),
        (array([0, 3, 2]), array([[0, 0], [1, 1], [1, 0]])),
        (array([0, 3, 3]), array([[0, 0], [1, 1], [1, 1]])),
        (array([1, 0, 1]), array([[0, 1], [0, 0], [0, 1]])),
        (array([1, 0, 2]), array([[0, 1], [0, 0], [1, 0]])),
        (array([1, 0, 3]), array([[0, 1], [0, 0], [1, 1]])),
        (array([1, 1, 0]), array([[0, 1], [0, 1], [0, 0]])),
        (array([1, 1, 2]), array([[0, 1], [0, 1], [1, 0]])),
        (array([1, 1, 3]), array([[0, 1], [0, 1], [1, 1]])),
        (array([1, 2, 0]), array([[0, 1], [1, 0], [0, 0]])),
        (array([1, 2, 1]), array([[0, 1], [1, 0], [0, 1]])),
        (array([1, 2, 2]), array([[0, 1], [1, 0], [1, 0]])),
        (array([1, 2, 3]), array([[0, 1], [1, 0], [1, 1]])),
    ],
)
def test_get_bet_non_uniform(daily_bets_env_non_uniform, actions, bets):
    assert numpy.array_equal(daily_bets_env_non_uniform.get_bet(actions), bets)


@pytest.mark.parametrize(
    "current_step,expected_results",
    [
        (0, array([[0, 1], [0, 0], [0, 0]], dtype=numpy.float64)),
        (1, array([[1, 0], [0, 0], [0, 0]], dtype=numpy.float64)),
        (2, array([[1, 0], [1, 0], [1, 0]], dtype=numpy.float64)),
    ],
)
def test_get_results_non_uniform(
    daily_bets_env_non_uniform, current_step, expected_results
):
    daily_bets_env_non_uniform.current_step = current_step
    results = daily_bets_env_non_uniform.get_results()
    assert numpy.array_equal(results, expected_results)


@pytest.mark.parametrize(
    "current_step,action,expected_reward,finished",
    [
        (0, array([_(0), _(0)]), 0, False),
        (1, array([_(0), _(0)]), 0, True),
        (0, array([_(2), _(2)]), 2, False),
        (1, array([_(2), _(2)]), 6, True),
        (0, array([_(2), _(1)]), -2, False),
        (1, array([_(2), _(1)]), 2, True),
        (0, array([_(1), _(1)]), 0, False),
        (1, array([_(1), _(1)]), -2, True),
        (0, array([_(3), _(3)]), 2, False),
        (1, array([_(3), _(3)]), 4, True),
    ],
)
def test_step(daily_bets_env, current_step, action, expected_reward, finished):
    daily_bets_env.current_step = current_step
    odds, reward, done, truncated, info = daily_bets_env.step(action)
    assert reward == expected_reward
    assert done == finished


@pytest.mark.parametrize(
    "current_step,action,expected_reward,finished",
    [
        (0, array([_(0)] * 3), 0, False),
        (1, array([_(0)] * 3), 0, False),
        (2, array([_(0)] * 3), 0, True),
        (0, array([_(2)] * 3), -1, False),
        (1, array([_(2)] * 3), 3, False),
        (2, array([_(2)] * 3), 10, True),
        (0, array([_(3)] * 3), 0, False),
        (1, array([_(3)] * 3), 2, False),
        (2, array([_(3)] * 3), 7, True),
        (0, array([_(2), _(2)]), -1, False),
        (1, array([_(2), _(1)]), 3, False),
        (2, array([_(2), _(3)]), 5, True),
        (0, array([_(1), _(1)]), 1, False),
        (1, array([_(1), _(1)]), -1, False),
        (2, array([_(1), _(3)]), 1, True),
    ],
)
def test_step_non_uniform(
    daily_bets_env_non_uniform, current_step, action, expected_reward, finished
):
    daily_bets_env_non_uniform.current_step = current_step
    odds, reward, done, truncated, info = daily_bets_env_non_uniform.step(action)
    assert reward == expected_reward
    assert done == finished


@pytest.mark.parametrize("current_step_value", [0, 1])
def test_get_odds_in_observation_space(daily_bets_env, current_step_value):
    daily_bets_env.current_step = current_step_value
    assert daily_bets_env.observation_space.contains(daily_bets_env.get_odds())


@pytest.mark.parametrize("current_step_value", [0, 1, 2])
def test_get_odds_in_observation_space_non_uniform(
    daily_bets_env_non_uniform, current_step_value
):
    daily_bets_env_non_uniform.current_step = current_step_value
    assert daily_bets_env_non_uniform.observation_space.contains(
        daily_bets_env_non_uniform.get_odds()
    )


def test_info(daily_bets_env):
    action = array([_(0), _(0)])
    info = daily_bets_env.create_info(action)
    assert info["current_step"] == 0
    numpy.testing.assert_array_equal(
        info["odds"], numpy.array([[1.0, 2.0], [4.0, 3.0]])
    )
    numpy.testing.assert_array_equal(
        info["verbose_action"], numpy.array([[["w", "l"]], [["w", "l"]]])
    )
    numpy.testing.assert_array_equal(info["action"], numpy.array([_(0), _(0)]))
    assert info["balance"] == 10
    assert info["reward"] == 0
    assert not info["legal_bet"]
    assert info["results"] is None
    assert not info["done"]
    # daily_bets_env.pretty_print_info(info)
