import shutil
import pytest
import numpy
from gymnasium import envs
from oddsgym.envs.footballdata import FootballDataDailyPercentageEnv
from oddsgym.utils.constants import SupportedSport
from oddsgym.utils.constants.football import CSV_CACHE_PATH
from oddsgym.utils.csv_downloader import create_csv_cache


@pytest.mark.parametrize('kwargs', [{'country': 'England', 'league': 'Premier League', 'start': 1999, 'end': 2011},
                                    {'country': 'France', 'league': 'Premier League', 'start': 2010, 'end': 2011},
                                    {'country': 'England', 'league': 'Premier League', 'start': 2010, 'end': 3000},
                                    {'country': 'England', 'league': 'Premier League', 'start': 2010, 'end': 2009},
                                    {'country': 'Brazil', 'league': 'Premier League', 'start': 2010, 'end': 2011}])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        FootballDataDailyPercentageEnv(config=kwargs)


@pytest.mark.parametrize('optimizer,reward1,reward2,reward3', [('balance', 1, 3.3, 3.3 - 1e-5),
                                                               ('reward', 1, 2.3, -10 * 1e-6)])
def test_env(optimizer, reward1, reward2, reward3):
    env = FootballDataDailyPercentageEnv(config={"optimize": optimizer, "starting_bank": 10})
    assert env is not None
    assert env._extra_odds is not None
    assert env._extra_odds.shape == (380, 46)
    for column in env.ODDS_COLUMNS:
        assert column in env._extra_odds.columns
    action = numpy.zeros(shape=env.action_space.shape)
    action[0] = 0.1
    obs, reward, done, truncated, info = env.step(action)
    assert reward == reward1
    assert obs.shape == (10, 3)
    action = numpy.zeros(shape=env.action_space.shape)
    action[1] = 1 / 11
    obs, reward, done, truncated, info = env.step(action)
    numpy.testing.assert_almost_equal(reward, reward2)
    action = numpy.zeros(shape=env.action_space.shape)
    action[:] = 0.9
    obs, reward, done, truncated, info = env.step(action)
    numpy.testing.assert_almost_equal(reward, reward3)


def test_extra_env():
    env = FootballDataDailyPercentageEnv(config={"extra": True})
    assert env.observation_space.shape == (10, 42)
    action = numpy.zeros(shape=env.action_space.shape)
    action[0] = 0.1
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (10, 42)


def test_registeration():
    spec_ids = [spec.id for spec in envs.registry.values()]
    assert 'FootballDataDaily-v0' in spec_ids
    assert 'FootballDataDailyPercent-v0' in spec_ids


def test_caching():
    # cache directory is empty, so it will use the CSV from the site
    FootballDataDailyPercentageEnv(config={"start": 2019, "end": 2020})
    # create cache directory
    create_csv_cache(True, SupportedSport.football)
    FootballDataDailyPercentageEnv(config={"start": 2019, "end": 2020})
    shutil.rmtree(CSV_CACHE_PATH)
