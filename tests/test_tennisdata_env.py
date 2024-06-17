import shutil
import pytest
import numpy
from gymnasium import envs
from oddsgym.envs.tennisdata import TennisDataDailyPercentageEnv
from oddsgym.utils.constants import SupportedSport
from oddsgym.utils.constants.tennis import CSV_CACHE_PATH
from oddsgym.utils.csv_downloader import create_csv_cache


@pytest.mark.parametrize('kwargs', [{'year': 1999, 'tournament': 'ausopen'},
                                    {'year': 2010, 'tournament': 'masters'},
                                    {'year': 2020, 'tournament': 'ausopen'}])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        TennisDataDailyPercentageEnv(config=kwargs)


@pytest.mark.parametrize('optimizer,reward1,reward2,reward3', [('balance', 0.07, -0.84545454, -0.84546454),
                                                               ('reward', 0.07, -0.91545454, -10 * 1e-6)])
def test_env(optimizer, reward1, reward2, reward3):
    env = TennisDataDailyPercentageEnv(config={"optimize": optimizer, "starting_bank": 10})
    assert env is not None
    assert env._extra_odds is not None
    assert env._extra_odds.shape == (254, 24)
    for column in env.ODDS_COLUMNS:
        assert column in env._extra_odds.columns
    action = numpy.zeros(shape=env.action_space.shape)
    action[0] = 0.1
    obs, reward, done, truncated, info = env.step(action)
    numpy.testing.assert_almost_equal(reward, reward1)
    assert obs.shape == (88, 2)
    action = numpy.zeros(shape=env.action_space.shape)
    action[1] = 1 / 11
    obs, reward, done, truncated, info = env.step(action)
    numpy.testing.assert_almost_equal(reward, reward2)
    action = numpy.zeros(shape=env.action_space.shape)
    action[:] = 0.9
    obs, reward, done, truncated, info = env.step(action)
    numpy.testing.assert_almost_equal(reward, reward3)


def test_extra_env():
    env = TennisDataDailyPercentageEnv(config={"extra": True})
    assert env.observation_space.shape == (88, 20)
    action = numpy.zeros(shape=env.action_space.shape)
    action[0] = 0.1
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (88, 20)


def test_registeration():
    spec_ids = [spec.id for spec in envs.registry.values()]
    assert 'TennisDataDaily-v0' in spec_ids
    assert 'TennisDataDailyPercent-v0' in spec_ids


def test_caching():
    # cache directory is empty, so it will use the CSV from the site
    TennisDataDailyPercentageEnv(config={})
    # create cache directory
    create_csv_cache(False, SupportedSport.tennis)
    TennisDataDailyPercentageEnv(config={})
    shutil.rmtree(CSV_CACHE_PATH)
