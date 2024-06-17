import gymnasium as gym

from .base import BaseOddsEnv
from .base_percentage import BasePercentageOddsEnv
from .daily_bets import DailyOddsEnv, DailyPercentageOddsEnv
from .soccer import ThreeWaySoccerOddsEnv, ThreeWaySoccerPercentageOddsEnv
from .soccer import ThreeWaySoccerDailyOddsEnv, ThreeWaySoccerDailyPercentageOddsEnv
from .footballdata import FootballDataDailyEnv, FootballDataDailyPercentageEnv
from .tennisdata import TennisDataDailyEnv, TennisDataDailyPercentageEnv


__all__ = ['BaseOddsEnv', 'BasePercentageOddsEnv', 'DailyOddsEnv', 'DailyPercentageOddsEnv', 'ThreeWaySoccerOddsEnv',
           'ThreeWaySoccerPercentageOddsEnv', 'ThreeWaySoccerDailyOddsEnv', 'ThreeWaySoccerDailyPercentageOddsEnv',
           'FootballDataDailyEnv', 'FootballDataDailyPercentageEnv',
           'TennisDataDailyEnv', 'TennisDataDailyPercentageEnv']

# register the *-data.co.uk environments, so they would be ready for usage when installing with pip
gym.register(id='FootballDataDaily-v0',
             entry_point='oddsgym.envs:FootballDataDailyEnv',
             max_episode_steps=365)
gym.register(id='FootballDataDailyPercent-v0',
             entry_point='oddsgym.envs:FootballDataDailyPercentageEnv',
             max_episode_steps=365)
gym.register(id='TennisDataDaily-v0',
             entry_point='oddsgym.envs:TennisDataDailyEnv',
             max_episode_steps=365)
gym.register(id='TennisDataDailyPercent-v0',
             entry_point='oddsgym.envs:TennisDataDailyPercentageEnv',
             max_episode_steps=365)

# register the *-data.co.uk environments in ray, only if ray is installed
try:
    from ray import tune
except ImportError:  # pragma: no cover
    pass
else:  # pragma: no cover
    tune.register_env('FootballDataDaily-ray-v0', lambda env_config: gym.wrappers.FlattenObservation(FootballDataDailyEnv(env_config)))
    tune.register_env('FootballDataDailyPercent-ray-v0', lambda env_config: gym.wrappers.FlattenObservation(FootballDataDailyPercentageEnv(env_config)))
    tune.register_env('TennisDataDaily-ray-v0', lambda env_config: gym.wrappers.FlattenObservation(TennisDataDailyEnv(env_config)))
    tune.register_env('TennisDataDailyPercent-ray-v0', lambda env_config: gym.wrappers.FlattenObservation(TennisDataDailyPercentageEnv(env_config)))
