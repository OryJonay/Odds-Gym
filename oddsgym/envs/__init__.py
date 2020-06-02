import gym

from .base import BaseOddsEnv
from .base_percentage import BasePercentageOddsEnv
from .daily_bets import DailyOddsEnv, DailyPercentageOddsEnv
from .soccer import ThreeWaySoccerOddsEnv, ThreeWaySoccerPercentageOddsEnv
from .soccer import ThreeWaySoccerDailyOddsEnv, ThreeWaySoccerDailyPercentageOddsEnv
from .footballdata import FootballDataDailyPercentageEnv

__all__ = ['BaseOddsEnv', 'BasePercentageOddsEnv', 'DailyOddsEnv', 'DailyPercentageOddsEnv', 'ThreeWaySoccerOddsEnv',
           'ThreeWaySoccerPercentageOddsEnv', 'ThreeWaySoccerDailyOddsEnv', 'ThreeWaySoccerDailyPercentageOddsEnv',
           'FootballDataDailyPercentageEnv']

# register the football-data.co.uk environment, so it would be ready for usage when installing with pip
gym.register(id='FootballDataDailyPercent-v0',
             entry_point='oddsgym.envs:FootballDataDailyPercentageEnv',
             max_episode_steps=365)
