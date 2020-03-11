import numpy
from gym import spaces
from .base import BaseOddsEnv


class BasePercentageGamblingEnv(BaseOddsEnv):
    """Base class for sports betting environments supporting non fixed bet size"""

    def __init__(self, odds, odds_column_names, results=None):
        super().__init__(odds, odds_column_names, results)
        self.action_space = spaces.Box(low=numpy.array([0., 0.01]),
                                       high=numpy.array([self.action_space.n - 0.01, 1 / odds.shape[1]]))

    def step(self, action):
        form = int(numpy.floor(action[0]))
        self.single_bet_size = action[1] * self.balance
        return super().step(form)
