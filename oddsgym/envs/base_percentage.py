import numpy
from gym import spaces
from .base import BaseOddsEnv


class BasePercentageOddsEnv(BaseOddsEnv):
    """Base class for sports betting environments with non fixed bet size.

    Creates an OpenAI Gym environment that supports betting a non fixed amount
    on a single outcome for a single game.

    The main difference between the BaseOddsEnv is that the action space is defined
    differently (to accommodate that non fixed bet size).

    .. versionadded:: 0.2.0
    .. deprecated:: 0.4.0
        This environment will be renamed "BasePercentageOddsEnv" in 0.4.5
    .. versionchanged:: 0.4.5
        Name changed to "BasePercentageOddsEnv"

    Parameters
    ----------
    action_space : gym.spaces.Box of shape (2,)
        A 2-tuple, where the first index is the action itself and the second
        index is the percentage of the current balance to place on all the outcomes
        specified in the action.

    """

    def __init__(self, odds, odds_column_names, results=None):
        super().__init__(odds, odds_column_names, results)
        self.action_space = spaces.Box(low=numpy.array([0., 0.01]),
                                       high=numpy.array([self.action_space.n - 0.01, 1 / odds.shape[1]]))

    def step(self, action):
        form = int(numpy.floor(action[0]))
        self.single_bet_size = action[1] * self.balance
        return super().step(form)
