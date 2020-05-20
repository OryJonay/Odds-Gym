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
    .. versionchanged:: 0.4.5
        Name changed to "BasePercentageOddsEnv"

    Parameters
    ----------
    action_space : gym.spaces.Box of shape (N+1,), where N is the number of possible
        outcomes for the game.
        The (N+1)-tuple first index is the action itself and the rest of the
        indexes are the percentage of the current balance to place on matching
        outcome, so that action[i + 1] is the bet percentage for outcome[i].

        .. versionchanged:: 0.5.0
            Change action space so that each outcome has it's own independent bet
            percentage.
        .. versionchanged:: 0.6.0
            Change action space bounds to [-1, 1] and rescale the action back
            inside the step method.

        The rescaling an action :math:`(a, p_0, ..., p_{N-1}) \in \\text{action_space}, -1 \\leq a, p_0, ..., p_{N-1} \\leq 1`:

        .. math::
            \\begin{cases}
                a' = \\lfloor (a + 1) * (2^{N-1}) \\rfloor\\\\
                p_i' = |p_i|
            \\end{cases}

    """

    def __init__(self, odds, odds_column_names, results=None):
        super().__init__(odds, odds_column_names, results)
        self.bet_size_matrix = None
        self.action_space = spaces.Box(low=numpy.array([-1] * (odds.shape[1] + 1)),
                                       high=numpy.array([1.] * (odds.shape[1] + 1)))

    def step(self, action):
        form = self._rescale_form(action[0])
        self.bet_size_matrix = self._rescale_matrix(numpy.array(action[1:])) * self.balance
        return super().step(form)

    def legal_bet(self, bet):
        legal_percentage_bet = numpy.logical_xor(bet, numpy.where(self.bet_size_matrix != 0, 1, 0)).sum() == 0
        return legal_percentage_bet and super().legal_bet(bet)
