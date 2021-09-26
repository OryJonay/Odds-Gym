import gym
import numpy
import numexpr
from pandas import DataFrame
from .base import BaseOddsEnv


class DailyOddsEnv(BaseOddsEnv):
    """Base class for sports betting environments multiple games with a fixed
    bet size.

    Creates an OpenAI Gym environment that supports betting a fixed amount (1)
    on a single outcome but for multiple games.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    observation_space : gym.spaces.Box
        The observation space for the environment.
        The observation space shape is (M, N) where M is the maximum number of games in a single
        day and N is the number of possible outcomes for the game.

    action_space : gym.spaces.Box
        The action space for the environment.
        The action space shape is (M,), a list of numbers in [0, 2 ** N), representing
        on what outcomes to place a bet by conversion to a binary representation,
        where actions[i] is the action for odds[i].

        .. versionchanged:: 0.6.0
            Change action space bounds to [-1, 1] and rescale the action back
            inside the step method.
            For explanation on how the rescaling works, see :py:class:`~oddsgym.envs.base_percentage.BasePercentageOddsEnv`

    balance : float
        The current balance of the environment.

    starting_bank : int, default=10
        The starting bank / balance for the environment.

    days : array, dtype=np.datetime64
        Sorted array of the days.

    """

    HEADERS = ['Date', 'Current Step', 'Odds', 'Verbose Action', 'Action', 'Balance', 'Reward', 'Results', 'Done']

    def __init__(self, odds, odds_column_names, results=None, max_number_of_games='auto', *args, **kwargs):
        """Initializes a new environment.

        We initialize the days array (because this environment doesn't iterate
        the games a single game at a time, but by a group of games that happened
        in the same date) and calculate the day with the most games in it so
        that the observation & action spaces will be defined correctly.

        Parameters
        ----------
        odds: dataframe of shape (n_games, n_odds + 1)
            A list of games, with their betting odds and the date in which
            the game occurs.

            .. warning::
                Please note that the environment expects the date column
                in the dataframe to be named 'date' (lower cased).

        odds_column_names: list of str
            A list of column names with length == n_odds.

        results: list of int, default=None
            A list of the results, where results[i] is the outcome of odds[i].

        max_number_of_games: int or str, default='auto'
            The maximum number of games in a single name that the environment
            will support. The value 'auto' will calculate this value from the
            odds dataframe.
        """
        super().__init__(odds.drop('date', axis='columns'), odds_column_names, results, *args, **kwargs)
        self._odds_with_dates = odds.copy()
        self.days = odds['date'].unique()
        self.days.sort()
        self._max_number_of_games = None
        if max_number_of_games == 'auto':
            self.max_number_of_games = odds.set_index('date').groupby(by='date').size().max()
        elif isinstance(max_number_of_games, int) and max_number_of_games > 0:
            self.max_number_of_games = max_number_of_games
        else:
            raise ValueError("Invalid value for max_number_of_games: {}.\nPass 'auto' for automatic "
                             "calculation of the maximum number of games"
                             ", or an integer higher than 0".format(max_number_of_games))
        self.observation_space = gym.spaces.Box(low=0., high=float('Inf'),
                                                shape=(self.max_number_of_games, self._odds.shape[1]),
                                                dtype=numpy.float64)
        self.action_space = gym.spaces.Box(low=-1,
                                           high=1,
                                           shape=(self.max_number_of_games,))
        self.bet_size_matrix = numpy.ones(shape=self.observation_space.shape)

    def _get_current_index(self):
        current_day = self.days[self.current_step]
        return self._odds_with_dates[self._odds_with_dates['date'] == current_day].index

    def get_odds(self):
        """Returns the odds for the current step.

        Returns
        -------
        odds : numpy.ndarray of shape (max_games, n_odds)
            The odds for the current step.
            If in the current step there were less then max_games, the odds
            dataframe is appended with (max_games - current_games) zeroed rows
            (rows with only 0).
        """
        current_odds = self._odds.iloc[self._get_current_index()]
        filler_odds = DataFrame(numpy.zeros(numpy.array([self.max_number_of_games, self._odds.shape[1]]) -
                                            numpy.array([current_odds.shape[0], 0])),
                                columns=self._odds_columns_names)
        return current_odds.append(filler_odds, ignore_index=True).values

    def get_bet(self, action):
        """Returns the betting matrix for the action provided.

        Parameters
        ----------
        action : array of shape (max_games, )
            An action provided by the agent.

        Returns
        -------
        bet : array of shape (max_games, n_odds)
            The betting matrix, where for each row, each outcome specified in
            action[row] has a value of 1 and 0 otherwise.
        """
        full_actions = numpy.zeros(shape=(self.max_number_of_games, self._odds.shape[1]))
        actions = numpy.concatenate([super(DailyOddsEnv, self).get_bet(numpy.floor(part_action).astype(int))
                                     for part_action in action])
        full_actions[numpy.arange(actions.shape[0])] = actions
        return full_actions

    def get_results(self):
        """Returns the results matrix for the current step.

        Returns
        -------
        results : array of shape (max_games, n_odds)
            The results matrix, where for each row (single game),
            the index of the outcome that happened value is 1 and the rest of
            the indexes values are 0.
            If in the current step there were less then max_games, the result
            matrix is appended with (max_games - current_games) zeroed rows
            (rows with only 0).
        """
        index = self._get_current_index()
        current_results = self._results.iloc[index]
        results = numpy.zeros(shape=(current_results.shape[0], self._odds.shape[1]))
        results[numpy.arange(results.shape[0]), current_results.values] = 1
        filler_results = numpy.zeros(numpy.array([self.max_number_of_games, self._odds.shape[1]]) -
                                     numpy.array([current_results.shape[0], 0]))
        return numpy.concatenate([results, filler_results])

    def get_reward(self, bet, odds, results):
        """ Calculates the reward, while taking to account invalid bets

        Parameters
        ----------
        bet : array of shape (n_games, n_odds)
        odds: dataframe of shape (n_games, n_odds)
            A list of games, with their betting odds.
        results : array of shape (max_games, n_odds)

        Returns
        -------
        reward : float
            The amount of reward returned after previous action
        """
        used_results = numpy.ones_like(results)
        zero_rows_count = numpy.sum(~results.any(1))
        if zero_rows_count > 0:
            used_results[-zero_rows_count:, :] = 0
        bet_size_matrix = self.bet_size_matrix  # noqa: F841
        reward = numexpr.evaluate('sum(bet * bet_size_matrix * results * odds)')
        expense = numexpr.evaluate('sum(bet * used_results * bet_size_matrix)')
        return reward - expense

    def legal_bet(self, bet):
        """Checks if the bet is legal, while taking to account invalid bets.

        Checks that the bet does not exceed the current balance.

        Parameters
        ----------
        bet : array of shape (n_games, n_odds)
            The bet to check.

        Returns
        -------
        legal : bool
            True if the bet is legal, False otherwise.
        """
        results = self.get_results()
        used_results = numpy.ones_like(results)
        zero_rows_count = numpy.sum(~results.any(1))
        if zero_rows_count > 0:
            used_results[-zero_rows_count:, :] = 0
        return (bet * used_results * self.bet_size_matrix).sum() <= self.balance

    def finish(self):
        """Checks if the episode has reached an end.

        The episode has reached an end if there are no more days left.

        Returns
        -------
        finish : bool
            True if the current_step is equal to len(self.days), False otherwise
        """
        return self.current_step == self.days.shape[0]  # no more days left to bet

    def create_info(self, action):
        """Creates the info dictionary for the given action.

        The info dictionary holds the following information:
            * the verbose actions
            * the current step
            * the balance at the start of the current step
            * the relevant odds for the current step
            * the bet size for a single outcome

        Parameters
        ----------
        action : array of shape (max_games,)
            An action provided by the agent.

        Returns
        -------
        info : dict
            The info dictionary.
        """
        return {'date': self.days[self.current_step], 'current_step': self.current_step,
                'odds': self.get_odds(),
                'verbose_action': [self._verbose_actions[act] for act in numpy.floor(action).astype(int)],
                'action': action,
                'balance': self.balance, 'reward': 0,
                'legal_bet': False, 'results': None, 'done': False}

    def step(self, action):
        return super().step(numpy.array([self._rescale_form(form) for form in action]))


class DailyPercentageOddsEnv(DailyOddsEnv):
    """Base class for sports betting environments multiple games with a non fixed
    bet size.

    Creates an OpenAI Gym environment that supports betting a non fixed amount
    on a single outcome but for multiple games.

    .. versionadded:: 0.5.0

    Parameters
    ----------
    action_space: gym.spaces.Box of shape (n_games * n_odds,)
        A vector with the action and the percentage for each outcome, so that:

        .. math::

            a[i] = \\begin{cases}
                \\text{the action} & i\\mod\\text{(n_odds + 1)}\\equiv 0\\\\
                \\text{percentage of outcome j} & i\\mod\\text{(n_odds + 1)}\\equiv j
            \\end{cases}

        For the game :math:`\\lfloor \\frac{i}{\\text{n_games}} \\rfloor`.

        .. versionchanged:: 0.6.0
            Change action space bounds to [-1, 1] and rescale the action back
            inside the step method.
            For explanation on how the rescaling works, see :py:class:`~oddsgym.envs.base_percentage.BasePercentageOddsEnv`
        .. versionchanged:: 0.7.0
            Reduce dimesionality of action space by deducing action from
            betting percentages

    """

    def __init__(self, odds, odds_column_names, results=None, *args, **kwargs):
        super().__init__(odds, odds_column_names, results, *args, **kwargs)
        lower_bound = [[-1] * self._odds.shape[1] for i in numpy.arange(self.max_number_of_games)]
        upper_bound = [[1] * self._odds.shape[1] for i in numpy.arange(self.max_number_of_games)]
        vector_size = self.max_number_of_games * self._odds.shape[1]
        self.action_space = gym.spaces.Box(low=numpy.array(lower_bound).reshape(vector_size),
                                           high=numpy.array(upper_bound).reshape(vector_size))

    def step(self, action):
        full_action = numpy.zeros(self.max_number_of_games * (self._odds.shape[1]))
        full_action[numpy.arange(action.shape[0])] = action
        full_action = full_action.reshape(self.max_number_of_games, (self._odds.shape[1]))

        current_bet_size_matrix = self._rescale_matrix(full_action) * self.balance
        full_bet_size_matrix = numpy.zeros(shape=(self.max_number_of_games, self._odds.shape[1]))
        full_bet_size_matrix[numpy.arange(current_bet_size_matrix.shape[0])] = current_bet_size_matrix

        self.bet_size_matrix = full_bet_size_matrix

        form_binary_repr = numpy.where(self.bet_size_matrix != 0, 1, 0)
        explicit_forms = form_binary_repr.dot(1 << numpy.arange(form_binary_repr.shape[-1] - 1, -1, -1))
        forms = numpy.array([(form / (2 ** (self._odds.shape[1] - 1)) - 1) for form in explicit_forms])

        return super().step(forms)
