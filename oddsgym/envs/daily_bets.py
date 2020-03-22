import gym
import numpy
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
        The observation space shape is (M, N) where N is the number of possible
        outcomes for the game and M is the maximum number of games in a single
        day.

    action_space : gym.spaces.Box
        The action space for the environment.
        The action space shape is (M,), a list of numbers in [0, 2 ** N), represneting
        on what outcomes to place a bet by conversion to a binary represenation,
        where actions[i] is the action for odds[i].

    balance : float
        The current balance of the environment.

    STARTING_BANK : int, default=10
        The starting bank / balance for the environment.

    days : array, dtype=np.datetime64
        Sorted array of the days.

    """

    def __init__(self, odds, odds_column_names, results=None):
        """Initializes a new environment.

        We initialize the days array (because this environment doesn't iterate
        the games a single game at a time, but by a group of games that happened
        in the same date) and calculate the day with the most games in it so we
        that the observation & action spaces will be defined correctly.

        Parameters
        ----------
        odds: dataframe of shape (n_games, n_odds + 1)
            A list of games, with their betting odds and the date in which
            the game occures.

            .. warning::
                Please note that the environment expects the date column
                in the dataframe to be named 'date' (lower cased).

        odds_column_names: list of str
            A list of column names with length == n_odds.

        results: list of int, default=None
            A list of the results, where results[i] is the outcome of odds[i].
        """
        super().__init__(odds.drop('date', 'columns'), odds_column_names, results)
        self._odds_with_dates = odds.copy()
        self.days = odds['date'].unique()
        self.days.sort()
        max_number_of_games = odds.set_index('date').groupby(by='date').size().max()
        self.observation_space = gym.spaces.Box(low=1., high=float('Inf'),
                                                shape=(max_number_of_games, self._odds.shape[1]))
        self.action_space = gym.spaces.Box(low=0,
                                           high=2 ** self._odds.shape[1] - 0.01,
                                           shape=(max_number_of_games,))

    def get_odds(self):
        """Returns the odds for the current step.

        Returns
        -------
        odds : dataframe of shape (max_games, n_odds)
            The odds for the current step.
            If in the current step there were less then max_games, the odds
            dataframe is appended with (max_games - current_games) zeroed rows
            (rows with only 0).
        """
        current_day = self.days[self.current_step]
        current_odds = self._odds.iloc[self._odds_with_dates[self._odds_with_dates['date'] == current_day].index]
        filler_odds = DataFrame(numpy.zeros(numpy.array([*self.observation_space.shape]) -
                                            numpy.array([current_odds.shape[0], 0])),
                                columns=self._odds_columns_names)
        return current_odds.append(filler_odds, ignore_index=True)

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
        full_actions = numpy.zeros([*self.observation_space.shape])
        actions = numpy.concatenate([super(DailyOddsEnv, self).get_bet(numpy.floor(part_action))
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
        current_day = self.days[self.current_step]
        index = self._odds_with_dates[self._odds_with_dates['date'] == current_day].index
        current_results = self._results.iloc[index]
        results = numpy.zeros(shape=(current_results.shape[0], self._odds.shape[1]))
        results[numpy.arange(results.shape[0]), current_results.values] = 1
        filler_results = numpy.zeros(numpy.array([*self.observation_space.shape]) -
                                     numpy.array([current_results.shape[0], 0]))
        return numpy.concatenate([results, filler_results])

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
        return {'action': [self._verbose_actions[act] for act in numpy.floor(action)],
                'current_step': self.current_step,
                'starting_balance': self.balance,
                'odds': self.get_odds(),
                'single_bet_size': self.single_bet_size}
