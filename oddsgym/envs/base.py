import gym
import numpy
from more_itertools import powerset
from pandas import DataFrame


class BaseOddsEnv(gym.Env):
    """Base class for sports betting environments.

    Creates an OpenAI Gym environment that supports betting a fixed amount (1)
    on a single outcome for a single game.

    .. versionadded: 0.1.0

    Parameters
    ----------
    observation_space : gym.spaces.Box
        The observation space for the environment.
        The observation space shape is (1, N) where N is the number of possible
        outcomes for the game.

        .. versionchanged: 0.3.0
            Changed definition of space

    action_space : gym.spaces.Discrete
        The action space for the environment.
        The action space is a single number in [0, 2 ** N), represneting
        on what outcomes to place a bet by conversion to a binary represenation.

        .. versionchanged: 0.3.0
            Changed definition of space

    balance : float
        The current balance of the environment.

    STARTING_BANK : int, default=10
        The starting bank / balance for the environment.
    """

    metadata = {'render.modes': ['human']}
    STARTING_BANK = 10

    def __init__(self, odds, odds_column_names, results=None):
        """Initializes a new environment

        Parameters
        ----------
        odds: dataframe of shape (n_games, n_odds)
            A list of games, with their betting odds.
        odds_column_names: list of str
            A list of column names with length == n_odds.
        results: list of int, default=None
            A list of the results, where results[i] is the outcome of odds[i].
        """

        super().__init__()
        self._odds = odds.copy()
        self._results = results
        self._odds_columns_names = odds_column_names
        self._verbose_actions = {act: [verbose_act] for verbose_act, act in zip(list(powerset(odds_column_names)),
                                                                                list(range(2 ** odds.shape[1])))}
        self.observation_space = gym.spaces.Box(low=1., high=float('Inf'), shape=(1, odds.shape[1]))
        self.action_space = gym.spaces.Discrete(2 ** odds.shape[1])
        self.balance = self.STARTING_BANK
        self.current_step = 0
        self.single_bet_size = 1

    def get_odds(self):
        """Returns the odds for the current step.

        Returns
        -------
        odds : dataframe of shape (1, n_odds)
            The odds for the current step.
        """
        return DataFrame([self._odds[self.current_step]])

    def get_bet(self, action):
        """Returns the betting matrix for the action provided.

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        bet : array of shape (1, n_odds)
            The betting matrix, where each outcome specified in the action
            has a value of 1 and 0 otherwise.
        """
        verbose_actions = self._verbose_actions[action]
        bet = numpy.array([[int(name in verbose_action) for name in self._odds_columns_names]
                           for verbose_action in verbose_actions])
        return bet

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of episode is reached,
        you are responsible for calling reset() to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        observation : dataframe of shape (1, n_odds)
            The agent's observation of the current environment
        reward : float
            The amount of reward returned after previous action
        done : bool
            Whether the episode has ended, in which case further step() calls will return undefined results
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        odds = self.get_odds()
        reward = 0
        done = False
        single_bet_size = self.single_bet_size
        info = self.create_info(action)
        if self.balance < 1:  # no more money :-(
            done = True
        else:
            bet = self.get_bet(action)
            if self.legal_bet(bet):  # making sure agent has enough money for the bet
                results = self.get_results()
                reward = ((bet * results * odds).values.sum() * single_bet_size) - \
                    (numpy.count_nonzero(bet) * single_bet_size)
                self.balance += reward
                info.update({'results': results.argmax()})
                self.current_step += 1
                if self.finish():
                    done = True
                    self.current_step = 0
            else:
                reward = -numpy.inf
        return self.get_odds(), reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns
        -------
        observation : dataframe of shape (1, n_odds)
            the initial observation.
        """
        self.balance = self.STARTING_BANK
        self.current_step = 0
        return self.get_odds()

    def render(self, mode='human'):
        """Outputs the current balance and the current step.

        Returns
        -------
        msg : str
            A string with the current balance and the current step.
        """
        return 'Current balance at step {}: {}'.format(self.current_step, self.balance)

    def finish(self):
        """Checks if the episode has reached an end.

        The episode has reached an end if there are no more games to bet.

        Returns
        -------
        finish : bool
            True if the current_step is equal to n_games, False otherwise
        """
        return self.current_step == self._odds.shape[0]  # no more games left to bet

    def get_results(self):
        """Returns the results matrix for the current step.

        Returns
        -------
        result : array of shape (1, n_odds)
            The result matrix, where the index of the outcome that happened
            value is 1 and the rest of the indexes values are 0.
        """
        result = numpy.zeros(shape=self.observation_space.shape)
        result[numpy.arange(result.shape[0], dtype=numpy.int32),
               numpy.array([self._results[self.current_step]], dtype=numpy.int32)] = 1
        return result

    def legal_bet(self, bet):
        """Checks if the bet is legal.

        Checks that the bet does not exceed the current balance.

        Parameters
        ----------
        bet : array of shape (1, n_odds)
            The bet to check.

        Returns
        -------
        legal : bool
            True if the bet is legal, False otherwise.
        """
        return numpy.count_nonzero(bet) * self.single_bet_size <= self.balance

    def create_info(self, action):
        """Creates the info dictionary for the given action.

        The info dictionary holds the following information:
            * the verbose action
            * the current step
            * the balance at the start of the current step
            * the relevant odds for the current step
            * the bet size for a single outcome

        Parameters
        ----------
        action : int
            An action provided by the agent.

        Returns
        -------
        info : dict
            The info dictionary.
        """
        return {'action': self._verbose_actions[action], 'current_step': self.current_step,
                'starting_balance': self.balance, 'odds': self.get_odds(),
                'single_bet_size': self.single_bet_size}
