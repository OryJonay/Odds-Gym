import gym
import numpy
from more_itertools import powerset
from pandas import DataFrame


class BaseOddsEnv(gym.Env):
    """Base class for sports betting environments."""

    metadata = {'render.modes': ['human']}
    STARTING_BANK = 10

    def __init__(self, odds, odds_column_names, results=None):
        """Initialization function

        Args:
            odds: A numpy array of size (number_of_games, number_of_possible_bets).
                  A list of games, with their betting odds.
            odds_column_names: A list of length odds.shape[1] (number_of_possible_bets),
                               giving the name of the bet (i.e ['1', '2', 'X'] or ['home', 'away', 'draw']).
            results: A numpy array of size (number_of_games, 1).
                     A list of the winner of each game (encoded like the odds,
                     i.e result '0' for winning odds[0]).
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
        return DataFrame([self._odds[self.current_step]])

    def get_bet(self, action):
        verbose_actions = self._verbose_actions[action]
        bet = numpy.array([[int(name in verbose_action) for name in self._odds_columns_names]
                           for verbose_action in verbose_actions])
        return bet

    def step(self, action):
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
        self.balance = self.STARTING_BANK
        self.current_step = 0
        return self.get_odds()

    def render(self, mode='human'):
        return 'Current balance at step {}: {}'.format(self.current_step, self.balance)

    def finish(self):
        return self.current_step == self._odds.shape[0]  # no more games left to bet

    def get_results(self):
        result = numpy.zeros(shape=self.observation_space.shape)
        result[numpy.arange(result.shape[0]), numpy.array([self._results[self.current_step]])] = 1
        return result

    def legal_bet(self, bet):
        return numpy.count_nonzero(bet) * self.single_bet_size <= self.balance

    def create_info(self, action):
        return {'action': self._verbose_actions[action], 'current_step': self.current_step,
                'starting_balance': self.balance, 'odds': self.get_odds(),
                'single_bet_size': self.single_bet_size}
