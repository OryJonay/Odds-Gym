import gym
import numpy
from more_itertools import powerset

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
        self._odds = odds
        self._results = results
        self._odds_columns_names = odds_column_names
        self._verbose_actions = {act: verbose_act for verbose_act, act in zip(list(powerset(odds_column_names)),
                                                                              list(range(2 ** odds.shape[1])))}
        self.observation_space = gym.spaces.Box(low=1., high=float('Inf'), shape=(odds.shape[1],))
        self.action_space = gym.spaces.Discrete(2 ** odds.shape[1])
        self.balance = self.STARTING_BANK
        self.current_step = 0

    def get_odds(self):
        return self._odds[self.current_step]

    def step(self, action):
        odds = self.get_odds()
        reward = 0
        done = False
        info = {'action': self._verbose_actions[action], 'current_step': self.current_step, 'starting_balance': self.balance,
                'odds': odds}
        if self.balance < 1:  # no more money :-(
            done = True
        if self.current_step == self._odds.shape[0]:  # no more games to bet
            done = True
            self.current_step = 0
        else:
            verbose_action = self._verbose_actions[action]
            bet = numpy.array([int(name in verbose_action) for name in self._odds_columns_names])
            if numpy.count_nonzero(bet) <= self.balance:  # making sure agent has enough money for the bet
                result = numpy.zeros_like(bet)
                result.put((self._results[self.current_step],), 1)
                reward = (bet * result * odds).sum() - numpy.count_nonzero(bet)
                self.balance += reward
                info.update({'result': result.argmax()})
                self.current_step += 1
            else:
                reward = -numpy.count_nonzero(bet)
        return self.get_odds(), reward, done, info

    def reset(self):
        self.balance = self.STARTING_BANK
        self.current_step = 0
        return self.get_odds()

    def render(self, mode='human'):
        return 'Current balance at step {}: {}'.format(self.current_step, self.balance)
