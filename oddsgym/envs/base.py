import gym
import numpy
from more_itertools import powerset

class BaseOddsEnv(gym.Env):
    """Base class for sports betting environments."""

    metadata = {'render.modes': ['human']}
    STARTING_BANK = 1

    def __init__(self, odds, odds_column_names, results=None):
        """Initialization function

        Args:
            odds: A numpy array of size (number_of_games, number_of_possible_bets). A list of games, with their betting odds.
            odds_column_names: A list of length odds.shape[1] (number_of_possible_bets), giving the name of the bet (i.e ['1', '2', 'X'] or ['home', 'away', 'draw']).
            results: A numpy array of size (number_of_games, 1). A list of the winner of each game (encoded like the odds, i.e result '0' for winning odds[0]).
        """

        super().__init__(self)
        self._odds = odds
        self._results = results
        self._verbose_actions = {act: verbose_act for verbose_act, act in zip(list(powerset(odds_column_names)),
                                                                              list(powerset(range(odds.shape[1]))))}
        self._actions = list(self._verbose_actions.keys())
        self.balance = self.STARTING_BANK
        self.current_step = 0

    def step(self, action):
        observation = self._odds[self.current_step]
        reward = -float('Inf')
        done = False
        info = {'action': action, 'current_step': self.current_step, 'balance': self.balance}
        if self.balance < 1:  # no more \ not enough money :-(
            done = True
        else:
            bet = numpy.zeros(len(observation))
            bet.put(action, 1)
            if self._results:
                result = self._results[self.current_step]
                current_odds = self._odds[self.current_step]
                reward = (bet * result * current_odds).sum() - numpy.count_nonzero(bet)
                self.balance += reward
            self.current_step += 1
        return observation, reward, done, info

    def reset(self):
        self.balance = self.STARTING_BANK
        self.current_step = 0

    def render(self, mode='human'):
        print('Current Balance at step {}: {}'.format(self.current_step, self.balance))
