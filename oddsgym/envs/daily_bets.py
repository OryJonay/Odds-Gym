import gym
import numpy
from pandas import DataFrame
from .base import BaseOddsEnv

class DailyOddsEnv(BaseOddsEnv):

    def __init__(self, odds, odds_column_names, results=None):
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
        current_day = self.days[self.current_step]
        current_odds = self._odds.iloc[self._odds_with_dates[self._odds_with_dates['date'] == current_day].index]
        filler_odds = DataFrame(numpy.zeros(numpy.array([*self.observation_space.shape]) -
                                            numpy.array([current_odds.shape[0], 0])),
                                columns=self._odds_columns_names)
        return current_odds.append(filler_odds, ignore_index=True)

    def get_bet(self, action):
        full_actions = numpy.zeros([*self.observation_space.shape])
        actions = numpy.concatenate([super(DailyOddsEnv, self).get_bet(numpy.floor(part_action))
                                     for part_action in action])
        full_actions[numpy.arange(actions.shape[0])] = actions
        return full_actions

    def get_results(self):
        current_day = self.days[self.current_step]
        index = self._odds_with_dates[self._odds_with_dates['date'] == current_day].index
        current_results = self._results.iloc[index]
        results = numpy.zeros(shape=(current_results.shape[0], self._odds.shape[1]))
        results[numpy.arange(results.shape[0]), current_results.values] = 1
        filler_results = numpy.zeros(numpy.array([*self.observation_space.shape]) -
                                     numpy.array([current_results.shape[0], 0]))
        return numpy.concatenate([results, filler_results])

    def finish(self):
        return self.current_step == self.days.shape[0]  # no more days left to bet

    def create_info(self, action):
        return {'action': [self._verbose_actions[act] for act in numpy.floor(action)],
                'current_step': self.current_step,
                'starting_balance': self.balance,
                'odds': self.get_odds(),
                'single_bet_size': self.single_bet_size}
