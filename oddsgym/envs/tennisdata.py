import itertools
import os
import numpy
import pandas
import gym
from .tennis import TennisOddsEnv
from ..utils.constants.tennis import CSV_CACHE_PATH, CSV_URL, TOURNAMENTS, SITES, YEARS


class TennisDataMixin(object):
    """Mixin to creating an odds dataframe from www.tennis-data.co.uk"""

    def _create_odds_dataframe(self, tournament, year):
        if tournament not in TOURNAMENTS:
            raise ValueError(f'Tournament {tournament} not supported')
        if year not in YEARS:
            raise ValueError(f'Year {year} not supported')
        if os.path.exists(CSV_CACHE_PATH):
            csvs = [pandas.read_csv(os.path.join(CSV_CACHE_PATH,
                                                 f'{i}',
                                                 f'{tournament}{women}.csv'))
                    for i in range(min(YEARS), year if year != min(YEARS) else year + 1)
                    for women in ['', 'w']]
        else:
            csvs = [pandas.read_csv(CSV_URL.format(tournament=tournament,
                                                   year=i,
                                                   women=women))
                    for i in range(min(YEARS), year if year != min(YEARS) else year + 1)
                    for women in ['', 'w']]
        raw_odds_data = pandas.concat(csvs)
        raw_odds_data['Date'] = pandas.to_datetime(raw_odds_data['Date'], dayfirst=True)
        odds = [''.join(odd) for odd in itertools.product(SITES, ['W', 'L'])
                if ''.join(odd) in raw_odds_data.columns]
        odds_dataframe = raw_odds_data[['Winner', 'Loser', 'Date'] + odds].copy()
        odds_dataframe.rename({'Winner': 'winner', 'Loser': 'loser', 'Date': 'date'},
                              axis='columns', inplace=True)
        odds_dataframe['result'] = 0
        for odd in ['W', 'L']:
            odds_columns = [column for column in odds_dataframe.columns if column.endswith(odd)]
            odds_dataframe[f'Max{odd}'] = odds_dataframe[odds_columns].max(axis='columns')
            odds_dataframe[f'Avg{odd}'] = odds_dataframe[odds_columns].mean(axis='columns')
            odds_dataframe[f'Min{odd}'] = odds_dataframe[odds_columns].min(axis='columns')
            odds_dataframe[f'Median{odd}'] = odds_dataframe[odds_columns].median(axis='columns')
        odds_dataframe
        return odds_dataframe


class TennisDataDailyEnv(TennisDataMixin, TennisOddsEnv):
    sport = 'Tennis, with odds from www.tennis-data.co.uk'
    versionadded = '0.8.0'

    ENV_COLUMNS = ['winner', 'loser', 'date', 'result']
    ODDS_COLUMNS = ['win', 'lose']

    def __init__(self, tournament='ausopen', year=2010, columns='max', extra=False,
                 optimize='reward', *args, **kwargs):
        """Initializes a new environment

        Parameters
        ----------
        tournament: str, defaults to "ausopen"
            The name of the tournament.
        year: int, defaults to 2010
            Year of the tournament.
        columns: {"max", "avg"}, defaults to "max"
            Which columns to use for the odds data. "max" uses the maximum value
            for each odd from all sites, "avg" uses the average value.
        extra: bool, default to False
            Use extra odds for the observation space.
        optimize: {"balance", "reward"}, default to "balance"
            Which type of optimization to use.
        """
        odds_dataframe = self._create_odds_dataframe(tournament, year)
        odds_dataframe.rename({f'{columns.title()}W': 'win',
                               f'{columns.title()}L': 'lose'}, axis='columns', inplace=True)
        super().__init__(odds_dataframe[self.ENV_COLUMNS + self.ODDS_COLUMNS],
                         *args, **kwargs)
        self._extra_odds = odds_dataframe
        self._extra = extra
        self._optimize = optimize
        if self._extra:
            self.observation_space = gym.spaces.Box(low=0., high='Inf',
                                                    shape=(self.observation_space.shape[0],
                                                           self._extra_odds.shape[1] - len(self.ENV_COLUMNS)))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self._extra:
            obs = self.get_extra_odds()
        if self._optimize == 'balance':
            if not info['legal_bet']:
                self.balance -= self.starting_bank * 1e-6
            reward = self.balance - self.starting_bank
        else:
            if not info['legal_bet']:
                reward = -(self.starting_bank * 1e-6)
        return obs, reward, done, info

    def get_extra_odds(self):
        extra_odds = numpy.zeros([*self.observation_space.shape])
        current_odds = self._extra_odds.iloc[self._get_current_index()].drop(self.ENV_COLUMNS, axis='columns').values
        extra_odds[numpy.arange(current_odds.shape[0])] = current_odds
        return extra_odds


class TennisDataDailyPercentageEnv(TennisDataDailyEnv):
    versionadded = '0.8.0'
