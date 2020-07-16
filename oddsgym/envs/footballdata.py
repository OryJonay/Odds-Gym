import datetime
import itertools
import numpy
import pandas
import gym
from .soccer import ThreeWaySoccerDailyPercentageOddsEnv, ThreeWaySoccerDailyOddsEnv


class FootballDataMixin(object):
    """Mixin to creating an odds dataframe from www.football-data.co.uk"""

    CSV_URL = 'http://www.football-data.co.uk/mmz4281/{start}{end}/{country}{league}.csv'

    COUNTRIES = {'England': 'E', 'Scotland': 'SC', 'Germany': 'D', 'Italy': 'I', 'Spain': 'SP', 'France': 'F',
                 'Netherlands': 'N', 'Belgium': 'B', 'Portugal': 'P', 'Turkey': 'T', 'Greece': 'G'}

    LEAGUES = {'England': {'Premier League': 0, 'Championship': 1, 'League 1': 2, 'League 2': 3, 'Conference': 'C'},
               'Scotland': {'Premier League': 0, 'Division 1': 1, 'Division 2': 2, 'Division 3': 3},
               'Germany': {'Bundesliga 1': 1, 'Bundesliga 2': 2},
               'Italy': {'Serie A': 1, 'Serie B': 2},
               'Spain': {'Primera Division': 1, 'Segunda Division': 2},
               'France': {'Ligue 1': 1, 'Ligue 2': 2},
               'Netherlands': {'Eredivise': 1},
               'Belgium': {'Jupiler League': 1},
               'Portugal': {'Liga 1': 1},
               'Turkey': {'Super Lig': 1},
               'Greece': {'Super League': 1}}

    SITES = ["B365", "BS", "BW", "GB", "IW", "LB", "PS", "SO", "SB", "SJ", "SY", "VC", "WH", "P"]

    def _create_odds_dataframe(self, country, league, start, end):
        country = country.title()
        league = league.title()
        if country not in self.COUNTRIES:
            raise ValueError(f'Country {country} not supported')
        if league not in self.LEAGUES[country]:
            raise ValueError(f'League {league} not supported')
        if start < 2010:
            raise ValueError(f'Start year must be greater than 2010, start year is {start}')
        if start > end:
            raise ValueError(f'Start year can not be greater than end year')
        if end > datetime.datetime.today().year:
            raise ValueError(f'End year can not be greater than current year')
        raw_odds_data = pandas.concat([pandas.read_csv(self.CSV_URL.format(country=self.COUNTRIES[country],
                                                                           league=self.LEAGUES[country][league],
                                                                           start=str(i).zfill(2),
                                                                           end=str(i + 1).zfill(2)))
                                       for i in range(int(str(start)[-2:]), int(str(end)[-2:]))])
        raw_odds_data['Date'] = pandas.to_datetime(raw_odds_data['Date'], dayfirst=True)
        odds = [''.join(odd) for odd in itertools.product(self.SITES, ['H', 'D', 'A'])
                if ''.join(odd) in raw_odds_data.columns]
        odds_dataframe = raw_odds_data[['HomeTeam', 'AwayTeam', 'Date'] + odds].copy()
        odds_dataframe.rename({'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Date': 'date'},
                              axis='columns', inplace=True)
        odds_dataframe['result'] = raw_odds_data['FTR'].map({'H': 0, 'A': 2, 'D': 1})
        odds_dataframe.dropna(subset=['result'], inplace=True)
        odds_dataframe['result'] = odds_dataframe['result'].astype(int)
        for odd in ['H', 'D', 'A']:
            odds_columns = [column for column in odds_dataframe.columns if column.endswith(odd)]
            odds_dataframe[f'Max{odd}'] = odds_dataframe[odds_columns].max(axis='columns')
            odds_dataframe[f'Avg{odd}'] = odds_dataframe[odds_columns].mean(axis='columns')
            odds_dataframe[f'Min{odd}'] = odds_dataframe[odds_columns].min(axis='columns')
            odds_dataframe[f'Median{odd}'] = odds_dataframe[odds_columns].median(axis='columns')
        odds_dataframe
        return odds_dataframe


class FootballDataDailyEnv(FootballDataMixin, ThreeWaySoccerDailyOddsEnv):
    """Daily environment that uses data from from www.football-data.co.uk

    .. versionadded:: 0.8.0"""

    ENV_COLUMNS = ['home_team', 'away_team', 'date', 'result']
    ODDS_COLUMNS = ['home', 'draw', 'away']

    def __init__(self, country='England', league='Premier League', start=2010, end=2011, columns='max', extra=False,
                 optimize='reward', *args, **kwargs):
        """Initializes a new environment

        Parameters
        ----------
        country: str, defaults to "England"
            The name of the country in which the league is playing.
        league: str, defaults to "Premier League"
            The name of the league.
        start: int, defaults to 2010
            Start year of the league to use.
        end: int, defaults to 2011
            End year of the league to use.
        columns: {"max", "avg"}, defaults to "max"
            Which columns to use for the odds data. "max" uses the maximum value
            for each odd from all sites, "avg" uses the average value.
        extra: bool, default to False
            Use extra odds for the observation space.
        optimize: {"balance", "reward"}, default to "balance"
            Which type of optimization to use.
        """
        odds_dataframe = self._create_odds_dataframe(country, league, start, end)
        odds_dataframe.rename({f'{columns.title()}H': 'home',
                               f'{columns.title()}D': 'draw',
                               f'{columns.title()}A': 'away'}, axis='columns', inplace=True)
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
                self.balance -= self.STARTING_BANK * 1e-6
            reward = self.balance - self.STARTING_BANK
        else:
            if not info['legal_bet']:
                reward = -(self.STARTING_BANK * 1e-6)
        return obs, reward, done, info

    def get_extra_odds(self):
        extra_odds = numpy.zeros([*self.observation_space.shape])
        current_odds = self._extra_odds.iloc[self._get_current_index()].drop(self.ENV_COLUMNS, axis='columns').values
        extra_odds[numpy.arange(current_odds.shape[0])] = current_odds
        return extra_odds


class FootballDataDailyPercentageEnv(FootballDataDailyEnv, ThreeWaySoccerDailyPercentageOddsEnv):
    """Daily percentage environment that uses data from from www.football-data.co.uk

    .. versionadded:: 0.6.2"""

    pass
