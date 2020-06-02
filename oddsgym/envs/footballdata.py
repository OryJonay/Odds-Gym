import datetime
import pandas
from .soccer import ThreeWaySoccerDailyPercentageOddsEnv

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

        odds_dataframe = raw_odds_data[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'Date']].copy()
        odds_dataframe.rename({'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'B365H': 'home',
                               'B365A': 'away', 'B365D': 'draw', 'Date': 'date'}, axis='columns', inplace=True)
        odds_dataframe['result'] = raw_odds_data['FTR'].map({'H': 0, 'A': 2, 'D': 1})
        odds_dataframe.dropna(subset=['result'], inplace=True)
        odds_dataframe['result'] = odds_dataframe['result'].astype(int)
        return odds_dataframe

class FootballDataDailyPercentageEnv(FootballDataMixin, ThreeWaySoccerDailyPercentageOddsEnv):
    """Daily percentage environment that uses data from from www.football-data.co.uk

    .. versionadded:: 0.6.2"""

    def __init__(self, country='England', league='Premier League', start=2010, end=2011):
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
        """

        super().__init__(self._create_odds_dataframe(country, league, start, end))
