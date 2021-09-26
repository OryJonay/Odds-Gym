import pandas
from .meta import MetaEnvBuilder


class ThreeWaySoccerOddsEnv(metaclass=MetaEnvBuilder):
    sport = '3-way soccer'
    versionadded = '0.1.0'
    odds_column_names = ['home', 'draw', 'away']

    def __init__(self, soccer_bets_dataframe, *args, **kwargs):
        """Initializes a new environment.

        Parameters
        ----------
        soccer_bets_dataframe: dataframe of shape (n_games, n_odds + 3), n_odds == 3
            A list of games, with their betting odds, results and the teams playing.

            .. warning::

                Please make sure that the odds columns are named "home, draw, away",
                that the results column is named "result" and that the team names
                columns are named "home_team, away_team" respectively.
        """
        odds = soccer_bets_dataframe[self.odds_columns]
        results = soccer_bets_dataframe['result'] if soccer_bets_dataframe['result'].notna().all() else None
        if not self.daily_env:
            odds = odds.values
            results = results.values
        self.teams = soccer_bets_dataframe[['home_team', 'away_team']]
        super().__init__(odds, self.odds_column_names, results, *args, **kwargs)

    def render(self, mode='human'):
        """Outputs the current team names, balance and step.

        Returns
        -------
        msg : str
            A string with the current team names, balance and step.
        """
        index = self._get_current_index()
        teams = self.teams.iloc[index]
        teams = teams.itertuples() if isinstance(teams, pandas.DataFrame) else [teams]
        teams_str = ', '.join(['Home Team {} VS Away Team {}'.format(row.home_team, row.away_team)
                               for row in teams])
        teams_str = teams_str + "."
        print(teams_str)
        super().render(mode)


class ThreeWaySoccerPercentageOddsEnv(ThreeWaySoccerOddsEnv):
    versionadded = '0.2.0'


class ThreeWaySoccerDailyOddsEnv(ThreeWaySoccerOddsEnv):
    versionadded = '0.5.0'


class ThreeWaySoccerDailyPercentageOddsEnv(ThreeWaySoccerOddsEnv):
    versionadded = '0.5.0'
