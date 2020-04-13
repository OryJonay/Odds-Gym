from .base import BaseOddsEnv
from .base_percentage import BasePercentageOddsEnv
from .daily_bets import DailyOddsEnv, DailyPercentageOddsEnv


class ThreeWaySoccerOddsEnv(BaseOddsEnv):
    """Environment for 3-way soccer betting with a fixed bet size.

    .. versionadded:: 0.1.0
    """

    def __init__(self, soccer_bets_dataframe):
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
        odds_column_names = ['home', 'draw', 'away']
        odds = soccer_bets_dataframe[odds_column_names].values
        results = soccer_bets_dataframe['result'].values if soccer_bets_dataframe['result'].notna().all() else None
        super().__init__(odds, odds_column_names, results)
        self.teams = soccer_bets_dataframe[['home_team', 'away_team']].values

    def render(self, mode='human'):  # pragma: no cover
        """Outputs the current team names, balance and step.

        Returns
        -------
        msg : str
            A string with the current team names, balance and step.
        """
        return 'Home Team {} VS Away Team {}. {}'.format(self.teams[self.current_step % self._odds.shape[0]][0],
                                                         self.teams[self.current_step % self._odds.shape[0]][1],
                                                         super().render(mode))


class ThreeWaySoccerPercentageOddsEnv(ThreeWaySoccerOddsEnv, BasePercentageOddsEnv):
    """Environment for 3-way soccer betting with a non fixed bet size.

    .. versionadded:: 0.2.0
    """
    pass


class ThreeWaySoccerDailyOddsEnv(DailyOddsEnv):
    """Environment for 3-way soccer betting, grouped by day, with a fixed bet size.

    .. versionadded:: 0.5.0
    """

    def __init__(self, soccer_bets_dataframe):
        """Initializes a new environment.



        Parameters
        ----------
        soccer_bets_dataframe: dataframe of shape (n_games, n_odds + 4), n_odds == 3
            A list of games, with their betting odds, results, the teams playing and the dates.

            .. warning::

                Please make sure that the odds columns are named "home, draw, away",
                that the results column is named "result", that the team names
                columns are named "home_team, away_team" respectively and that the
                date column is named "date".
        """
        odds_column_names = ['home', 'draw', 'away']
        odds = soccer_bets_dataframe[odds_column_names + ['date']]
        results = soccer_bets_dataframe['result'] if soccer_bets_dataframe['result'].notna().all() else None
        super().__init__(odds, odds_column_names, results)
        self.teams = soccer_bets_dataframe[['home_team', 'away_team']]

    def render(self, mode='human'):  # pragma: no cover
        """Outputs the current team names, balance and step.

        Returns
        -------
        msg : str
            A string with the current team names, balance and step.
        """
        index = self._get_current_index()
        teams_str = ', '.join(['Home Team {} VS Away Team {}'.format(row.home_team, row.away_team)
                               for row in self.teams.iloc[index].itertuples()])
        balance_str = super().render(mode)
        return '. '.join((teams_str, balance_str))


class ThreeWaySoccerDailyPercentageOddsEnv(ThreeWaySoccerDailyOddsEnv, DailyPercentageOddsEnv):
    """Environment for 3-way soccer betting, grouped by date, with a non fixed bet size.

    .. versionadded:: 0.5.0
    """
    pass
