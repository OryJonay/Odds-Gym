from .base import BaseOddsEnv
from .base_percentage import BasePercentageGamblingEnv


class ThreeWaySoccerOddsEnv(BaseOddsEnv):
    """Environment for 3-way soccer betting with a fixed bet size."""

    def __init__(self, soccer_bets_dataframe):
        """Initializes a new environment.

        .. versionadded: 0.1.0

        Parameters
        ----------
        soccer_bets_dataframe: dataframe of shape (n_games, n_odds + 3)
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


class ThreeWaySoccerPercentageOddsEnv(ThreeWaySoccerOddsEnv, BasePercentageGamblingEnv):
    """Environment for 3-way soccer betting with a non fixed bet size.

    .. versionadded: 0.2.0
    """
    pass
