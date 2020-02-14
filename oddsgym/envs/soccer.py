from .base import BaseOddsEnv
from .base_percentage import BasePercentageGamblingEnv

class ThreeWaySoccerOddsEnv(BaseOddsEnv):
    """Environment for 3-way soccer betting"""

    def __init__(self, soccer_bets_dataframe):
        odds_column_names = ['home', 'draw', 'away']
        odds = soccer_bets_dataframe[odds_column_names].values
        results = soccer_bets_dataframe['result'].values if soccer_bets_dataframe['result'].notna().all() else None
        super().__init__(odds, odds_column_names, results)
        self.teams = soccer_bets_dataframe[['home_team', 'away_team']].values

    def render(self, mode='human'):  # pragma: no cover
        return 'Home Team {} VS Away Team {}. {}'.format(self.teams[self.current_step % self._odds.shape[0]][0],
                                                         self.teams[self.current_step % self._odds.shape[0]][1],
                                                         super().render(mode))


class ThreeWaySoccerPercentageOddsEnv(ThreeWaySoccerOddsEnv, BasePercentageGamblingEnv):
    """Environment for 3-way soccer betting with non fixed bet size"""
    pass
