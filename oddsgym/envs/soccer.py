from .base import BaseOddsEnv

class SoccerOddsEnv(BaseOddsEnv):
    """Environment for soccer betting"""

    def __init__(self, soccer_bets_dataframe):
        odds_column_names = ['home', 'away', 'draw']
        odds = soccer_bets_dataframe[odds_column_names].values
        results = soccer_bets_dataframe['result'].values if soccer_bets_dataframe.notna().all() else None
        super().__init__(self, odds, odds_column_names, results)
        self.teams = soccer_bets_dataframe[['home_team', 'away_team']].values

    def render(self, mode='human'):
        return 'Home Team {} VS Away Team {}. {}'.format(self.teams[self.current_step][0],
                                                         self.teams[self.current_step][1],
                                                         super().render(self, mode))
