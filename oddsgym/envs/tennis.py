import pandas

from .meta import MetaEnvBuilder


class TennisOddsEnv(metaclass=MetaEnvBuilder):
    sport = "tennis"
    versionadded = "0.8.0"
    odds_column_names = ["win", "lose"]

    def __init__(self, tennis_bets_dataframe, *args, **kwargs):
        """Initializes a new environment.

        Parameters
        ----------
        tennis_bets_dataframe: dataframe of shape (n_matches, n_odds + 3), n_odds == 2
            A list of matches, with their betting odds, results and the players playing.

            .. warning::

                Please make sure that the odds columns are named "win, lose",
                that the results column is named "result" and that the player names
                columns are named "winner, loser" respectively.
        """
        odds = tennis_bets_dataframe[self.odds_columns]
        results = (
            tennis_bets_dataframe["result"]
            if tennis_bets_dataframe["result"].notna().all()
            else None
        )
        if not self.daily_env:
            odds = odds.values
            results = results.values
        self.players = tennis_bets_dataframe[["winner", "loser"]]
        self.HEADERS.insert(2, "Players")
        super().__init__(odds, self.odds_column_names, results, *args, **kwargs)

    def create_info(self, action):
        info = super().create_info(action)
        index = self._get_current_index()
        players = self.players.iloc[index]
        players = (
            players.itertuples() if isinstance(players, pandas.DataFrame) else [players]
        )
        players_str = "\n".join(
            [
                "[blue]{}[/] VS [magenta]{}[/]".format(row.winner, row.loser)
                for row in players
            ]
        )
        info.update(players=players_str)
        return info


class TennisPercentageOddsEnv(TennisOddsEnv):
    pass


class TennisDailyOddsEnv(TennisOddsEnv):
    pass


class TennisDailyPercentageOddsEnv(TennisOddsEnv):
    pass
