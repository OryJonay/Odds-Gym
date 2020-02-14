from pytest import fixture
from numpy import array
from pandas import DataFrame
from oddsgym.envs.base import BaseOddsEnv
from oddsgym.envs.base_percentage import BasePercentageGamblingEnv
from oddsgym.envs.soccer import ThreeWaySoccerOddsEnv, ThreeWaySoccerPercentageOddsEnv

@fixture()
def basic_env(request):
    return BaseOddsEnv(array([[1, 2], [3, 4]]), ['w', 'l'], [1, 0])

@fixture()
def basic_percentge_env(request):
    return BasePercentageGamblingEnv(array([[2, 1]]), ['w', 'l'], [0])

@fixture()
def three_way_env(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG', 'home': 1, 'draw': 2, 'away': 3, 'result': 1},
                                       {'home_team': 'MCB', 'away_team': 'MTA', 'home': 4, 'draw': 3, 'away': 2, 'result': 0}])
    return ThreeWaySoccerOddsEnv(soccer_bets_dataframe)

@fixture()
def three_way_percentage_env(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG', 'home': 1, 'draw': 2, 'away': 3, 'result': 1},
                                       {'home_team': 'MCB', 'away_team': 'MTA', 'home': 4, 'draw': 3, 'away': 2, 'result': 0}])
    return ThreeWaySoccerPercentageOddsEnv(soccer_bets_dataframe)