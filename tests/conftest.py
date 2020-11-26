from datetime import datetime, timedelta
from pytest import fixture
from numpy import array
from pandas import DataFrame
from oddsgym.envs.base import BaseOddsEnv
from oddsgym.envs.base_percentage import BasePercentageOddsEnv
from oddsgym.envs.soccer import ThreeWaySoccerOddsEnv, ThreeWaySoccerPercentageOddsEnv
from oddsgym.envs.soccer import ThreeWaySoccerDailyOddsEnv, ThreeWaySoccerDailyPercentageOddsEnv
from oddsgym.envs.daily_bets import DailyOddsEnv, DailyPercentageOddsEnv
from oddsgym.envs.tennis import TennisOddsEnv, TennisPercentageOddsEnv
from oddsgym.envs.tennis import TennisDailyOddsEnv, TennisDailyPercentageOddsEnv


@fixture()
def basic_env(request):
    return BaseOddsEnv(array([[1, 2], [3, 4]]), ['w', 'l'], [1, 0])


@fixture()
def basic_percentage_env(request):
    return BasePercentageOddsEnv(array([[2, 1]]), ['w', 'l'], [0])


@fixture()
def three_way_env(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 1},
                                       {'home_team': 'MCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 0}])
    return ThreeWaySoccerOddsEnv(soccer_bets_dataframe)


@fixture()
def three_way_percentage_env(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 1},
                                       {'home_team': 'MCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 0}])
    return ThreeWaySoccerPercentageOddsEnv(soccer_bets_dataframe)


@fixture()
def daily_bets_env(request):
    dataframe = DataFrame([{'w': 1, 'l': 2, 'date': datetime.today().date() - timedelta(days=1), 'result': 1},
                           {'w': 4, 'l': 3, 'date': datetime.today().date() - timedelta(days=1), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           ])
    return DailyOddsEnv(dataframe.drop('result', 'columns'), ['w', 'l'], dataframe['result'])


@fixture()
def daily_bets_env_non_uniform(request):
    dataframe = DataFrame([{'w': 1, 'l': 2, 'date': datetime.today().date() - timedelta(days=2), 'result': 1},
                           {'w': 4, 'l': 3, 'date': datetime.today().date() - timedelta(days=1), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           {'w': 5, 'l': 4, 'date': datetime.today().date(), 'result': 0},
                           ])
    return DailyOddsEnv(dataframe.drop('result', 'columns'), ['w', 'l'], dataframe['result'])


@fixture()
def daily_bets_percentage_env(request):
    dataframe = DataFrame([{'w': 1, 'l': 2, 'date': datetime.today().date() - timedelta(days=1), 'result': 1},
                           {'w': 4, 'l': 3, 'date': datetime.today().date() - timedelta(days=1), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           ])
    return DailyPercentageOddsEnv(dataframe.drop('result', 'columns'), ['w', 'l'], dataframe['result'])


@fixture()
def daily_bets_percentage_env_non_uniform(request):
    dataframe = DataFrame([{'w': 1, 'l': 2, 'date': datetime.today().date() - timedelta(days=2), 'result': 1},
                           {'w': 4, 'l': 3, 'date': datetime.today().date() - timedelta(days=1), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           {'w': 4, 'l': 3, 'date': datetime.today().date(), 'result': 0},
                           {'w': 5, 'l': 4, 'date': datetime.today().date(), 'result': 0},
                           ])
    return DailyPercentageOddsEnv(dataframe.drop('result', 'columns'), ['w', 'l'], dataframe['result'])


@fixture()
def three_way_daily_env(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 1,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'MCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 0,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'PSG', 'away_team': 'MCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 2,
                                        'date': datetime.today().date() - timedelta(days=1)},
                                       {'home_team': 'FCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 1,
                                        'date': datetime.today().date() - timedelta(days=1)},
                                       {'home_team': 'PSG', 'away_team': 'FCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 0,
                                        'date': datetime.today().date()},
                                       {'home_team': 'MTA', 'away_team': 'MCB',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 2,
                                        'date': datetime.today().date()}])
    return ThreeWaySoccerDailyOddsEnv(soccer_bets_dataframe)


@fixture()
def three_way_daily_env_non_uniform(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 1,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'MCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 0,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'PSG', 'away_team': 'MCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 2,
                                        'date': datetime.today().date() - timedelta(days=1)},
                                       {'home_team': 'FCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 1,
                                        'date': datetime.today().date()},
                                       {'home_team': 'PSG', 'away_team': 'FCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 0,
                                        'date': datetime.today().date()},
                                       {'home_team': 'MTA', 'away_team': 'MCB',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 2,
                                        'date': datetime.today().date()}])
    return ThreeWaySoccerDailyOddsEnv(soccer_bets_dataframe)


@fixture()
def three_way_daily_percentage_env(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 1,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'MCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 0,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'PSG', 'away_team': 'MCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 2,
                                        'date': datetime.today().date() - timedelta(days=1)},
                                       {'home_team': 'FCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 1,
                                        'date': datetime.today().date() - timedelta(days=1)},
                                       {'home_team': 'PSG', 'away_team': 'FCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 0,
                                        'date': datetime.today().date()},
                                       {'home_team': 'MTA', 'away_team': 'MCB',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 2,
                                        'date': datetime.today().date()}])
    return ThreeWaySoccerDailyPercentageOddsEnv(soccer_bets_dataframe)


@fixture()
def three_way_daily_percentage_env_non_uniform(request):
    soccer_bets_dataframe = DataFrame([{'home_team': 'FCB', 'away_team': 'PSG',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 1,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'MCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 0,
                                        'date': datetime.today().date() - timedelta(days=2)},
                                       {'home_team': 'PSG', 'away_team': 'MCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 2,
                                        'date': datetime.today().date() - timedelta(days=1)},
                                       {'home_team': 'FCB', 'away_team': 'MTA',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 1,
                                        'date': datetime.today().date()},
                                       {'home_team': 'PSG', 'away_team': 'FCB',
                                        'home': 1, 'draw': 2, 'away': 3, 'result': 0,
                                        'date': datetime.today().date()},
                                       {'home_team': 'MTA', 'away_team': 'MCB',
                                        'home': 4, 'draw': 3, 'away': 2, 'result': 2,
                                        'date': datetime.today().date()}])
    return ThreeWaySoccerDailyPercentageOddsEnv(soccer_bets_dataframe)


@fixture
def tennis_env(request):
    tennis_bets_dataframe = DataFrame([{'winner': 'Berrettini M.', 'loser': 'Harris A.',
                                        'win': 1.11, 'lose': 6.68, 'result': 0},
                                       {'winner': 'Berankis R.', 'loser': 'Carballes Baena R.',
                                        'win': 1.73, 'lose': 2.1, 'result': 0}])
    return TennisOddsEnv(tennis_bets_dataframe)


@fixture
def tennis_percentage_env(request):
    tennis_bets_dataframe = DataFrame([{'winner': 'Berrettini M.', 'loser': 'Harris A.',
                                        'win': 1.11, 'lose': 6.68, 'result': 0},
                                       {'winner': 'Berankis R.', 'loser': 'Carballes Baena R.',
                                        'win': 1.73, 'lose': 2.1, 'result': 0}])
    return TennisPercentageOddsEnv(tennis_bets_dataframe)


@fixture
def tennis_daily_env(request):
    tennis_bets_dataframe = DataFrame([{'winner': 'Berrettini M.', 'loser': 'Harris A.',
                                        'win': 1.11, 'lose': 6.68, 'result': 0, 'date': datetime(2020, 1, 20).date()},
                                       {'winner': 'Berankis R.', 'loser': 'Carballes Baena R.',
                                        'win': 1.73, 'lose': 2.1, 'result': 0, 'date': datetime(2020, 1, 20).date()},
                                       {'winner': 'Cilic M.', 'loser': 'Moutet C.',
                                        'win': 1.47, 'lose': 2.7, 'result': 0, 'date': datetime(2020, 1, 21).date()},
                                       {'winner': 'Davidovich Fokina A.', 'loser': 'Gombos N.',
                                        'win': 1.78, 'lose': 2.04, 'result': 0, 'date': datetime(2020, 1, 21).date()},
                                       {'winner': 'Hurkacz H.', 'loser': 'Novak D.',
                                        'win': 1.39, 'lose': 2.98, 'result': 0, 'date': datetime(2020, 1, 21).date()},
                                       {'winner': 'Querrey S.', 'loser': 'Berankis R.',
                                        'win': 1.34, 'lose': 3.27, 'result': 0, 'date': datetime(2020, 1, 22).date()}])
    return TennisDailyOddsEnv(tennis_bets_dataframe)


@fixture
def tennis_daily_percentage_env(request):
    tennis_bets_dataframe = DataFrame([{'winner': 'Berrettini M.', 'loser': 'Harris A.',
                                        'win': 1.11, 'lose': 6.68, 'result': 0, 'date': datetime(2020, 1, 20).date()},
                                       {'winner': 'Berankis R.', 'loser': 'Carballes Baena R.',
                                        'win': 1.73, 'lose': 2.1, 'result': 0, 'date': datetime(2020, 1, 20).date()},
                                       {'winner': 'Cilic M.', 'loser': 'Moutet C.',
                                        'win': 1.47, 'lose': 2.7, 'result': 0, 'date': datetime(2020, 1, 21).date()},
                                       {'winner': 'Davidovich Fokina A.', 'loser': 'Gombos N.',
                                        'win': 1.78, 'lose': 2.04, 'result': 0, 'date': datetime(2020, 1, 21).date()},
                                       {'winner': 'Hurkacz H.', 'loser': 'Novak D.',
                                        'win': 1.39, 'lose': 2.98, 'result': 0, 'date': datetime(2020, 1, 21).date()},
                                       {'winner': 'Querrey S.', 'loser': 'Berankis R.',
                                        'win': 1.34, 'lose': 3.27, 'result': 0, 'date': datetime(2020, 1, 22).date()}])
    return TennisDailyPercentageOddsEnv(tennis_bets_dataframe)
