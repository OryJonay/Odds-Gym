*****************
API Documentation
*****************

Generic Environments
********************

Environment that are agnostic to the sport \ game of chance.

BaseOddsEnv
###########
.. autoclass:: oddsgym.envs.base.BaseOddsEnv
    :members:

BasePercentageOddsEnv
#########################
.. autoclass:: oddsgym.envs.base_percentage.BasePercentageOddsEnv

DailyOddsEnv
############
.. autoclass:: oddsgym.envs.daily_bets.DailyOddsEnv

DailyPercentageOddsEnv
######################
.. autoclass:: oddsgym.envs.daily_bets.DailyPercentageOddsEnv

Sports Specific Environments
****************************

Environment that are sport specific.

ThreeWaySoccerOddsEnv
#####################
.. autoclass:: oddsgym.envs.soccer.ThreeWaySoccerOddsEnv
    :members: __init__

ThreeWaySoccerPercentageOddsEnv
###############################
.. autoclass:: oddsgym.envs.soccer.ThreeWaySoccerPercentageOddsEnv

ThreeWaySoccerDailyOddsEnv
##########################
.. autoclass:: oddsgym.envs.soccer.ThreeWaySoccerDailyOddsEnv

ThreeWaySoccerDailyPercentageOddsEnv
####################################
.. autoclass:: oddsgym.envs.soccer.ThreeWaySoccerDailyPercentageOddsEnv

TennisOddsEnv
#####################
.. autoclass:: oddsgym.envs.tennis.TennisOddsEnv
    :members: __init__

TennisPercentageOddsEnv
###############################
.. autoclass:: oddsgym.envs.tennis.TennisPercentageOddsEnv

TennisDailyOddsEnv
##########################
.. autoclass:: oddsgym.envs.tennis.TennisDailyOddsEnv

TennisDailyPercentageOddsEnv
####################################
.. autoclass:: oddsgym.envs.tennis.TennisDailyPercentageOddsEnv

Site and Sports Specific Environments
*************************************

Environment that are site and sport specific (receives the odds data from a specific site).

FootballDataDailyEnv
####################################
.. autoclass:: oddsgym.envs.footballdata.FootballDataDailyEnv
    :members: __init__

FootballDataDailyPercentageEnv
####################################
.. autoclass:: oddsgym.envs.footballdata.FootballDataDailyPercentageEnv

TennisDataDailyEnv
####################################
.. autoclass:: oddsgym.envs.tennisdata.TennisDataDailyEnv
    :members: __init__

TennisDataDailyPercentageEnv
####################################
.. autoclass:: oddsgym.envs.tennisdata.TennisDataDailyPercentageEnv