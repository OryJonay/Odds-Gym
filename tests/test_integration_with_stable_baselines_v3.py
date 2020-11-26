import pytest
try:
    from stable_baselines3 import A2C, PPO, SAC, TD3
except ImportError:
    pytest.skip("skipping stable-baselines3 integration tests because package is not installed",
                allow_module_level=True)


# supported RL algorithms by environment
ENV_DICT = {'BaseOddsEnv': [A2C, PPO],
            'BasePercentageOddsEnv': [A2C, PPO, SAC, TD3],
            'DailyOddsEnv': [A2C, PPO, SAC, TD3],
            'DailyPercentageOddsEnv': [A2C, PPO, SAC],
            'FootballDataDailyEnv': [A2C, PPO, SAC, TD3],
            'FootballDataDailyPercentageEnv': [A2C, PPO, SAC],
            'TennisDataDailyEnv': [A2C, PPO, SAC, TD3],
            'TennisDataDailyPercentageEnv': [A2C, PPO, SAC]}


@pytest.mark.parametrize("alg", ENV_DICT['BaseOddsEnv'])
def test_basic_env_with_(basic_env, alg):
    model = alg('MlpPolicy', basic_env)
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['BasePercentageOddsEnv'])
def test_basic_percentage_env_with_(basic_percentage_env, alg):
    model = alg('MlpPolicy', basic_percentage_env)
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['DailyOddsEnv'])
def test_daily_env_with_(daily_bets_env, alg):
    model = alg('MlpPolicy', daily_bets_env)
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['DailyPercentageOddsEnv'])
def test_daily_percentage_env_with_(daily_bets_percentage_env, alg):
    model = alg('MlpPolicy', daily_bets_percentage_env)
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['FootballDataDailyEnv'])
def test_football_co_uk_daily_env_with_(alg):
    model = alg('MlpPolicy', 'FootballDataDaily-v0')
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['FootballDataDailyPercentageEnv'])
def test_football_co_uk_daily_percentage_env_with_(alg):
    model = alg('MlpPolicy', 'FootballDataDailyPercent-v0')
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['TennisDataDailyEnv'])
def test_tennis_co_uk_daily_env_with_(alg):
    model = alg('MlpPolicy', 'TennisDataDaily-v0')
    model.learn(total_timesteps=10)


@pytest.mark.parametrize("alg", ENV_DICT['TennisDataDailyPercentageEnv'])
def test_tennis_co_uk_daily_percentage_env_with_(alg):
    model = alg('MlpPolicy', 'FootballDataDailyPercent-v0')
    model.learn(total_timesteps=10)
