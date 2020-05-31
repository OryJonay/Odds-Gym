import pytest
try:
    from stable_baselines import A2C, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TD3, TRPO
except ImportError:
    pytest.skip("skipping stable-baselines integration tests because package is not installed", allow_module_level=True)


# supported RL algorithems by environment
ENV_DICT = {'BaseOddsEnv': [A2C, ACKTR, DQN, PPO1, PPO2, TRPO],
            'BasePercentageOddsEnv': [A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TD3, TRPO],
            'DailyOddsEnv': [A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TD3, TRPO],
            'DailyPercentageOddsEnv': [A2C, ACKTR, DDPG, PPO1, PPO2, SAC, TD3, TRPO]}


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


@pytest.mark.parametrize("alg", ENV_DICT['DailyOddsEnv'])
def test_daily_percentage_env_with_(daily_bets_percentage_env, alg):
    model = alg('MlpPolicy', daily_bets_percentage_env)
    model.learn(total_timesteps=10)
