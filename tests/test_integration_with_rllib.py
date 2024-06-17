import pytest
try:
    import ray
    from ray.rllib.algorithms.appo import APPOConfig
    # from ray.rllib.agents.es import ESTrainer
    # from ray.rllib.agents.ddpg import TD3Trainer, ApexDDPGTrainer
    # from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
    from ray.rllib.algorithms.impala import ImpalaConfig
    from ray.rllib.algorithms.marwil import MARWILConfig
    # from ray.rllib.agents.pg import PGTrainer
    from ray.rllib.algorithms.ppo import PPOConfig
except ImportError:
    pytest.skip("skipping rllib integration tests because package is not installed",
                allow_module_level=True)


# supported RL algorithms by environment
# ENV_DICT = {'FootballDataDailyEnv': [A2CTrainer, A3CTrainer, ESTrainer, TD3Trainer, ApexDDPGTrainer,
#                                      DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer, PPOTrainer, APPOTrainer],
#             'FootballDataDailyPercentageEnv': [A2CTrainer, A3CTrainer, ESTrainer, TD3Trainer, ApexDDPGTrainer,
#                                                DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer, PPOTrainer,
#                                                APPOTrainer],
#             'TennisDataDailyEnv': [A2CTrainer, A3CTrainer, ESTrainer, TD3Trainer, ApexDDPGTrainer,
#                                    DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer, PPOTrainer, APPOTrainer],
#             'TennisDataDailyPercentageEnv': [A2CTrainer, A3CTrainer, ESTrainer, TD3Trainer, ApexDDPGTrainer,
#                                              DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer, PPOTrainer,
#                                              APPOTrainer]
#             }

ENV_DICT = {'FootballDataDailyEnv': [APPOConfig, ImpalaConfig, MARWILConfig, PPOConfig],
            'FootballDataDailyPercentageEnv': [APPOConfig, ImpalaConfig, MARWILConfig, PPOConfig],
            'TennisDataDailyEnv': [APPOConfig, ImpalaConfig, MARWILConfig, PPOConfig],
            'TennisDataDailyPercentageEnv': [APPOConfig, ImpalaConfig, MARWILConfig, PPOConfig],
            }

ray.init(logging_level='ERROR', num_cpus=2, num_gpus=0)


def _get_model_config(alg, framework):
    model_config = alg().resources(num_gpus=0, num_learner_workers=1, num_cpus_for_local_worker=1, num_cpus_per_worker=1).debugging(log_level="ERROR").framework(framework=framework)
    return model_config


@pytest.mark.parametrize("alg", ENV_DICT['FootballDataDailyEnv'])
@pytest.mark.parametrize("framework", ['torch'])
def test_football_co_uk_daily_env_with_(alg, framework):
    model_config = _get_model_config(alg, framework)
    model_config = model_config.environment(env="FootballDataDaily-ray-v0")
    model = model_config.build()
    model.train()


@pytest.mark.parametrize("alg", ENV_DICT['FootballDataDailyPercentageEnv'])
@pytest.mark.parametrize("framework", ['torch'])
def test_football_co_uk_daily_percentage_env_with_(alg, framework):
    model_config = _get_model_config(alg, framework)
    model_config = model_config.environment(env="FootballDataDailyPercent-ray-v0")
    model = model_config.build()
    model.train()


@pytest.mark.parametrize("alg", ENV_DICT['TennisDataDailyEnv'])
@pytest.mark.parametrize("framework", ['torch'])
def test_tennis_co_uk_daily_env_with_(alg, framework):
    model_config = _get_model_config(alg, framework)
    model_config = model_config.environment(env="TennisDataDaily-ray-v0")
    model = model_config.build()
    model.train()


@pytest.mark.parametrize("alg", ENV_DICT['TennisDataDailyPercentageEnv'])
@pytest.mark.parametrize("framework", ['torch'])
def test_tennis_co_uk_daily_percentage_env_with_(alg, framework):
    model_config = _get_model_config(alg, framework)
    model_config = model_config.environment(env="TennisDataDailyPercent-ray-v0")
    model = model_config.build()
    model.train()
