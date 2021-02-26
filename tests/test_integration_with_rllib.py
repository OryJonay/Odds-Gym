tests/test_integration_with_rllib.pyimport pytest
try:
    import ray
    from ray.rllib.agents.a3c import A2CTrainer, A3CTrainer
    from ray.rllib.agents.ars import ARSTrainer
    from ray.rllib.agents.es import ESTrainer
    from ray.rllib.agents.ddpg import TD3Trainer, ApexDDPGTrainer
    from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
    from ray.rllib.agents.impala import ImpalaTrainer
    from ray.rllib.agents.marwil import MARWILTrainer
    from ray.rllib.agents.pg import PGTrainer
    from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
except ImportError:
    pytest.skip("skipping rllib integration tests because package is not installed",
                allow_module_level=True)


# supported RL algorithms by environment
ENV_DICT = {'FootballDataDailyEnv': [A2CTrainer, A3CTrainer, ARSTrainer, ESTrainer, TD3Trainer, ApexDDPGTrainer,
                                     DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer, PPOTrainer, APPOTrainer],
            'FootballDataDailyPercentageEnv': [A2CTrainer, A3CTrainer, ARSTrainer, ESTrainer, TD3Trainer,
                                               ApexDDPGTrainer, DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer,
                                               PPOTrainer, APPOTrainer],
            'TennisDataDailyEnv': [A2CTrainer, A3CTrainer, ARSTrainer, ESTrainer, TD3Trainer, ApexDDPGTrainer,
                                   DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer, PPOTrainer, APPOTrainer],
            'TennisDataDailyPercentageEnv': [A2CTrainer, A3CTrainer, ARSTrainer, ESTrainer, TD3Trainer,
                                             ApexDDPGTrainer, DDPGTrainer, ImpalaTrainer, MARWILTrainer, PGTrainer,
                                             PPOTrainer, APPOTrainer]
            }

ray.init(logging_level='ERROR')


@pytest.mark.parametrize("alg", ENV_DICT['FootballDataDailyEnv'])
@pytest.mark.parametrize("framework", ['tf', 'torch'])
def test_football_co_uk_daily_env_with_(alg, framework):
    model_config = alg._default_config.copy()
    model_config.update(num_gpus=0, num_workers=1, log_level='ERROR', train_batch_size=10)
    model_config['model'].update(fcnet_hiddens=[3, 3])
    if ray.__version__ <= '0.8.5':
        if alg not in (ApexDDPGTrainer, ImpalaTrainer):
            model_config.update(use_pytorch=framework == 'torch')
    else:
        model_config.update(framework=framework)
    if alg == ApexDDPGTrainer:
        model_config.update(num_workers=2)
    if alg in [PPOTrainer, APPOTrainer]:
        model_config.update(num_sgd_iter=10)
        if alg == PPOTrainer:
            model_config.update(sgd_minibatch_size=10)
    model = alg(env="FootballDataDaily-ray-v0", config=model_config)
    model.train()


@pytest.mark.parametrize("alg", ENV_DICT['FootballDataDailyPercentageEnv'])
@pytest.mark.parametrize("framework", ['tf', 'torch'])
def test_football_co_uk_daily_percentage_env_with_(alg, framework):
    model_config = alg._default_config.copy()
    model_config.update(num_gpus=0, num_workers=1, log_level='ERROR', train_batch_size=10)
    model_config['model'].update(fcnet_hiddens=[3, 3])
    if ray.__version__ <= '0.8.5':
        if alg not in (ApexDDPGTrainer, ImpalaTrainer):
            model_config.update(use_pytorch=framework == 'torch')
    else:
        model_config.update(framework=framework)
    if alg == ApexDDPGTrainer:
        model_config.update(num_workers=2)
    if alg in [PPOTrainer, APPOTrainer]:
        model_config.update(num_sgd_iter=10)
        if alg == PPOTrainer:
            model_config.update(sgd_minibatch_size=10)
    model = alg(env="FootballDataDailyPercent-ray-v0", config=model_config)
    model.train()


@pytest.mark.parametrize("alg", ENV_DICT['TennisDataDailyEnv'])
@pytest.mark.parametrize("framework", ['tf', 'torch'])
def test_tennis_co_uk_daily_env_with_(alg, framework):
    model_config = alg._default_config.copy()
    model_config.update(num_gpus=0, num_workers=1, log_level='ERROR', train_batch_size=10)
    model_config['model'].update(fcnet_hiddens=[3, 3])
    if ray.__version__ <= '0.8.5':
        if alg not in (ApexDDPGTrainer, ImpalaTrainer):
            model_config.update(use_pytorch=framework == 'torch')
    else:
        model_config.update(framework=framework)
    if alg == ApexDDPGTrainer:
        model_config.update(num_workers=2)
    if alg in [PPOTrainer, APPOTrainer]:
        model_config.update(num_sgd_iter=10)
        if alg == PPOTrainer:
            model_config.update(sgd_minibatch_size=10)
    model = alg(env="TennisDataDaily-ray-v0", config=model_config)
    model.train()


@pytest.mark.parametrize("alg", ENV_DICT['TennisDataDailyPercentageEnv'])
@pytest.mark.parametrize("framework", ['tf', 'torch'])
def test_tennis_co_uk_daily_percentage_env_with_(alg, framework):
    model_config = alg._default_config.copy()
    model_config.update(num_gpus=0, num_workers=1, log_level='ERROR', train_batch_size=10)
    model_config['model'].update(fcnet_hiddens=[3, 3])
    if ray.__version__ <= '0.8.5':
        if alg not in (ApexDDPGTrainer, ImpalaTrainer):
            model_config.update(use_pytorch=framework == 'torch')
    else:
        model_config.update(framework=framework)
    if alg == ApexDDPGTrainer:
        model_config.update(num_workers=2)
    if alg in [PPOTrainer, APPOTrainer]:
        model_config.update(num_sgd_iter=10)
        if alg == PPOTrainer:
            model_config.update(sgd_minibatch_size=10)
    model = alg(env="TennisDataDailyPercent-ray-v0", config=model_config)
    model.train()
