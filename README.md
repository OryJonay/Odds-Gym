# Sports odds betting environment 
[![PyPI version](https://badge.fury.io/py/oddsgym.svg)](https://badge.fury.io/py/oddsgym)
![build](https://github.com/OryJonay/Odds-Gym/workflows/build/badge.svg) 
[![Coverage Status](https://coveralls.io/repos/github/OryJonay/Odds-Gym/badge.svg?branch=master)](https://coveralls.io/github/OryJonay/Odds-Gym?branch=master)
[![Hatch project](https://img.shields.io/badge/tool-%F0%9F%A5%9A%20Hatch-4051b5.svg)](https://github.com/pypa/hatch) 
[![Endpoint Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fruff%2Fmain%2Fassets%2Fbadge%2Fv2.json&label=linting)](https://github.com/astral-sh/ruff) 
[![License - Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-9400d3.svg)](https://spdx.org/licenses/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) 

A sports betting environment for [Farma Foundation Gymnasium](https://gymnasium.farama.org/) based environments.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Extending the Environments](#extending-the-environments)
5. [Registering Environments with rllib](#registering-environments-with-rllib)
4. [Documentation](#documentation)

## Installation

    pip install oddsgym

## Environment

The starting bank is X (X > 0), representing X available bets. Actions are all available bets for a game (depends on sport), placing 1 bet for each option. Also, the agent cannot bet over his current bank (i.e., can't place 3 bets when the current bank is 2).

For example, in 3-way betting for soccer, the available actions are:

    1. Bet on home team
    2. Bet on away team
    3. Bet on draw
    4. Bet on home team and away team
    5. Bet on home team and draw
    6. Bet on away team and draw
    7. Bet on home team and away team and draw
    8. Don't place a bet for this game

### Step
A step is placing a bet on a single game. In each step, the agent knows the betting odds for this game. The reward for each step is the amount of money won (negative reward when losing money).

### Episode
An episode is betting for a whole year or when "striking out" (losing all the money).

## Usage

### Basic Usage
Here is an example of how to use the `ThreeWaySoccerOddsEnv` environment:

```python
import gymnasium as gym
from oddsgym.envs import ThreeWaySoccerOddsEnv

# Create the environment
env = ThreeWaySoccerOddsEnv(soccer_bets_dataframe)

# Reset the environment
obs, info = env.reset()

# Take a random action
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

```

### Using Pre-Registered Environments
You can also use the pre-registered environments directly with `gym.make`:

```python
import gymnasium as gym

# Create the environment
env = gym.make("FootballDataDaily-v0")

# Reset the environment
obs, info = env.reset()

# Take a random action
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

```

## Extending the Environments

### Creating Custom Environments
To create a custom environment, you can extend the base classes provided by `oddsgym`. Here is an example of how to create a custom environment for a new sport:

```python
import pandas as pd
from oddsgym.envs.base import BaseOddsEnv

class CustomSportOddsEnv(BaseOddsEnv):
    sport = "custom_sport"
    odds_column_names = ["outcome1", "outcome2", "outcome3"]

    def __init__(self, custom_sport_bets_dataframe, *args, **kwargs):
        odds = custom_sport_bets_dataframe[self.odds_column_names]
        results = custom_sport_bets_dataframe["result"]
        super().__init__(odds, self.odds_column_names, results, *args, **kwargs)

    def create_info(self, action):
        info = super().create_info(action)
        # Add custom info here
        return info
```

### Registering Custom Environments
You can register your custom environment with `gym` to use it like any other pre-registered environment:

```python
import gymnasium as gym
from oddsgym.envs import CustomSportOddsEnv

gym.register(
    id="CustomSport-v0",
    entry_point="path.to.your.module:CustomSportOddsEnv",
    max_episode_steps=365,
)
```

## Registering Environments with rllib

To use the environments with rllib, you need to register them with rllib. Here is an example of how to register the `FootballDataDailyEnv` environment:

```python
import gymnasium as gym
from ray import tune
from oddsgym.envs import FootballDataDailyEnv

# Register the environments with rllib
tune.register_env(
    "FootballDataDaily-ray-v0",
    lambda env_config: gym.wrappers.FlattenObservation(FootballDataDailyEnv(env_config))
)
```

Once registered, you can use these environments with rllib models.

## Documentation

The full documentation is hosted at: https://oryjonay.github.io/Odds-Gym
