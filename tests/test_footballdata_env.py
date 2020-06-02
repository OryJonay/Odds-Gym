import pytest
from gym import envs
from oddsgym.envs.footballdata import FootballDataDailyPercentageEnv


@pytest.mark.parametrize('kwargs', [{'country': 'England', 'league': 'Premier League', 'start': 1999, 'end': 2011},
                                    {'country': 'France', 'league': 'Premier League', 'start': 2010, 'end': 2011},
                                    {'country': 'England', 'league': 'Premier League', 'start': 2010, 'end': 3000},
                                    {'country': 'England', 'league': 'Premier League', 'start': 2010, 'end': 2009},
                                    {'country': 'Brazil', 'league': 'Premier League', 'start': 2010, 'end': 2011}])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        FootballDataDailyPercentageEnv(**kwargs)


def test_creation():
    assert FootballDataDailyPercentageEnv() is not None


def test_registeration():
    assert 'FootballDataDailyPercent-v0' in [spec.id for spec in envs.registry.all()]
