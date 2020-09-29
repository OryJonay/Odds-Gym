from .base import BaseOddsEnv
from .base_percentage import BasePercentageOddsEnv
from .daily_bets import DailyOddsEnv, DailyPercentageOddsEnv


class MetaEnvBuilder(type):
    def __new__(cls, name, bases, attr):
        new_bases = list(bases)
        percentage_env, daily_env = 'Percentage' in name, 'Daily' in name
        docstring = (f'Environment for {attr["sport"]} betting{", grouped by date," if daily_env else ""}'
                     f' with a {"non fixed" if percentage_env else "fixed"} bet size.\n\n    '
                     f'.. versionadded:: {attr["versionadded"]}\n    ')
        if percentage_env and daily_env:
            new_bases += [DailyPercentageOddsEnv]
        elif daily_env:
            new_bases += [DailyOddsEnv]
        elif percentage_env:
            new_bases += [BasePercentageOddsEnv]
        else:
            new_bases += [BaseOddsEnv]
        attr['percentage_env'] = percentage_env
        attr['daily_env'] = daily_env
        attr['__doc__'] = docstring
        return super(MetaEnvBuilder, cls).__new__(cls, name, tuple(new_bases), attr)
