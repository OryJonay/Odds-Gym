from .base import BaseOddsEnv
from .base_percentage import BasePercentageOddsEnv
from .daily_bets import DailyOddsEnv, DailyPercentageOddsEnv


class MetaEnvBuilder(type):
    def __new__(cls, name, bases, attr):
        def safe_get(attribute):
            if attribute in attr:
                return attr[attribute]
            for base in bases:
                if getattr(base, attribute, None):
                    return getattr(base, attribute)

        new_bases = list(bases)
        percentage_env, daily_env = 'Percentage' in name, 'Daily' in name
        docstring = (f'Environment for {safe_get("sport")} betting{", grouped by date," if daily_env else ""}'
                     f' with a {"non fixed" if percentage_env else "fixed"} bet size.\n\n    '
                     f'.. versionadded:: {safe_get("versionadded")}\n    ')
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
        attr['odds_columns'] = safe_get('odds_column_names') + ['date'] * daily_env
        attr['odds_column_names'] = safe_get('odds_column_names')
        attr['__doc__'] = docstring
        return super(MetaEnvBuilder, cls).__new__(cls, name, tuple(new_bases), attr)
