from enum import Enum
from . import football
from . import tennis


SPORT_CSV_URLS = {"football": football.CSV_URL,
                  "tennis": tennis.CSV_URL}

SPORT_CSV_CACHE_PATHS = {"football": football.CSV_CACHE_PATH,
                         "tennis": tennis.CSV_CACHE_PATH}

SPORT_YEARS = {"football": football.YEARS,
               "tennis": tennis.YEARS}

SPORT_KWARGS = {"football": football.url_kwargs,
                "tennis": tennis.url_kwargs}


class SupportedSport(str, Enum):
    football = "football"
    tennis = "tennis"

    @property
    def csv_url(self):
        return SPORT_CSV_URLS[self.value]

    @property
    def csv_cache_url(self):
        return SPORT_CSV_CACHE_PATHS[self.value]

    @property
    def years(self):
        return SPORT_YEARS[self.value]

    @property
    def url_kwargs(self):
        return SPORT_KWARGS[self.value]()
