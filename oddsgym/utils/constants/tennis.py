import os
from itertools import product


CSV_URL = 'http://www.tennis-data.co.uk/{year}{women}/{tournament}.csv'

TOURNAMENTS = ['ausopen', 'frenchopen', 'usopen']

CSV_CACHE_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'csv_cache', 'tennis'))

SITES = ["B365", "B&W", "CB", "EX", "IW", "LB", "PS", "SB", "SJ", "GB", "UB"]

YEARS = range(2010, 2020)


def url_kwargs():
    for tournament, women in product(TOURNAMENTS, ['', 'w']):
        yield {'tournament': tournament, 'women': women}
