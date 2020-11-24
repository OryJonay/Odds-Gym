import os


CSV_URL = 'http://www.football-data.co.uk/mmz4281/{start}{end}/{country}{league}.csv'

COUNTRIES = {'England': 'E', 'Scotland': 'SC', 'Germany': 'D', 'Italy': 'I', 'Spain': 'SP', 'France': 'F',
             'Netherlands': 'N', 'Belgium': 'B', 'Portugal': 'P', 'Turkey': 'T', 'Greece': 'G'}

LEAGUES = {'England': {'Premier League': 0, 'Championship': 1, 'League 1': 2, 'League 2': 3, 'Conference': 'C'},
           'Scotland': {'Premier League': 0, 'Division 1': 1, 'Division 2': 2, 'Division 3': 3},
           'Germany': {'Bundesliga 1': 1, 'Bundesliga 2': 2},
           'Italy': {'Serie A': 1, 'Serie B': 2},
           'Spain': {'Primera Division': 1, 'Segunda Division': 2},
           'France': {'Ligue 1': 1, 'Ligue 2': 2},
           'Netherlands': {'Eredivise': 1},
           'Belgium': {'Jupiler League': 1},
           'Portugal': {'Liga 1': 1},
           'Turkey': {'Super Lig': 1},
           'Greece': {'Super League': 1}}

CSV_CACHE_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', 'csv_cache', 'football'))

SITES = ["B365", "BS", "BW", "GB", "IW", "LB", "PS", "SO", "SB", "SJ", "SY", "VC", "WH", "P"]

YEARS = range(10, 20)


def url_kwargs():
    for country in LEAGUES:
        for league in LEAGUES[country].values():
            yield {'country': COUNTRIES[country], 'league': league}
