import os
import asyncio
from contextlib import closing
import aiohttp
import typer
from oddsgym.utils.constants.football import CSV_URL, COUNTRIES, LEAGUES, CSV_CACHE_PATH


async def download_file(session, start, end, country, league):
    url = CSV_URL.format(start=start, end=end, country=country, league=league)
    async with session.get(url) as response:
        assert response.status == 200, url
        return country, league, await response.read()


async def download_multiple(fast):
    leagues = [(COUNTRIES[country], league) for country in LEAGUES for league in LEAGUES[country].values()]
    seasons = range(19, 20)
    if not fast:
        seasons = range(10, 20)
    with typer.progressbar(seasons) as seasons_progress:
        for season in seasons_progress:
            start, end = str(season).zfill(2), str(season + 1).zfill(2)
            season_csv_path = os.path.join(CSV_CACHE_PATH, f'{start}-{end}')
            if not os.path.exists(season_csv_path):
                os.mkdir(season_csv_path)
            async with aiohttp.ClientSession() as session:
                download_futures = [download_file(session, start, end, country, league) for country, league in leagues]
                for download_future in asyncio.as_completed(download_futures):
                    country, league, result = await download_future
                    with open(os.path.join(season_csv_path, f'{country}{league}.csv'), 'w') as csv_file:
                        csv_file.write(result.decode('latin-1'))
    return leagues


def create_csv_cache(fast: bool = typer.Option(False, "--fast", "-f", help="Run only on last seasons")):
    if not os.path.exists(CSV_CACHE_PATH):
        os.mkdir(CSV_CACHE_PATH)
    asyncio.set_event_loop(asyncio.new_event_loop())
    with closing(asyncio.get_event_loop()) as loop:
        loop.run_until_complete(download_multiple(fast))


def main():  # pragma: no cover
    typer.run(create_csv_cache)
