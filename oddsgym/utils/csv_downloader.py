import os
import asyncio
from contextlib import closing
import aiohttp
import typer
from oddsgym.utils.constants import SupportedSport


async def download_file(session, url_template, **kwargs):
    url = url_template.format(**kwargs)
    while True:
        try:
            async with session.get(url) as response:
                assert response.status == 200, url
                return (kwargs.get("country", kwargs.get("tournament")),
                        kwargs.get("league", kwargs.get("women")),
                        await response.read())
        except AssertionError:  # pragma: no cover
            asyncio.sleep(1)


async def download_multiple(fast, sport):
    url_kwargs = list(sport.url_kwargs)
    years = sport.years
    if fast:
        years = years[-1:]
    with typer.progressbar(years) as years_progress:
        for year in years_progress:
            year_csv_path = os.path.join(sport.csv_cache_url,
                                         sport.csv_url.split('/')[-2].format(start=year, end=year + 1,
                                                                             year=year, women=''))
            if not os.path.exists(year_csv_path):
                os.mkdir(year_csv_path)
            async with aiohttp.ClientSession() as session:
                download_futures = [download_file(session, sport.csv_url, start=year,
                                                  end=year + 1, year=year, **kwargs)
                                    for kwargs in url_kwargs]
                for download_future in asyncio.as_completed(download_futures):
                    prefix, suffix, result = await download_future
                    with open(os.path.join(year_csv_path, f'{prefix}{suffix}.csv'), 'w') as csv_file:
                        csv_file.write(result.decode('latin-1'))


def create_csv_cache(fast: bool = typer.Option(False, "--fast", "-f", help="Run only on last seasons / years"),
                     sport: SupportedSport = typer.Option(SupportedSport.football, "--sport", "-s",
                                                          help="Which sport to run", case_sensitive=False)):
    if not os.path.exists(sport.csv_cache_url):
        os.makedirs(sport.csv_cache_url)
    asyncio.set_event_loop(asyncio.new_event_loop())
    with closing(asyncio.get_event_loop()) as loop:
        loop.run_until_complete(download_multiple(fast, sport))


def main():  # pragma: no cover
    typer.run(create_csv_cache)
