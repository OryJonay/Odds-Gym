import os
import shutil
import pytest
import glob
import typer
from typer.testing import CliRunner
from oddsgym.utils.csv_downloader import create_csv_cache
from oddsgym.utils.constants import SPORT_CSV_CACHE_PATHS


app = typer.Typer()
app.command()(create_csv_cache)
runner = CliRunner()


@pytest.mark.parametrize("arguments,expected", (('default', 220),
                                                (['-s', 'football'], 220),
                                                (['-s', 'tennis'], 60),
                                                (['-f', '-s', 'football'], 22),
                                                (['-f', '-s', 'tennis'], 6),
                                                ))
def test_csv_download(arguments, expected):
    sport = arguments[-1] if arguments != 'default' else 'football'
    csv_cache_path = SPORT_CSV_CACHE_PATHS[sport]
    if os.path.exists(csv_cache_path):
        shutil.rmtree(csv_cache_path)
    if arguments == 'default':
        result = runner.invoke(app)
    else:
        result = runner.invoke(app, arguments)
    try:
        assert result.exit_code == 0
        assert len(glob.glob(os.path.join(f'{csv_cache_path}', '**', '*.csv'))) == expected
    except AssertionError:
        raise
    finally:
        shutil.rmtree(csv_cache_path)
