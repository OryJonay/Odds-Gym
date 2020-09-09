import os
import shutil
import pytest
import typer
from typer.testing import CliRunner
from oddsgym.utils.csv_downloader import create_csv_cache
from oddsgym.utils.constants.football import CSV_CACHE_PATH


app = typer.Typer()
app.command()(create_csv_cache)
runner = CliRunner()


@pytest.mark.parametrize("arguments", ('all', '-f', '--fast'))
def test_csv_download(arguments):
    if os.path.exists(CSV_CACHE_PATH):
        shutil.rmtree(CSV_CACHE_PATH)
    if arguments == 'all':
        result = runner.invoke(app)
    else:
        result = runner.invoke(app, [arguments])
    shutil.rmtree(CSV_CACHE_PATH)
    assert result.exit_code == 0
