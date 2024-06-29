import pathlib

import invoke

from tasks.utils import run

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]


@invoke.task
def fix(c):
    """
    Fix lint issues.
    """
    options = f'--config {str(BASE_DIR / "pysen.toml")}'
    run(c, f"pysen {options} run format", pty=True)


@invoke.task
def check(c):
    """
    Check lint issues.
    """
    options = f'--config {str(BASE_DIR / "pysen.toml")}'
    run(c, f"pysen {options} run lint", pty=True)
