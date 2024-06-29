import pathlib

import invoke

from tasks.utils import run

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]


def _run_prose_lock(c, requirements_options: str, lockfile: str) -> None:
    run(
        c,
        f"prose lock {requirements_options} -l {lockfile}"
        " --extra-index-url https://pypi.pfn.io/simple",
        pty=True,
    )
    run(c, f"python3 scripts/modify_lockfile.py --lock {lockfile} --print-ignored", pty=True)


@invoke.task
def update(c):
    """
    Update lockfiles
    """
    requirements_options = (
        " -r requirements.txt" " -r requirements-full.txt" " -r requirements-custom_func.txt"
    )
    strict_requirements_options = requirements_options + " -r requirements-strict.txt"
    dev_requirements_options = " -r requirements-dev.txt"
    _run_prose_lock(c, requirements_options, "requirements.lock")
    _run_prose_lock(c, strict_requirements_options, "requirements-strict.lock")
    _run_prose_lock(c, requirements_options + dev_requirements_options, "requirements-dev.lock")
    _run_prose_lock(
        c, strict_requirements_options + dev_requirements_options, "requirements-dev-strict.lock"
    )
