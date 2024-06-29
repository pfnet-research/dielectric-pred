import os as os_mod
import pathlib
import sys

import invoke

import tasks.docker
from tasks.utils import run

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]


@invoke.task(
    help={
        "target": "Build target. prod or dev (required)",
        "docker": 'Build binary in docker container. (default "false")',
    }
)
def build(c, target, docker=False):
    """
    Build binary model
    """
    model_dir = ROOT_DIR / "deploy" / "teanet"

    def command(opt=""):
        return f'bash -c "cd {model_dir} && make {opt} {target}"'

    if docker:
        return tasks.docker.run(
            c,
            f'env DEPLOY_INSTALL_PREFIX=/opt/pfn/pfp/deploy \
            {command(opt="BUILD_DIR=build.docker")}',
        )
    else:
        return run(c, command())


@invoke.task
def md5(c, file):
    """
    Calculate md5 in base64 format
    """
    return run(c, f'gsutil hash -m {file} | grep "(md5)" | cut -f 4')


@invoke.task(
    help={
        "branch": 'branch (default "refs/heads/dev")',
        "commit": 'commit id (default "None" means download latest artifacts)',
        "os": 'os (default "ubuntu20.04")',
        "out-dir": 'output directory. (default ".")',
    }
)
def download(c, branch="refs/heads/dev", commit=None, os="ubuntu20.04", out_dir="."):
    """
    Download artifacts from GCS
    """
    branch = f"gs://chem-pfn-private-ci/deepmi/{branch}"
    daytime_regex = "[0-9]" * 8 + "-" + "[0-9]" * 6

    if commit is None:
        commit_regex = "[0-9a-f]" * 40
    else:
        commit_regex = commit

    url = f"{branch}/{daytime_regex}/{commit_regex}"
    dirs = run(c, f"gsutil ls -d {url}").stdout.split()
    if dirs == []:
        print(f"No artifacts found on ${url}")
        sys.exit(1)

    commit_dir = sorted(dirs, reverse=True)[0]
    os_dir = f"{commit_dir}{os}"
    files = run(c, f"gsutil ls {os_dir}").stdout.split()
    if files == []:
        print(f"No artifacts found on {os_dir}")
        sys.exit(1)

    out_dir = pathlib.Path(out_dir)

    def out_subdir(f):
        return pathlib.Path(f).parent.relative_to(os_dir)

    for f in files:
        out = out_dir / out_subdir(f)
        os_mod.makedirs(out, mode=0o755, exist_ok=True)
        run(c, f"gsutil cp {f} {out}")

    md5_urls = filter(lambda f: pathlib.Path(f).suffix == ".md5", files)
    for md5_url in md5_urls:
        out = out_dir / out_subdir(md5_url)
        downloaded_md5 = out / pathlib.Path(md5_url).name
        with open(f"{downloaded_md5}", "r") as f:
            expected_md5 = f.read().strip()
        downloaded_file = out / pathlib.Path(downloaded_md5).stem
        actual_md5 = md5(c, downloaded_file).stdout.strip()
        if actual_md5 != expected_md5:
            print(
                f"Error: {f} does not match the md5 ({actual_md5}), \
                but expected is ({expected_md5})."
            )
            sys.exit(1)
