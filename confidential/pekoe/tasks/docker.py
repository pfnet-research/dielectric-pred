import os
import pathlib
import sys
from datetime import datetime

import invoke
from invoke.exceptions import UnexpectedExit

import tasks.utils as utils

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT_DIR / "docker" / "deploy" / "Dockerfile"
DEFAULT_DOCKER_IMAGE = "harbor.mnj.pfn.io/chem/pekoe-native-module-builder-ubuntu20.04"


@invoke.task(
    help={
        "prefix": 'Image name prefix. \
        Image names will be "{prefix}-{target}-{os}" (default "pekoe")',
        "target": 'Build target (default "")',
        "os": 'OS. (default "ubuntu20.04")',
        "tag": 'Image tag. (default "%Y%m%d-%H%M%S")',
        "push": 'Push images after being built (default "false")',
        "no-latest": 'Do not tag as latest (default "false")',
        "build-args": "Build args for example key1=v1,key=v2",
    }
)
def build(
    c,
    prefix="pekoe",
    target="",
    os="ubuntu20.04",
    tag="",
    push=False,
    no_latest=False,
    build_args="",
):
    """
    Build docker images
    """

    now = datetime.now()
    dt = now.strftime("%Y%m%d-%H%M%S")

    opt_build_args = ""
    if build_args != "":
        opt_build_args = " ".join(map(lambda kv: f"--build-arg {kv}", build_args.split(",")))

    if prefix == "":
        raise ValueError("prefix cannot be empty")

    if tag == "":
        tag = dt

    targets = [
        "deploy-minimum",
        "builder-base",
        "libtorch-builder",
        "libtorch-binary",
        "native-module-builder",
        "native-module",
        "deploy",
    ]

    if target == "":
        target = targets[-1]

    def target_exists(target):
        return targets.count(target) > 0

    assert target_exists(target)

    def opt_cache_from_latest(prefix, target, os):
        try:
            utils.run(c, f"docker pull {prefix}-{target}-{os}:latest", pty=True)
            return f" --cache-from {prefix}-{target}-{os}:latest"
        except UnexpectedExit as e:
            if e.result.exited == -2:
                sys.exit(1)
            return ""

    def python_version(os):
        if os == "ubuntu20.04":
            return "3.8"
        else:
            return "3.7"

    def build_target(prefix, target, os, tag, opt_cache_from=""):
        opt_target = f"--target {target}"
        opt_tag = f"-t {prefix}-{target}-{os}:{tag}"
        if not no_latest:
            opt_tag += f" -t {prefix}-{target}-{os}:latest"
        utils.run(c, "docker images -a")
        utils.run(
            c,
            f"DOCKER_BUILDKIT=1 docker build \
            --ssh default \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --build-arg DEPLOY_PYTHON_VERSION={python_version(os)} \
            {opt_build_args} {opt_cache_from} {opt_target} {opt_tag} {ROOT_DIR} -f {DOCKERFILE}",
        )
        if push:
            utils.run(c, f"docker push {prefix}-{target}-{os}:{tag}", pty=True)
            if not no_latest:
                utils.run(c, f"docker push {prefix}-{target}-{os}:latest", pty=True)

    opt_cache_from = opt_cache_from_latest(prefix, target, os)
    if targets.index(target) >= targets.index("libtorch-binary"):
        opt_cache_from += opt_cache_from_latest(prefix, "libtorch-binary", os)

    build_target(prefix, target, os, tag, opt_cache_from)

    for t in targets:
        if t == target:
            break
        build_target(prefix, t, os, tag)


@invoke.task(help={"regex": "Remove images which match the regular expression"})
def rm(c, regex):
    """
    Remove docker images which match regular expression
    """

    docker_images = 'docker images -q --format "{{.Repository}}:{{.Tag}} {{.ID}}"'
    grep = f'egrep "[^ ]*{regex}"'
    result = utils.run(c, f"{docker_images} | {grep}")
    images = result.stdout.split()
    if len(images):
        ids = " ".join(set(images[1::2]))
        utils.run(c, f"docker rmi {ids}")


@invoke.task(
    help={
        "command": "Run command (required)",
        "image": f'Dokcer image. (default: "{DEFAULT_DOCKER_IMAGE}")',
        "dir": f'Directory to be mounted on container. (default: "{ROOT_DIR}")',
    }
)
def run(
    c,
    command,
    image=DEFAULT_DOCKER_IMAGE,
    dir=None,
):
    """
    Run a command in docker container
    """
    dir = dir or str(ROOT_DIR)

    def is_rootless():
        options = utils.run(c, "docker info -f '{{.SecurityOptions}}'").stdout
        return "rootless" in options

    if is_rootless():
        user_options = ""
    else:
        user_options = f"-u {os.geteuid()}:{os.geteuid()} \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro"

    path = os.path.abspath(dir)

    return utils.run(
        c,
        f"docker run --rm -it {user_options} -v {path}:{path} -w={path} {image} \
        env CCACHE_DIR=/tmp/.ccahe {command}",
        pty=True,
    )
