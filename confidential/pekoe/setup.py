import pathlib
from dataclasses import dataclass
from typing import Sequence, Mapping

from setuptools import find_packages, setup

# Get __version__ variable from pekoe/_version.py
setup_py_path = pathlib.Path(__file__).parent.absolute()
exec(open(setup_py_path / "pekoe" / "_version.py").read())

# adapt requirements.txt format to install_requires format
path_prefix = f"file://{setup_py_path}"


@dataclass
class PathReplacement:
    pkg_name: str
    path: str


def read_requires(fname: str, replacements: Mapping[str, PathReplacement]) -> Sequence[str]:
    used = set()
    with open(fname) as f:
        requires = f.read().splitlines()
        for i, req in enumerate(requires):
            if req in replacements:
                repl = replacements[req]
                if (setup_py_path / repl.path).exists():
                    # path exists locally (most likely `pip install .`)
                    requires[i] = f"{repl.pkg_name} @ {path_prefix}/{repl.path}"
                else:
                    # path does not exist (`pip install` from pypi)
                    requires[i] = f"{repl.pkg_name}=={__version__}"  # NOQA
                used.add(req)
    not_used = set(replacements.keys()) - used
    assert len(not_used) == 0, f"replacements {not_used} have not been applied"
    return requires


setup_requires: Sequence[str] = []
install_requires = read_requires(
    "requirements.txt", {"-e ./release": PathReplacement("pfp-base", "release"),},
)
dev_requires = read_requires("requirements-dev.txt", {})
strict_requires = read_requires(
    "requirements-strict.txt",
    {
        "-e ./extensions/teanet_conv_helper": PathReplacement(
            "teanet-conv-helper", "extensions/teanet_conv_helper"
        ),
    },
)

extra_require = {
    "dev": dev_requires,
    "strict": strict_requires,
}

package_data = {"pekoe.nn.models": ["configs/*"], "pekoe": ["py.typed"]}

setup(
    name="pekoe",
    version=__version__,  # type: ignore[name-defined] # NOQA
    description="Preferred Neural Network Potential",
    author="Preferred Networks, Inc",
    author_email="chem@preferred.jp",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extra_require,
    package_data=package_data,
)
