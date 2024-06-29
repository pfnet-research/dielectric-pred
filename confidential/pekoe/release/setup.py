import os

from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "ase>=3.19.3",
    "numpy>=1.18.1",
    "PyYAML>=5.1.2",
]

# Get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, "pfp", "_version.py")).read())

# Non-python files
package_data = {"pfp": ["nn/models/crystal/lib/*", "py.typed"]}

setup(
    name="pfp-base",
    version=__version__,  # NOQA
    description="Preferred Neural Network Potential (Binary version)",
    author="Preferred Networks, Inc",
    author_email="chem@preferred.jp",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires,
    package_data=package_data,
)
