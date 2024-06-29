from pathlib import Path

from setuptools import find_packages, setup

PEKOE_PATH = Path(__file__).resolve().parent.parent

package_data = {"pekoe_util": ["py.typed"]}

setup(
    name="pekoe-util",
    version="0.0.1",
    description="pekoe calculator interface for pfmd",
    packages=find_packages(),
    setup_requires=[],
    install_requires=[
        "ase>=3.19.3",
        f"pekoe @ file://localhost/{PEKOE_PATH}/",
    ],
    package_data=package_data,
)
