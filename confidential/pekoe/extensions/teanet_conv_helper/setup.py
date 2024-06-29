import os

from setuptools import setup, Extension, find_packages


# Get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, "teanet_conv_helper", "_version.py")).read())

install_requires=["torch"]
cmdclass = {}
ext_modules = []

setup(name='teanet-conv-helper',
    version=__version__,  # NOQA
    description="teanet-conv-helper: Optional module for pekoe",
    author="Preferred Networks, Inc",
    author_email="chem@preferred.jp",
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)
