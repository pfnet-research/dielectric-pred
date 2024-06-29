# pekoe-util

This package provides an interface to use pekoe's calculator in pfmd.

## Installation
Inside the root directory of the pekoe repository:
```bash
# Install pekoe
python setup.py install
cd pekoe-util
# Install pekoe-util
python setup.py install
```

## Usage
When using pfmd:
```bash
export PFMD_CALCULATOR_MODULE="pekoe_util.pekoe_calculator"
```