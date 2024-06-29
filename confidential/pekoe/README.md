# pekoe (repository)

This repository has three python packages: `pfp`, `pekoe-util`, and`pekoe`. `pekoe` is the name of this repository, also the name of python package.

## pekoe (library)

`pekoe` is one of the library in this repository. Typically this library is all you need.

You can install pekoe from PFN internal pypi (released version) or from source (dev version). To install it from pypi, the command may be:

```
pip install pekoe --extra-index-url https://pypi.pfn.io/simple
```

You can also install it from source by:

```
pip install .
```

If you can use clean environment (e.g. virtualenv, etc.), you can prepare well-tested (CI-tested) environment using lockfile and extra-require `[strict]` option like:

```
pip install -r requirements.txt -c requirements-strict.lock
pip install .[strict]
```

`[strict]` option can be used for pypi as well.

```
pip install pekoe[strict] --extra-index-url https://pypi.pfn.io/simple
```

We now use `prose-pfn` instead of `poetry` for maintaining package dependency and lockfiles. You don't have to prepare `prose-pfn` unless you are not a library developer.

### pekoe developer

dev-related libraries can be prepared by:

```
pip install -r requirements.txt -r requirements-dev.txt -r requirements-full.txt -c requirements.lock --extra-index-url https://pypi.pfn.io/simple
pip install -e . --no-deps --no-use-pep517
```

You can ignore `requirements-full.txt` if you don't use extensions.


The lockfiles can be created using invoke command (`prose-pfn` comes here).

```
invoke lock.update
```


## pfp

`pfp` is python package of neural network potential for universal molecular dynamics simulation. It runs the simulation with binary model. **It is released for customers.**

### Prepare environment to build a binary model

We support Ubuntu 20.04 as official environment to run `pfp`, but you can run `pfp` on Ubuntu 18.04. To build and run `pfp`, you need to install some packages like this:

```
apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgfortran3 \
    libopenblas-base \
    libpython3.7 \
    libquadmath0 \
    python3.7 \
    python3.7-dev \
    ccache \
    cmake \
    git \
    libblas3 \
    libblas-dev \
    libeigen3-dev \
    libnuma1 \
    libomp5 \
    libopenblas-dev \
    ninja-build \
    ssh-client \
    unzip \
    zip
```

### Build a binary model

In order to use `pfp` package, you need to build binary model from a ONNX file. You can use `invoke` tasks for that.

```
invoke deploy.build prod
```

### Run `pfp`'s example

Then you can run an example.

```
python3 release/examples/ase_opt.py
```

## pekoe-util

`pekoe-util` provides an interface to use pekoe's calculator in pfmd.

## Development in Docker container

### Docker images

We have docker images for development on `harbor.mnj.pfn.io`

* `harbor.mnj.pfn.io/chem/pekoe-deploy-minimum`
* `harbor.mnj.pfn.io/chem/pekoe-builder-base`
* `harbor.mnj.pfn.io/chem/pekoe-libtorch-builder`
* `harbor.mnj.pfn.io/chem/pekoe-libtorch-binary`
* `harbor.mnj.pfn.io/chem/pekoe-native-module-builder`
* `harbor.mnj.pfn.io/chem/pekoe-native-module`
* `harbor.mnj.pfn.io/chem/pekoe-deploy`

We explain some of the images here.

`harbor.mnj.pfn.io/chem/pekoe-deploy-minimum` is minimum image to run `pfp`, we use this image to test artifacts for release.

`harbor.mnj.pfn.io/chem/pekoe-native-module-builder` is to build binary model, if it is difficult to prepare environment explained above, you can use this image.

`harbor.mnj.pfn.io/chem/pekoe-deploy` has latest binary model on `dev` branch, you can run some calculation on k8s with this image.

For other images, please see `docker/deploy/Dockerfile`.

### Build docker images

We have `invoke` tasks to do that.

```
invoke docker.build
```

## Release artifacts

`FlexCI` creates release artifacts on `dev` branch and store them on GCS (gs://chem-pfn-private-ci/deepmi). You can download them with `gsutil`, or if you just would like to download the latest one, you can use `invoke` task like this:

```
invoke deploy.download
```

This task downloads the artifacts and check their md5sum.

## Release

Use [bump2version](https://github.com/c4urself/bump2version) to bump the version for both `pekoe` and `pfp`

e.g.

```bash
# To increase major version:
$ bumpversion major

# To increase minor version:
$ bumpversion minor

# To increase patch version:
$ bumpversion patch
```

## Others

`pyenv-installer` (https://github.com/pyenv/pyenv-installer) can help to install `pyenv`. After installing `pyenv`, you can install any version of python like this:

```bash
$ pyenv install 3.7.5
```

Please see the official document (https://github.com/pyenv/pyenv) for the detail.
