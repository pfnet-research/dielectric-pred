# Pytorch used by binary model

## Description
This directory has some files to build pytorch library used by binary models.

In order to run binary model with pytorch library, we neeed to build pytorch from soruce with some options. Unfortunately, building pytorch takes too long time, so we recommend you to get pytorch binary instead of building it.

## How to get pytorch binary

`asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/pekoe-libtorch-binary` image contains it.
You can use make command to get and install it like this:

```
$ make
```

Default install directory is `$(pwd)../../install`. If you'd like to specify the install directory, you can do that like this:

```
$ make INSTALL_PREFIX=/opt/pfn/pfp/deploy
```

## How to build `asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/pekoe-libtorch-binary`

You can build a image after putting libtorch-${TORCH_VERSION}-cu101.zip here:

```
$ docker build . -t asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/pekoe-libtorch-binary
```

## How to build pytorch

## on k8s

You can use `build_libtorch.bash` under `scripts` directory to build pytorch on k8s. It's faster and you don't need to prepare environment on your machine. Please see the script for more detail.

## on your local machine

You can use `build.bash` here, but we recommend to use k8s one.
