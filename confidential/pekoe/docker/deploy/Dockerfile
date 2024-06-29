# syntax = docker/dockerfile:experimental
ARG LIBTORCH_BINARY_IMAGE=asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/pekoe-libtorch-binary
ARG CUDA_IMAGE=asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/nvidia/cuda:10.1-cudnn7-devel-ubuntu20.04
ARG DEPLOY_PYTHON_VERSION=3.8
FROM ${CUDA_IMAGE} as deploy-minimum
ARG DEPLOY_PYTHON_VERSION

# NOTE: Avoiding GPG error
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gfortran \
    libopenblas-base \
    libpython${DEPLOY_PYTHON_VERSION} \
    libquadmath0 \
    python${DEPLOY_PYTHON_VERSION} \
    python${DEPLOY_PYTHON_VERSION}-dev \
    python${DEPLOY_PYTHON_VERSION}-distutils \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY release/requirements.txt .
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python${DEPLOY_PYTHON_VERSION} && \
    python${DEPLOY_PYTHON_VERSION} -m pip install --no-cache-dir -r requirements.txt

FROM deploy-minimum as builder-base
ARG DEPLOY_PYTHON_VERSION

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
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
    zip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /usr/lib/ccache:$PATH
RUN ccache -M 32G

ENV TORCH_VERSION=1.5.0

COPY requirements-dev.txt requirements-dev.txt
COPY requirements-strict.lock requirements-strict.lock
RUN python${DEPLOY_PYTHON_VERSION} -m pip install --no-cache-dir -r requirements-dev.txt -c requirements-strict.lock --extra-index-url https://pypi.pfn.io/simple

FROM builder-base as libtorch-builder
WORKDIR /tmp/pfp
COPY ./deploy/external/pytorch pytorch

# Make copy source configurable https://github.com/docker/for-mac/issues/2155
ARG LIBTORCH_BINARY_IMAGE
FROM ${LIBTORCH_BINARY_IMAGE} as libtorch-binary

FROM builder-base as native-module-builder
WORKDIR /tmp/pfp

COPY ./deploy/external/libtorch-dev libtorch-dev
RUN make -C libtorch-dev INSTALL_PREFIX=/opt/pfn/pfp/deploy/install install clean

ARG TORCH_VERSION=${TORCH_VERSION}
COPY --from=libtorch-binary /libtorch-${TORCH_VERSION}-cu101.zip pytorch/libtorch-${TORCH_VERSION}-cu101.zip
COPY ./deploy/external/pytorch pytorch
RUN make -C pytorch INSTALL_PREFIX=/opt/pfn/pfp/deploy/install install clean

COPY ./deploy/external/pfvm pfvm
COPY ./deploy/external/obfuscator obfuscator
ENV SSH_AUTH_SOCK_MOUNTED /tmp/agent.sock
RUN --mount=type=ssh \
    mkdir -p -m 600 /root/.ssh && \
    export SSH_AUTH_SOCK=${SSH_AUTH_SOCK-${SSH_AUTH_SOCK_MOUNTED}} && \
    ssh-keyscan -H github.pfidev.jp >> /root/.ssh/known_hosts && \
    make -C pfvm INSTALL_PREFIX=/opt/pfn/pfp/deploy/install install clean && \
    make -C obfuscator INSTALL_PREFIX=/opt/pfn/pfp/deploy/install install clean

FROM native-module-builder as native-module
ARG DEPLOY_PYTHON_VERSION
ARG BUILD_DEV_DIR=build.dev
ARG BUILD_PROD_DIR=build
WORKDIR /opt/pfn/pfp

COPY ./deploy/teanet deploy/teanet

COPY ./pekoe pekoe
COPY ./examples examples
COPY ./release release
COPY requirements.txt .
COPY requirements-strict.lock .
RUN python${DEPLOY_PYTHON_VERSION} -m pip install --no-cache-dir -r requirements.txt -c requirements-strict.lock --extra-index-url https://pypi.pfn.io/simple
RUN python${DEPLOY_PYTHON_VERSION} -m pip install ./release --no-deps --no-use-pep517
# NODE(takamoto): It is only used for exporting onnx.
RUN python${DEPLOY_PYTHON_VERSION} -m pip install --no-cache-dir pytorch-pfn-extras==0.6.5 torch==1.13.0
RUN PYTHONPATH=. \
    python${DEPLOY_PYTHON_VERSION} ./examples/pekoe/ase_molecule.py \
        --output-onnx deploy/teanet/target_onnx --output-onnx-only

RUN --mount=type=cache,target=/root/.ccache \
    cd deploy/teanet && \
    export DEPLOY_PYTHON_VERSION=${DEPLOY_PYTHON_VERSION} && \
    make BUILD_DIR=${BUILD_DEV_DIR} build-dev && \
    make BUILD_DIR=${BUILD_PROD_DIR} build-prod

FROM deploy-minimum as deploy
ARG BUILD_PROD_DIR=build

WORKDIR /repo
COPY release release
COPY --from=native-module \
     /opt/pfn/pfp/deploy/teanet/lib/model_settings.yaml \
     /opt/pfn/pfp/deploy/teanet/lib/shift_energies.json \
     /opt/pfn/pfp/deploy/teanet/lib/gaussian_shift_energies.json \
     /opt/pfn/pfp/deploy/teanet/lib/vasp_U_shift_energies_v1_4_0.json \
     /opt/pfn/pfp/deploy/teanet/lib/vasp_wo_U_shift_energies_v1_4_0.json \
     release/pfp/nn/models/crystal/lib/
COPY --from=native-module /opt/pfn/pfp/deploy/teanet/${BUILD_PROD_DIR}/libinfer_prod.so release/pfp/nn/models/crystal/lib/libinfer.so

RUN pip3 install -e release
