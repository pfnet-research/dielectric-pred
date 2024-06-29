FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as base

ARG PYTHON_VERSION="3.9"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python-is-python3 \
    python3-wheel \
    software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libpython${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 10

RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /repo

COPY pyproject.toml requirements* ./
COPY release release
COPY pekoe-util pekoe-util
COPY extensions extensions

ARG REQUIREMENTS_LOCK="requirements.lock"

ENV PATH=${HOME}/.local/bin:${PATH}
RUN pip install --user -r requirements.txt -r requirements-dev.txt -c ${REQUIREMENTS_LOCK} --no-cache-dir --extra-index-url https://pypi.pfn.io/simple
