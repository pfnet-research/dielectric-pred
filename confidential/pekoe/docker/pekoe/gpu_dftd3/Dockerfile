FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    gfortran \
    git \
    libpython3.8 \
    pkg-config \
    python3.8-dev \
    python3.8-distutils \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install DFTD3
RUN cd /tmp && \
    wget https://www.chemiebn.uni-bonn.de/pctc/mulliken-center/software/dft-d3/dftd3.tgz && \
    tar zxvf dftd3.tgz && \
    make && \
    mv dftd3 /usr/local/bin && \
    rm dftd3.tgz

RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3.8 \
    && pip install --upgrade pip \
    && pip install \
    cupy-cuda111==9.4.0 \
    pytorch-pfn-extras==0.6.5

WORKDIR /repo

COPY pyproject.toml requirements* ./
COPY release release
COPY pekoe-util pekoe-util
COPY extensions extensions

ENV PATH=${HOME}/.local/bin:${PATH}
RUN pip install --user -r requirements.txt -c requirements.lock --no-cache-dir --extra-index-url https://pypi.pfn.io/simple
