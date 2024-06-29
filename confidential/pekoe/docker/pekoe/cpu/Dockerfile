FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && pip install --upgrade pip

WORKDIR /repo

COPY pyproject.toml requirements* ./
COPY release release
COPY pekoe-util pekoe-util
COPY extensions extensions

ENV PATH=${HOME}/.local/bin:${PATH}
RUN pip install --user -r requirements.txt -c requirements.lock --no-cache-dir --extra-index-url https://pypi.pfn.io/simple
