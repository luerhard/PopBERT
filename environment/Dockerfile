# hash:sha256:34003d50728038c6e3ff410b22c29c0e16bb4bb7aed1e39202b0df7d0223143a
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        gfortran=4:11.2.0-1ubuntu1 \
        libbz2-dev=1.0.8-5build1 \
        libcurl4-openssl-dev=7.81.0-1ubuntu1.16 \
        libfontconfig1-dev=2.13.1-4.2ubuntu5 \
        libfreetype6-dev=2.11.1+dfsg-1ubuntu0.2 \
        libfribidi-dev=1.0.8-2ubuntu3.1 \
        libgit2-dev=1.1.0+dfsg.1-4.1ubuntu0.1 \
        libharfbuzz-dev=2.7.4-1ubuntu3.1 \
        libjpeg-dev=8c-2ubuntu10 \
        liblapack-dev=3.10.0-2ubuntu1 \
        libopenblas-dev=0.3.20+ds-1 \
        libpng-dev=1.6.37-3build5 \
        libssl-dev=3.0.2-0ubuntu1.15 \
        libtiff5-dev=4.3.0-6ubuntu0.8 \
        libxml2-dev=2.9.13+dfsg-1ubuntu0.4 \
        pkg-config=0.29.2-1ubuntu3 \
        r-base=4.1.2-1ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    dvc[all]==3.49.0 \
    poetry==1.8.2
