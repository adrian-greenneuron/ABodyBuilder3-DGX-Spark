# syntax=docker/dockerfile:1
# =============================================================================
# ABodyBuilder3 Dockerfile Optimized for NVIDIA DGX Spark (ARM64 / Blackwell)
# =============================================================================
# Using NGC PyTorch base image:
# - Pre-installed PyTorch 2.10, CUDA 13.0, cuDNN, nccl
# - Optimized for NVIDIA hardware (ARM64 + Blackwell sm_120/sm_121 support)
# - Includes Flash Attention 2.7.4, Triton 3.5.0, CUTLASS 4.0.0
# =============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE}

LABEL maintainer="ABodyBuilder3"
LABEL description="ABodyBuilder3 for DGX Spark (Blackwell sm_120/sm_121)"
LABEL cuda.version="13.0"
LABEL pytorch.version="2.10"

# -----------------------------------------------------------------------------
# Build Configuration
# -----------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="12.0" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    aria2 \
    cmake \
    doxygen \
    swig \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# OpenMM (Source Build for Blackwell/sm_120 Support)
# -----------------------------------------------------------------------------
# Conda binaries are incompatible with Blackwell driver (PTX error).
# We must build from source linking against the local CUDA toolkit.
WORKDIR /tmp
# hadolint ignore=DL3003,SC2046
RUN git clone https://github.com/openmm/openmm.git \
    && cd openmm \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    && make -j"$(nproc)" install \
    && cd python && OPENMM_INCLUDE_PATH=/usr/local/include OPENMM_LIB_PATH=/usr/local/lib python3 setup.py install \
    && cd /tmp && rm -rf openmm

# Install pdbfixer from source
# hadolint ignore=DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/openmm/pdbfixer.git

# -----------------------------------------------------------------------------
# Python Dependencies (Relaxed versions for CUDA 13 / PyTorch 2.10 compatibility)
# -----------------------------------------------------------------------------
# Note: We relax numpy, scipy, and pandas versions from the original pinned versions
# to maintain compatibility with the newer PyTorch 2.10 in the NGC container.
# hadolint ignore=DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    loguru \
    typer \
    "click!=8.1.0" \
    einops \
    dm-tree \
    ml_collections \
    tqdm \
    "lightning>=2.0.4" \
    python-box \
    tensorboard \
    tensorboardX \
    cloudpathlib \
    levenshtein \
    scipy \
    pandas \
    transformers \
    sentencepiece \
    accelerate \
    biopython

# -----------------------------------------------------------------------------
# Clone ABodyBuilder3
# -----------------------------------------------------------------------------
WORKDIR /opt
COPY . /opt/abodybuilder3
WORKDIR /opt/abodybuilder3

# -----------------------------------------------------------------------------
# Install ABodyBuilder3
# -----------------------------------------------------------------------------
# hadolint ignore=DL3013
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation -e .

# -----------------------------------------------------------------------------
# Download Model Weights
# -----------------------------------------------------------------------------
# Model weights are required for inference
RUN mkdir -p output/ zenodo/ \
    && wget -q --progress=dot:giga -P zenodo/ https://zenodo.org/records/11354577/files/output.tar.gz \
    && tar -xzf zenodo/output.tar.gz -C output/ \
    && rm -rf zenodo/

# -----------------------------------------------------------------------------
# Validation & Runtime
# -----------------------------------------------------------------------------
# Verify all imports work
RUN python3 -c "import abodybuilder3; import torch; import openmm; import pdbfixer; print(f'ABodyBuilder3 on PyTorch {torch.__version__}, CUDA {torch.version.cuda}, OpenMM {openmm.version.short_version}')"

WORKDIR /opt/abodybuilder3
CMD ["/bin/bash"]
