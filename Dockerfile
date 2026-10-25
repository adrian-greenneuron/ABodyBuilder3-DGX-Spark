# syntax=docker/dockerfile:1
# =============================================================================
# ABodyBuilder3 Dockerfile Optimized for NVIDIA DGX Spark (ARM64 / Blackwell)
# =============================================================================
# Using NGC PyTorch base image:
# - Pre-installed PyTorch 2.10, CUDA 13.0, cuDNN, nccl
# - Optimized for NVIDIA hardware (ARM64 + Blackwell sm_120/sm_121 support)
# - Includes Flash Attention 2.7.4, Triton 3.5.0, CUTLASS 4.0.0
# =============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
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
    libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# OpenMM (Source Build for Blackwell/sm_120 Support)
# -----------------------------------------------------------------------------
# Conda binaries are incompatible with Blackwell driver (PTX error).
# We must build from source linking against the local CUDA toolkit.
WORKDIR /tmp
# hadolint ignore=DL3003,SC2046
RUN git clone https://github.com/openmm/openmm.git \
    && git -C openmm checkout --detach 07b738c560f31f539674528c3a8ca629fc8320ba \
    && cd openmm \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    && make -j"$(nproc)" install \
    && cd python && OPENMM_INCLUDE_PATH=/usr/local/include OPENMM_LIB_PATH=/usr/local/lib python3 setup.py install \
    && cd /tmp && rm -rf openmm

# Exact Python versions from the original image; NGC supplies the pinned Torch stack.
COPY runtime-constraints.txt /opt/runtime-constraints.txt

# Install pdbfixer from source
# hadolint ignore=DL3013
RUN \
    pip install --no-cache-dir -c /opt/runtime-constraints.txt git+https://github.com/openmm/pdbfixer.git@94cfa4c0ca551cdc5f13320f9a658efd59f2b881

# -----------------------------------------------------------------------------
# Reduce (MolProbity/Richardson Lab) for fast hydrogen addition
# -----------------------------------------------------------------------------
# reduce needs het dictionary files for proper operation
# hadolint ignore=DL3003
RUN git clone https://github.com/rlabduke/reduce.git /tmp/reduce \
    && git -C /tmp/reduce checkout --detach 9d4ac1f5f103daa18ea471f11fe1cf9b2cd8ca1f \
    && cd /tmp/reduce \
    && mkdir build && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && cp reduce_src/reduce /usr/local/bin/ \
    && cp /tmp/reduce/reduce_wwPDB_het_dict.txt /usr/local/ \
    && cd / && rm -rf /tmp/reduce

# -----------------------------------------------------------------------------
# Python dependencies (captured original versions for a controlled comparison)
# -----------------------------------------------------------------------------
# Keep the model/PyTorch stack at the tested versions; runtime constraints prevent drift.
# hadolint ignore=DL3013
RUN \
    pip install --no-cache-dir -c /opt/runtime-constraints.txt \
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
ARG SOURCE_REF=8e4b5af7c347dadcc52678c7ee61ea975eeef1a8
RUN git clone https://github.com/adrian-greenneuron/ABodyBuilder3-DGX-Spark.git /opt/abodybuilder3 \
    && git -C /opt/abodybuilder3 checkout --detach ${SOURCE_REF}
WORKDIR /opt/abodybuilder3

# -----------------------------------------------------------------------------
# Install ABodyBuilder3
# -----------------------------------------------------------------------------
# hadolint ignore=DL3013
RUN \
    pip install --no-cache-dir -c /opt/runtime-constraints.txt --no-build-isolation -e .

# -----------------------------------------------------------------------------
# Download Model Weights
# -----------------------------------------------------------------------------
# Model weights are required for inference
RUN mkdir -p output/ zenodo/ \
    && wget -q --progress=dot:giga -P zenodo/ https://zenodo.org/records/11354577/files/output.tar.gz \
    && echo "2c1d734ed74013865bd95fc2f95cff1e  zenodo/output.tar.gz" | md5sum --check - \
    && tar -xzf zenodo/output.tar.gz -C output/ \
    && rm -rf zenodo/ \
    && echo "835f0f4c3a183dfa71084a28b59aab87bc778c3c1f07682df879d8423f138938  output/base-loss/best_second_stage.ckpt" | sha256sum --check - \
    && echo "16be44c644eb31d63c78c22166d6cea5cee8d2b3eb2cb36ebee87f5cfeeb0177  output/language-loss/best_second_stage.ckpt" | sha256sum --check - \
    && echo "421aa6c52a5152f781b35007411985146fb520a90286a955161cb8e50584d615  output/plddt-loss/best_second_stage.ckpt" | sha256sum --check -

# -----------------------------------------------------------------------------
# Validation & Runtime
# -----------------------------------------------------------------------------
# Verify all imports work
RUN python3 -c "import abodybuilder3; import torch; import openmm; import pdbfixer; print(f'ABodyBuilder3 on PyTorch {torch.__version__}, CUDA {torch.version.cuda}, OpenMM {openmm.version.short_version}')"

WORKDIR /opt/abodybuilder3
CMD ["/bin/bash"]
