#!/bin/bash
# =============================================================================
# Build and run ABodyBuilder3 Docker image for DGX Spark
# =============================================================================
set -euo pipefail

# Configuration
IMAGE_NAME="${IMAGE_NAME:-abodybuilder3-dev-spark}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
AOT_BUILD_DIR="${AOT_BUILD_DIR:-$(pwd)/.aot-build/$(date -u +%Y%m%dT%H%M%SZ)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [build|run|test|shell]"
    echo ""
    echo "Commands:"
    echo "  build   Build the Docker image"
    echo "  run     Run a command in the container"
    echo "  test    Run import test to verify installation"
    echo "  shell   Start an interactive shell in the container"
    echo ""
    echo "Environment variables:"
    echo "  IMAGE_NAME    Name for the Docker image (default: abodybuilder3-dev-spark)"
    echo "  IMAGE_TAG     Tag for the Docker image (default: latest)"
    exit 1
}

build() {
    echo -e "${GREEN}Building ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
    
    # Check for GPU before building
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found. Building without GPU verification.${NC}"
    fi
    
    local cache_args=()
    if [[ "${NO_CACHE:-0}" == "1" ]]; then
        cache_args+=(--no-cache)
    fi
    local runtime_tag="${IMAGE_NAME}:${IMAGE_TAG}-runtime"
    DOCKER_BUILDKIT=1 docker build "${cache_args[@]}" \
        --progress=plain \
        -t "${runtime_tag}" \
        -f "${DOCKERFILE}" \
        .

    mkdir -p "${AOT_BUILD_DIR}"
    # GPU access is unavailable to ordinary docker build; prepare packages in
    # the freshly built runtime, then put only executables/metadata in the image.
    docker run --rm --gpus all --network none --ipc=host \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e TORCHINDUCTOR_COMPILE_THREADS=4 \
        -v "$(realpath "${AOT_BUILD_DIR}"):/aot" \
        "${runtime_tag}" python3 -u tools/prepare_aot.py --output /aot/packages
    DOCKER_BUILDKIT=1 docker build --network none --pull=false --progress=plain \
        --build-arg "RUNTIME_IMAGE=${runtime_tag}" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile.aot "${AOT_BUILD_DIR}"
    
    echo -e "${GREEN}Build complete: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
}

run_container() {
    docker run --rm -it \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$(pwd)":/workspace \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        "$@"
}

test_image() {
    echo -e "${GREEN}Testing ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
    
    run_container python3 -c "
import torch
import openmm
import pdbfixer
import abodybuilder3

print('=' * 60)
print('ABodyBuilder3 Docker Image Test')
print('=' * 60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'OpenMM version: {openmm.version.short_version}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('=' * 60)
print('All imports successful!')
"
}

shell() {
    echo -e "${GREEN}Starting shell in ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
    run_container /bin/bash
}

# Main
case "${1:-}" in
    build)
        build
        ;;
    run)
        shift
        run_container "$@"
        ;;
    test)
        test_image
        ;;
    shell)
        shell
        ;;
    *)
        usage
        ;;
esac
