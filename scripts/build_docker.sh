#!/bin/bash
#
# Build QuantConnect LEAN Docker image with Hidden-Regime
#
# Usage:
#   ./scripts/build_docker.sh [--pypi] [--tag TAG]
#
# Options:
#   --pypi      Build using PyPI version (production)
#   --tag TAG   Custom image tag (default: latest)
#   --no-cache  Build without cache

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building LEAN with Hidden-Regime${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Default values
USE_PYPI=false
IMAGE_TAG="latest"
NO_CACHE=""
DOCKERFILE="docker/Dockerfile"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pypi)
            USE_PYPI=true
            DOCKERFILE="docker/Dockerfile.pypi"
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Image name
IMAGE_NAME="quantconnect/lean:hidden-regime-${IMAGE_TAG}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Dockerfile: ${DOCKERFILE}"
echo "  Image: ${IMAGE_NAME}"
echo "  Build Date: ${BUILD_DATE}"
echo "  VCS Ref: ${VCS_REF}"
echo "  Use PyPI: ${USE_PYPI}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Build the image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    ${NO_CACHE} \
    -f ${DOCKERFILE} \
    -t ${IMAGE_NAME} \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg VCS_REF="${VCS_REF}" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Image: ${IMAGE_NAME}"
    echo ""
    echo "To configure LEAN CLI to use this image:"
    echo -e "${YELLOW}  lean config set engine-image ${IMAGE_NAME}${NC}"
    echo ""
    echo "To run a backtest:"
    echo -e "${YELLOW}  docker run --rm -v \$(pwd):/Lean/Algorithm.Python ${IMAGE_NAME}${NC}"
    echo ""
    echo "To start docker-compose services:"
    echo -e "${YELLOW}  cd docker && docker-compose up${NC}"
    echo ""
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Show image size
echo "Image size:"
docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
