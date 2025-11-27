#!/bin/bash
#
# Build QuantConnect LEAN Docker image with Hidden-Regime
#
# Usage:
#   ./scripts/build_docker.sh [OPTIONS]
#
# Options:
#   --tag TAG       Custom image tag (default: lean-hidden-regime:latest)
#   --no-cache      Build without using Docker cache
#   --help          Show this help message
#
# Examples:
#   ./scripts/build_docker.sh
#   ./scripts/build_docker.sh --tag my-lean:v1.0
#   ./scripts/build_docker.sh --no-cache

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
TAG="lean-hidden-regime:latest"
BUILD_ARGS=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Pre-flight checks
echo -e "${BLUE}Checking prerequisites...${NC}"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "  Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check required files exist
echo ""
echo -e "${BLUE}Verifying project structure...${NC}"
echo ""

if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo -e "${RED}✗ pyproject.toml not found${NC}"
    echo "  Expected at: $PROJECT_ROOT/pyproject.toml"
    echo "  Are you running from the project root?"
    exit 1
fi
echo -e "${GREEN}✓ pyproject.toml found${NC}"

if [ ! -d "$PROJECT_ROOT/hidden_regime" ]; then
    echo -e "${RED}✗ hidden_regime directory not found${NC}"
    echo "  Expected at: $PROJECT_ROOT/hidden_regime"
    exit 1
fi
echo -e "${GREEN}✓ hidden_regime module found${NC}"

if [ ! -d "$PROJECT_ROOT/hidden_regime_mcp" ]; then
    echo -e "${RED}✗ hidden_regime_mcp directory not found${NC}"
    echo "  Expected at: $PROJECT_ROOT/hidden_regime_mcp"
    exit 1
fi
echo -e "${GREEN}✓ hidden_regime_mcp module found${NC}"

if [ ! -f "$PROJECT_ROOT/docker/Dockerfile" ]; then
    echo -e "${RED}✗ Dockerfile not found${NC}"
    echo "  Expected at: $PROJECT_ROOT/docker/Dockerfile"
    exit 1
fi
echo -e "${GREEN}✓ Dockerfile found${NC}"

echo ""
echo -e "${BLUE}Building Docker image...${NC}"
echo -e "${YELLOW}Tag: $TAG${NC}"
echo ""

# Build the image from project root
cd "$PROJECT_ROOT"
docker build \
  -f docker/Dockerfile \
  -t "$TAG" \
  $BUILD_ARGS \
  .

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Build complete!${NC}"
echo ""
echo -e "${BLUE}Image information:${NC}"
docker images --filter "reference=$TAG" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Test the image:"
echo "     docker run --rm $TAG python -c \"import hidden_regime; print('Success')\""
echo ""
echo "  2. Run a backtest:"
echo "     bash scripts/backtest_docker.sh quantconnect_templates/basic_regime_switching.py"
echo ""
