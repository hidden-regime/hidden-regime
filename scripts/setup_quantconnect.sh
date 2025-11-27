#!/bin/bash
#
# Setup script for QuantConnect LEAN with Hidden-Regime
#
# This script:
# 1. Verifies Docker is installed and running
# 2. Builds the custom Docker image
# 3. Provides quick-start instructions
#
# Usage:
#   ./scripts/setup_quantconnect.sh [OPTIONS]
#
# Options:
#   --help      Show this help message
#
# What this sets up:
#   • Docker image with QuantConnect LEAN + hidden-regime
#   • Simple docker run workflow for strategy backtesting
#   • No external dependencies (Docker is sufficient)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Banner
echo ""
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Hidden-Regime × QuantConnect LEAN Setup                ║
║                                                           ║
║   Docker-based strategy backtesting                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites${NC}"
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo ""
    echo "Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed ($(docker --version))${NC}"

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running${NC}"
    echo ""
    echo "Please start Docker and try again"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

echo ""

# Step 2: Build Docker image
echo -e "${BLUE}Step 2: Building Docker image${NC}"
echo ""

bash scripts/build_docker.sh

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

echo ""

# Step 3: Summary and next steps
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Setup Complete! ✓                                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${GREEN}Your Docker environment is ready for backtesting!${NC}"
echo ""

echo -e "${BLUE}Quick Start:${NC}"
echo ""
echo "  1. Test the setup:"
echo "     docker run --rm lean-hidden-regime:latest python -c \\\"import hidden_regime; print('Success')\\\""
echo ""
echo "  2. Run a strategy backtest:"
echo "     bash scripts/backtest_docker.sh quantconnect_templates/basic_regime_switching.py"
echo ""
echo "  3. View results:"
echo "     ls backtest_results/"
echo ""

echo -e "${BLUE}Available templates:${NC}"
echo ""
ls -1 quantconnect_templates/*.py | sed 's/quantconnect_templates\//  • /' | sed 's/\.py$//'
echo ""

echo -e "${BLUE}Documentation:${NC}"
echo "  • Usage guide: ./quantconnect_templates/README.md"
echo "  • Strategy examples: ./quantconnect_templates/TEMPLATES_GUIDE.md"
echo ""

echo -e "${GREEN}Happy backtesting! 🚀${NC}"
echo ""
