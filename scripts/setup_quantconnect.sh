#!/bin/bash
#
# Complete setup script for QuantConnect LEAN with Hidden-Regime
#
# This script:
# 1. Checks prerequisites
# 2. Installs LEAN CLI (optional)
# 3. Builds custom Docker image
# 4. Configures LEAN CLI
# 5. Creates example project
#
# Usage:
#   ./scripts/setup_quantconnect.sh [--skip-lean-cli] [--skip-docker]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Flags
SKIP_LEAN_CLI=false
SKIP_DOCKER=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-lean-cli)
            SKIP_LEAN_CLI=true
            shift
            ;;
        --skip-docker)
            SKIP_DOCKER=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Hidden-Regime × QuantConnect LEAN Setup                ║
║                                                           ║
║   Setting up your 5-minute backtest workflow...          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Step 1: Check prerequisites
echo -e "${GREEN}Step 1: Checking prerequisites...${NC}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "  Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
else
    echo -e "${GREEN}✓ Docker installed$(docker --version | cut -d' ' -f3)${NC}"
fi

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running${NC}"
    echo "  Please start Docker and try again"
    exit 1
else
    echo -e "${GREEN}✓ Docker is running${NC}"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠ Python 3 not found (optional for local development)${NC}"
else
    echo -e "${GREEN}✓ Python 3 installed ($(python3 --version | cut -d' ' -f2))${NC}"
fi

# Check .NET (for LEAN CLI)
if ! command -v dotnet &> /dev/null; then
    echo -e "${YELLOW}⚠ .NET SDK not found (required for LEAN CLI)${NC}"
    if [ "$SKIP_LEAN_CLI" = false ]; then
        echo "  Install from: https://dotnet.microsoft.com/download"
        echo "  Or run with --skip-lean-cli to skip LEAN CLI installation"
        exit 1
    fi
else
    echo -e "${GREEN}✓ .NET SDK installed ($(dotnet --version))${NC}"
fi

echo ""

# Step 2: Install LEAN CLI (optional)
if [ "$SKIP_LEAN_CLI" = false ]; then
    echo -e "${GREEN}Step 2: Installing LEAN CLI...${NC}"
    echo ""

    if command -v lean &> /dev/null; then
        echo -e "${GREEN}✓ LEAN CLI already installed${NC}"
    else
        echo "Installing LEAN CLI via dotnet tool..."
        dotnet tool install -g QuantConnect.Lean.CLI

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ LEAN CLI installed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to install LEAN CLI${NC}"
            exit 1
        fi
    fi
    echo ""
else
    echo -e "${YELLOW}Step 2: Skipping LEAN CLI installation${NC}"
    echo ""
fi

# Step 3: Build Docker image
if [ "$SKIP_DOCKER" = false ]; then
    echo -e "${GREEN}Step 3: Building custom LEAN Docker image...${NC}"
    echo ""

    # Build the image
    bash scripts/build_docker.sh --tag latest

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Docker build failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}Step 3: Skipping Docker build${NC}"
    echo ""
fi

# Step 4: Configure LEAN CLI
if [ "$SKIP_LEAN_CLI" = false ] && command -v lean &> /dev/null; then
    echo -e "${GREEN}Step 4: Configuring LEAN CLI...${NC}"
    echo ""

    IMAGE_NAME="quantconnect/lean:hidden-regime-latest"

    # Configure engine image
    lean config set engine-image ${IMAGE_NAME}
    echo -e "${GREEN}✓ Configured LEAN to use: ${IMAGE_NAME}${NC}"

    # Configure research image (optional)
    lean config set research-image ${IMAGE_NAME}
    echo -e "${GREEN}✓ Configured research image${NC}"

    echo ""
else
    echo -e "${YELLOW}Step 4: Skipping LEAN CLI configuration${NC}"
    echo ""
fi

# Step 5: Summary and next steps
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Setup Complete! 🎉                                     ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}Your 5-minute backtest workflow is ready!${NC}"
echo ""
echo "Quick Start Options:"
echo ""

if [ "$SKIP_LEAN_CLI" = false ]; then
    echo -e "${YELLOW}Option 1: LEAN CLI (Recommended)${NC}"
    echo "  1. Create a new project:"
    echo "     lean project-create MyRegimeStrategy"
    echo ""
    echo "  2. Copy a template:"
    echo "     cp quantconnect_templates/basic_regime_switching.py MyRegimeStrategy/main.py"
    echo ""
    echo "  3. Run backtest:"
    echo "     cd MyRegimeStrategy && lean backtest MyRegimeStrategy"
    echo ""
fi

echo -e "${YELLOW}Option 2: Docker Compose${NC}"
echo "  1. Start services:"
echo "     cd docker && docker-compose up -d"
echo ""
echo "  2. View logs:"
echo "     docker-compose logs -f lean-hidden-regime"
echo ""

echo -e "${YELLOW}Option 3: Direct Docker${NC}"
echo "  docker run --rm -v \$(pwd)/quantconnect_templates:/Lean/Algorithm.Python \\"
echo "    quantconnect/lean:hidden-regime-latest"
echo ""

echo "Documentation:"
echo "  • Templates: ./quantconnect_templates/README.md"
echo "  • Roadmap: ./QC_ROADMAP.md"
echo "  • Phase 1 Summary: ./QUANTCONNECT_PHASE1_COMPLETE.md"
echo ""

echo -e "${GREEN}Happy Trading! 🚀${NC}"
