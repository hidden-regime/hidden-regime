#!/bin/bash
#
# Run strategy backtest in Docker container
#
# Usage:
#   bash scripts/backtest_docker.sh <strategy.py> [OPTIONS]
#
# Arguments:
#   strategy.py     Path to strategy file (relative to project root)
#
# Options:
#   --image TAG     Docker image tag (default: lean-hidden-regime:latest)
#   --help          Show this help message
#
# Examples:
#   bash scripts/backtest_docker.sh quantconnect_templates/basic_regime_switching.py
#   bash scripts/backtest_docker.sh my_strategy.py --image my-lean:v1.0
#
# The script:
# 1. Verifies Docker and image are available
# 2. Validates strategy file exists
# 3. Runs backtest in isolated Docker container
# 4. Mounts quantconnect_templates as volume
# 5. Extracts results to ./backtest_results/
# 6. Displays summary

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
STRATEGY_FILE="${1:-}"
IMAGE_NAME="lean-hidden-regime:latest"
RESULTS_DIR="backtest_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse remaining arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_NAME="$2"
            shift 2
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

# Validate inputs
if [ -z "$STRATEGY_FILE" ]; then
    echo -e "${RED}Error: Strategy file not specified${NC}"
    echo ""
    grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //'
    exit 1
fi

# Header
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   QuantConnect LEAN Strategy Backtest Runner    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check Docker is running
echo -e "${BLUE}Checking prerequisites...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check if strategy file exists
STRATEGY_ABS="$PROJECT_ROOT/$STRATEGY_FILE"
if [ ! -f "$STRATEGY_ABS" ]; then
    echo -e "${RED}✗ Strategy file not found: $STRATEGY_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Strategy file found${NC}"

# Check Docker image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Docker image not found: $IMAGE_NAME${NC}"
    echo ""
    echo "Building Docker image (this may take a few minutes)..."
    bash "$SCRIPT_DIR/build_docker.sh" --tag "$IMAGE_NAME" || {
        echo -e "${RED}✗ Failed to build Docker image${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ Docker image ready${NC}"

# Create results directory with timestamp
STRATEGY_NAME=$(basename "$STRATEGY_FILE" .py)
RESULT_PATH="$PROJECT_ROOT/$RESULTS_DIR/${STRATEGY_NAME}_${TIMESTAMP}"
mkdir -p "$RESULT_PATH"

# Copy strategy to results for reference
cp "$STRATEGY_ABS" "$RESULT_PATH/strategy.py"

echo ""
echo -e "${BLUE}Running backtest...${NC}"
echo -e "${YELLOW}Strategy: $STRATEGY_FILE${NC}"
echo -e "${YELLOW}Results:  $RESULT_PATH${NC}"
echo ""

# Run backtest in Docker container with volumes
# - Templates mounted as read-only volume for strategy access
# - Results volume for output (writable)
docker run --rm \
    --entrypoint bash \
    -w /Lean/Launcher/bin/Debug \
    -v "$PROJECT_ROOT/quantconnect_templates:/Lean/Algorithm.Python:ro" \
    -v "$RESULT_PATH:/Lean/Results:rw" \
    "$IMAGE_NAME" \
    -c '
        # Strategy file is mounted read-only, so we need to use absolute path
        STRATEGY_FILE="/Lean/Algorithm.Python/'"$(basename "$STRATEGY_FILE")"'"

        # Run LEAN backtest (with timeout to prevent hanging)
        # Working directory is set to /Lean/Launcher/bin/Debug for correct relative path resolution
        echo "Starting backtest execution..."
        timeout 600 dotnet QuantConnect.Lean.Launcher.dll \
            --algorithm-language Python \
            --algorithm-location "$STRATEGY_FILE" \
            --results-destination-folder /Lean/Results \
            2>&1 | tee /Lean/Results/backtest.log

        echo ""
        echo "Backtest execution completed"
    ' || {
    echo -e "${YELLOW}⚠ Backtest execution returned non-zero status${NC}"
}

echo ""
echo -e "${GREEN}✓ Backtest complete${NC}"
echo ""
echo -e "${BLUE}Results directory: $RESULT_PATH${NC}"
echo ""

# List generated files
if [ -d "$RESULT_PATH" ] && [ "$(ls -A "$RESULT_PATH")" ]; then
    echo -e "${BLUE}Generated files:${NC}"
    ls -lh "$RESULT_PATH" | awk 'NR > 1 { printf "  %s  %s\n", $5, $9 }'
else
    echo -e "${YELLOW}⚠ No results generated (check backtest.log for errors)${NC}"
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  • View logs: tail -f $RESULT_PATH/backtest.log"
echo "  • Analyze results: python scripts/analyze_backtest.py $RESULT_PATH"
echo ""
