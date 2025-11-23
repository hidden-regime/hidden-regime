#!/bin/bash
#
# Quick Backtest Script
#
# Run a backtest in under 5 minutes using a template algorithm
#
# Usage:
#   ./scripts/quick_backtest.sh [template_name] [project_name]
#
# Examples:
#   ./scripts/quick_backtest.sh basic_regime_switching MyStrategy
#   ./scripts/quick_backtest.sh multi_asset_rotation RiskParity

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
TEMPLATE="${1:-basic_regime_switching}"
PROJECT_NAME="${2:-QuickBacktest}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Hidden-Regime Quick Backtest${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if template exists
TEMPLATE_FILE="quantconnect_templates/${TEMPLATE}.py"
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template not found: $TEMPLATE_FILE"
    echo ""
    echo "Available templates:"
    ls -1 quantconnect_templates/*.py | sed 's/.*\//  - /' | sed 's/\.py$//'
    exit 1
fi

# Check if lean CLI is available
if ! command -v lean &> /dev/null; then
    echo -e "${YELLOW}LEAN CLI not found. Using Docker directly...${NC}"
    echo ""

    # Run with Docker
    docker run --rm \
        -v "$(pwd)/quantconnect_templates:/Lean/Algorithm.Python" \
        -v "$(pwd)/docker/results:/Results" \
        quantconnect/lean:hidden-regime-latest \
        --algorithm-file "${TEMPLATE}.py"

else
    echo "Using LEAN CLI..."
    echo ""

    # Create project if it doesn't exist
    if [ ! -d "$PROJECT_NAME" ]; then
        echo "Creating project: $PROJECT_NAME"
        lean project-create "$PROJECT_NAME" --language python
    fi

    # Copy template
    echo "Copying template: $TEMPLATE"
    cp "$TEMPLATE_FILE" "$PROJECT_NAME/main.py"

    # Run backtest
    echo ""
    echo -e "${GREEN}Running backtest...${NC}"
    echo ""

    cd "$PROJECT_NAME"
    lean backtest "$PROJECT_NAME"

    echo ""
    echo -e "${GREEN}Backtest complete!${NC}"
    echo "Results: $PROJECT_NAME/backtests/"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}========================================${NC}"
