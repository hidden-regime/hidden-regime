#!/bin/bash
# Fast unit tests only (excludes integration and performance tests)
# This script runs only unit tests for quick feedback during development

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running unit tests only (fast)...${NC}"

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Run only unit tests (exclude integration, performance, and slow tests)
echo -e "${YELLOW}Running unit tests...${NC}"
python -m pytest tests/ \
    -m "not integration and not performance and not slow" \
    --tb=short \
    --strict-markers \
    --disable-warnings \
    -x \
    --durations=10

echo -e "${GREEN}✓ Unit tests completed successfully!${NC}"
echo "For full test suite with coverage, run: scripts/test_full.sh"