#!/bin/bash
# Complete test suite with coverage reporting
# This script runs the full test suite including unit, integration, and performance tests
# with comprehensive coverage analysis.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running complete test suite with coverage...${NC}"

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Run pytest with coverage
echo -e "${YELLOW}Running pytest with coverage reporting...${NC}"
python -m pytest tests/ \
    --cov=hidden_regime \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-report=html \
    --verbose \
    --tb=short \
    --strict-markers \
    --disable-warnings

# Check if coverage meets minimum threshold
COVERAGE_THRESHOLD=60
echo -e "${YELLOW}Checking coverage threshold (${COVERAGE_THRESHOLD}%)...${NC}"

# Extract coverage percentage from coverage.xml if it exists
if [ -f coverage.xml ]; then
    COVERAGE=$(python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    coverage = float(root.attrib['line-rate']) * 100
    print(f'{coverage:.1f}')
except:
    print('0.0')
")
    
    echo "Current coverage: ${COVERAGE}%"
    
    if (( $(echo "$COVERAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then
        echo -e "${GREEN}✓ Coverage threshold met (${COVERAGE}% >= ${COVERAGE_THRESHOLD}%)${NC}"
    else
        echo -e "${RED}✗ Coverage below threshold (${COVERAGE}% < ${COVERAGE_THRESHOLD}%)${NC}"
        echo "Run 'scripts/test_coverage.sh' for detailed coverage analysis"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Full test suite completed successfully!${NC}"
echo "Coverage reports available at:"
echo "  - HTML: htmlcov/index.html"
echo "  - XML: coverage.xml"