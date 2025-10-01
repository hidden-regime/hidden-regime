#!/bin/bash
# Coverage analysis with detailed output and reporting
# This script provides detailed coverage analysis with missing line reports

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running coverage analysis...${NC}"

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Clean previous coverage data
echo -e "${BLUE}Cleaning previous coverage data...${NC}"
rm -f .coverage coverage.xml
rm -rf htmlcov/

# Run tests with coverage
echo -e "${YELLOW}Running tests with detailed coverage...${NC}"
python -m pytest tests/ \
    --cov=hidden_regime \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-report=json \
    --cov-fail-under=85 \
    --tb=short \
    --strict-markers \
    --disable-warnings

# Generate coverage report
echo -e "${BLUE}Generating detailed coverage reports...${NC}"
coverage html --title="Hidden Regime Coverage Report"
coverage json

# Display coverage summary
echo -e "${YELLOW}Coverage Summary:${NC}"
coverage report --show-missing --sort=cover

# Show low-coverage modules
echo -e "${YELLOW}Modules with lowest coverage:${NC}"
coverage report --show-missing --sort=cover | head -20

# Extract coverage data for analysis
if [ -f coverage.json ]; then
    python -c "
import json
import sys

try:
    with open('coverage.json') as f:
        data = json.load(f)
    
    total_coverage = data['totals']['percent_covered']
    
    print(f'\\n📊 Overall Coverage: {total_coverage:.1f}%')
    
    if total_coverage >= 85:
        print('✅ Coverage target met!')
    elif total_coverage >= 70:
        print('⚠️  Coverage approaching target (70-85%)')
    else:
        print('❌ Coverage below target (<70%)')
    
    print('\\n📁 Module Coverage Breakdown:')
    files = data['files']
    
    # Sort by coverage percentage
    sorted_files = sorted(files.items(), key=lambda x: x[1]['summary']['percent_covered'])
    
    for file_path, file_data in sorted_files[:10]:  # Show lowest 10
        coverage_pct = file_data['summary']['percent_covered']
        missing_lines = len(file_data['missing_lines'])
        print(f'  {file_path}: {coverage_pct:.1f}% ({missing_lines} lines missing)')

except Exception as e:
    print(f'Error analyzing coverage data: {e}')
    sys.exit(1)
"
fi

echo -e "\n${GREEN}Coverage analysis complete!${NC}"
echo -e "📄 Reports generated:"
echo -e "  • HTML Report: ${BLUE}htmlcov/index.html${NC}"
echo -e "  • XML Report:  ${BLUE}coverage.xml${NC}"
echo -e "  • JSON Report: ${BLUE}coverage.json${NC}"
echo -e "\n💡 To view HTML report:"
echo -e "  Open htmlcov/index.html in your browser"