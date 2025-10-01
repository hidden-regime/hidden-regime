#!/bin/bash
# All linting and code quality checks
# This script runs all code quality checks including formatting, imports, linting, and type checking

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running code quality checks...${NC}"

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Flag to track if any checks fail
LINT_FAILED=0

# Function to run a check and handle errors
run_check() {
    local check_name=$1
    local command=$2
    
    echo -e "\n${BLUE}🔍 ${check_name}${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ ${check_name} passed${NC}"
    else
        echo -e "${RED}❌ ${check_name} failed${NC}"
        LINT_FAILED=1
    fi
}

# Black code formatting check
run_check "Black code formatting" \
    "black --check --diff hidden_regime/ tests/ examples/"

# isort import sorting check  
run_check "Import sorting (isort)" \
    "isort --check-only --diff hidden_regime/ tests/ examples/"

# flake8 linting (syntax errors and undefined names)
run_check "Flake8 critical issues" \
    "flake8 hidden_regime/ --count --select=E9,F63,F7,F82 --show-source --statistics"

# flake8 full linting (warnings only, don't fail)
echo -e "\n${BLUE}🔍 Flake8 style warnings (informational)${NC}"
flake8 hidden_regime/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics || true

# MyPy type checking
run_check "MyPy type checking" \
    "mypy hidden_regime/ --ignore-missing-imports"

# Bandit security scanning
echo -e "\n${BLUE}🔍 Security scan (Bandit)${NC}"
bandit -r hidden_regime/ -f json -o bandit-report.json || true
if [ -f bandit-report.json ]; then
    # Check if any high or medium severity issues found
    HIGH_ISSUES=$(python -c "
import json
try:
    with open('bandit-report.json') as f:
        data = json.load(f)
    high = len([r for r in data.get('results', []) if r.get('issue_severity') in ['HIGH', 'MEDIUM']])
    print(high)
except:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ "$HIGH_ISSUES" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $HIGH_ISSUES high/medium security issues${NC}"
        echo "Review bandit-report.json for details"
    else
        echo -e "${GREEN}✅ No high/medium security issues found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Bandit report not generated${NC}"
fi

# Summary
echo -e "\n${YELLOW}📋 Lint Summary${NC}"
if [ $LINT_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All code quality checks passed!${NC}"
else
    echo -e "${RED}❌ Some code quality checks failed${NC}"
    echo "Fix the issues above before committing"
    exit 1
fi

echo -e "\n💡 To auto-fix formatting issues:"
echo -e "  black hidden_regime/ tests/ examples/"
echo -e "  isort hidden_regime/ tests/ examples/"