#!/bin/bash
# Complete local CI pipeline
# This script runs the complete CI pipeline locally, identical to what runs in GitHub Actions

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════╗"
echo "║              🚀 CI LOCAL PIPELINE            ║"
echo "║       Running complete CI checks locally     ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# Track timing
START_TIME=$(date +%s)

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo -e "${BLUE}🐍 Activating virtual environment...${NC}"
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Track which steps pass/fail
STEPS_PASSED=0
TOTAL_STEPS=4
FAILED_STEPS=()

# Function to run a CI step
run_ci_step() {
    local step_name=$1
    local script_path=$2
    local step_number=$3
    
    echo -e "\n${YELLOW}===================================================${NC}"
    echo -e "${YELLOW}🔄 Step $step_number/$TOTAL_STEPS: $step_name${NC}"
    echo -e "${YELLOW}===================================================${NC}"
    
    local step_start_time=$(date +%s)
    
    if [ -f "$script_path" ]; then
        if bash "$script_path"; then
            local step_end_time=$(date +%s)
            local step_duration=$((step_end_time - step_start_time))
            echo -e "${GREEN}✅ $step_name completed successfully (${step_duration}s)${NC}"
            STEPS_PASSED=$((STEPS_PASSED + 1))
        else
            local step_end_time=$(date +%s)
            local step_duration=$((step_end_time - step_start_time))
            echo -e "${RED}❌ $step_name failed (${step_duration}s)${NC}"
            FAILED_STEPS+=("$step_name")
            return 1
        fi
    else
        echo -e "${RED}❌ Script not found: $script_path${NC}"
        FAILED_STEPS+=("$step_name")
        return 1
    fi
}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}📁 Running from: $(pwd)${NC}"
echo -e "${BLUE}📁 Scripts located at: $SCRIPT_DIR${NC}"

# Step 1: Code Quality Checks
if ! run_ci_step "Code Quality (Linting)" "$SCRIPT_DIR/lint.sh" 1; then
    echo -e "${RED}💥 Code quality checks failed. Fix linting issues before proceeding.${NC}"
    exit 1
fi

# Step 2: Unit Tests (Fast)
if ! run_ci_step "Unit Tests" "$SCRIPT_DIR/test_unit.sh" 2; then
    echo -e "${RED}💥 Unit tests failed. Fix failing tests before proceeding.${NC}"
    exit 1
fi

# Step 3: Full Test Suite with Coverage
if ! run_ci_step "Full Test Suite & Coverage" "$SCRIPT_DIR/test_full.sh" 3; then
    echo -e "${RED}💥 Full test suite failed. Some tests are failing or coverage is insufficient.${NC}"
    echo -e "${YELLOW}💡 Run 'scripts/test_coverage.sh' for detailed coverage analysis${NC}"
    exit 1
fi

# Step 4: Package Build & Validation
if ! run_ci_step "Package Build & Validation" "$SCRIPT_DIR/build_check.sh" 4; then
    echo -e "${RED}💥 Package build failed. Check package configuration.${NC}"
    exit 1
fi

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo -e "\n${CYAN}"
echo "╔══════════════════════════════════════════════╗"
echo "║                🎉 CI SUMMARY                 ║"
echo "╚══════════════════════════════════════════════╝"
echo -e "${NC}"

if [ $STEPS_PASSED -eq $TOTAL_STEPS ]; then
    echo -e "${GREEN}✅ All CI checks passed! ($STEPS_PASSED/$TOTAL_STEPS)${NC}"
    echo -e "${GREEN}⏱️  Total time: ${TOTAL_DURATION}s${NC}"
    echo -e "\n${GREEN}🚀 Ready to commit and push!${NC}"
    
    # Additional success info
    echo -e "\n${BLUE}📋 What was checked:${NC}"
    echo -e "  ✅ Code formatting (Black, isort)"
    echo -e "  ✅ Linting (flake8, mypy)"
    echo -e "  ✅ Security scan (Bandit)" 
    echo -e "  ✅ Unit tests"
    echo -e "  ✅ Integration tests"
    echo -e "  ✅ Code coverage (≥60%)"
    echo -e "  ✅ Package build & validation"
    
else
    echo -e "${RED}❌ CI checks failed ($STEPS_PASSED/$TOTAL_STEPS passed)${NC}"
    echo -e "${RED}⏱️  Total time: ${TOTAL_DURATION}s${NC}"
    
    if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
        echo -e "\n${RED}💥 Failed steps:${NC}"
        for step in "${FAILED_STEPS[@]}"; do
            echo -e "  ❌ $step"
        done
    fi
    
    echo -e "\n${YELLOW}💡 To fix issues:${NC}"
    echo -e "  • Run individual scripts to debug: scripts/lint.sh, scripts/test_unit.sh, etc."
    echo -e "  • Check output above for specific error details"
    echo -e "  • Fix issues and run scripts/ci_local.sh again"
    
    exit 1
fi

echo -e "\n${CYAN}🎯 This matches exactly what will run in GitHub Actions CI!${NC}"