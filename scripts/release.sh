#!/bin/bash
# PyPI Release Script
# Automates the complete release workflow: clean → build → testpypi → verify → pypi
# Usage: ./scripts/release.sh [--dry-run]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo -e "${YELLOW}🧪 DRY-RUN MODE: No uploads will be performed${NC}"
fi

# Banner
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════╗"
echo "║          📦 PYPI RELEASE SCRIPT            ║"
echo "║    Automated deployment to PyPI/TestPyPI  ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

# Track timing
START_TIME=$(date +%s)
TOTAL_STEPS=6

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo -e "${BLUE}🐍 Activating virtual environment...${NC}"
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Helper function for step output
print_step() {
    local step_num=$1
    local step_name=$2
    echo -e "\n${YELLOW}═══════════════════════════════════════════${NC}"
    echo -e "${YELLOW}🔄 Step $step_num/$TOTAL_STEPS: $step_name${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
}

# Helper function for success message
print_success() {
    local message=$1
    echo -e "${GREEN}✅ $message${NC}"
}

# Helper function for error message
print_error() {
    local message=$1
    echo -e "${RED}❌ $message${NC}"
}

# ============================================================================
# STEP 1: Environment Setup & Validation
# ============================================================================
print_step 1 "Environment Setup & Validation"

# Check for uncommitted changes (safety measure)
if [ -n "$(git status --porcelain)" ]; then
    print_error "Repository has uncommitted changes. Please commit or stash before releasing."
    echo -e "${YELLOW}Run 'git status' to see uncommitted changes${NC}"
    exit 1
fi
print_success "Git repository is clean"

# Verify required tools
echo "Checking for required tools..."
for tool in python pip git; do
    if ! command -v "$tool" &> /dev/null; then
        print_error "Required tool not found: $tool"
        exit 1
    fi
done

# Check for Python modules
for module in build twine; do
    if ! python -c "import $module" 2>/dev/null; then
        print_error "Required Python module not found: $module"
        echo -e "${YELLOW}Install with: pip install $module${NC}"
        exit 1
    fi
done
print_success "All required tools available"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
echo -e "${BLUE}📁 Working directory: $(pwd)${NC}"

# ============================================================================
# STEP 2: Clean Build Artifacts
# ============================================================================
print_step 2 "Clean Build Artifacts"

echo "Removing previous build artifacts..."
rm -rf build/ dist/ *.egg-info/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
print_success "Build artifacts cleaned"

# ============================================================================
# STEP 3: Build Package
# ============================================================================
print_step 3 "Build Package"

echo "Running build_check.sh to build and validate package..."
if ! bash "$SCRIPT_DIR/build_check.sh"; then
    print_error "Package build failed. Check output above for details."
    exit 1
fi
print_success "Package built and validated"

# Extract version from wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n1)
if [ -z "$WHEEL_FILE" ]; then
    print_error "No wheel file found in dist/"
    exit 1
fi

# Extract version from wheel filename (hidden_regime-X.Y.Z-py3-*.whl)
VERSION=$(basename "$WHEEL_FILE" | sed -n 's/hidden_regime-\([^-]*\).*/\1/p')
echo -e "${BLUE}📦 Package version: $VERSION${NC}"

# ============================================================================
# STEP 4: Upload to TestPyPI
# ============================================================================
print_step 4 "Upload to TestPyPI"

echo "Uploading artifacts to TestPyPI..."
echo "  Source: $(ls dist/*.tar.gz | xargs basename)"
echo "  Wheel:  $(basename "$WHEEL_FILE")"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🧪 DRY-RUN: Skipping actual upload${NC}"
else
    if twine upload --repository testpypi dist/* --non-interactive; then
        print_success "Successfully uploaded to TestPyPI"
    else
        print_error "Failed to upload to TestPyPI"
        echo -e "${YELLOW}💡 Check your TestPyPI credentials in ~/.pypirc${NC}"
        exit 1
    fi
fi

# ============================================================================
# STEP 5: Verify Package Installation from TestPyPI
# ============================================================================
print_step 5 "Verify Installation from TestPyPI"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🧪 DRY-RUN: Skipping TestPyPI installation test${NC}"
    echo "In production mode, this step would:"
    echo "  1. Create isolated test venv"
    echo "  2. Install package from TestPyPI"
    echo "  3. Verify 'import hidden_regime' and version match"
    TEMP_DIR=""
else
    echo "Creating temporary virtual environment for testing..."

    # Create temp directory for test venv
    TEMP_DIR=$(mktemp -d)
    TEST_VENV="$TEMP_DIR/test_venv"

    # Create and activate test venv
    python -m venv "$TEST_VENV"
    # shellcheck source=/dev/null
    source "$TEST_VENV/bin/activate"

    echo "Installing package from TestPyPI..."

    # Install from testpypi, falling back to PyPI for dependencies
    # (TestPyPI doesn't always have all dependencies)
    if pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
        hidden-regime==$VERSION --quiet 2>/dev/null || \
       pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
        hidden-regime --quiet 2>/dev/null; then

        # Test import
        python -c "
import hidden_regime
import sys
print(f'✅ Package imported successfully from TestPyPI')
print(f'   Version: {hidden_regime.__version__}')

# Verify version matches
if hidden_regime.__version__ != '$VERSION':
    print(f'⚠️  Version mismatch: Expected $VERSION, got {hidden_regime.__version__}')
    sys.exit(1)
"
        print_success "Package verification passed"
    else
        print_error "Failed to install from TestPyPI"
        echo -e "${YELLOW}💡 Troubleshooting:${NC}"
        echo -e "  • Check TestPyPI upload completed: https://test.pypi.org/project/hidden-regime"
        echo -e "  • Verify credentials in ~/.pypirc"
        echo -e "  • Try manual installation: pip install -i https://test.pypi.org/simple/ hidden-regime"
        # Clean up test venv
        deactivate 2>/dev/null || true
        rm -rf "$TEMP_DIR"
        exit 1
    fi
fi

# Clean up test venv (only if created in non-dry-run mode)
if [ -n "$TEMP_DIR" ] && [ "$TEMP_DIR" != "" ]; then
    deactivate 2>/dev/null || true
    rm -rf "$TEMP_DIR"
    echo -e "${BLUE}📁 Cleaned up temporary test environment${NC}"
fi

# ============================================================================
# STEP 6: Upload to Production PyPI
# ============================================================================
print_step 6 "Upload to Production PyPI"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🧪 DRY-RUN: Skipping production upload${NC}"
    echo -e "${BLUE}📦 Artifacts ready in dist/:${NC}"
    ls -lh dist/
else
    echo -e "${RED}⚠️  IMPORTANT: This will upload to the production PyPI registry${NC}"
    echo -e "${RED}   Version: $VERSION${NC}"
    echo ""

    read -p "Do you want to upload to production PyPI? (yes/no): " -r
    echo

    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Uploading to production PyPI..."

        if twine upload dist/* --non-interactive; then
            print_success "Successfully uploaded to production PyPI!"
            echo ""
            echo -e "${GREEN}📦 Release complete!${NC}"
            echo -e "${BLUE}Installation command for users:${NC}"
            echo -e "  ${CYAN}pip install hidden-regime==$VERSION${NC}"
            echo ""
            echo -e "${BLUE}Package information:${NC}"
            echo -e "  PyPI: https://pypi.org/project/hidden-regime/$VERSION"
            echo -e "  Docs: https://docs.hiddenregime.com"
        else
            print_error "Failed to upload to production PyPI"
            echo -e "${YELLOW}💡 Check your PyPI credentials in ~/.pypirc${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}📋 Upload cancelled. Artifacts remain in dist/${NC}"
        echo -e "${BLUE}To retry uploading, run:${NC}"
        echo -e "  ${CYAN}twine upload dist/*${NC}"
    fi
fi

# ============================================================================
# Final Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════╗"
echo "║            🎉 RELEASE COMPLETE            ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Release Summary:${NC}"
echo -e "  Version: $VERSION"
echo -e "  Duration: ${TOTAL_DURATION}s"

if [ "$DRY_RUN" = true ]; then
    echo -e "  Mode: ${YELLOW}DRY-RUN (no uploads performed)${NC}"
else
    echo -e "  Mode: ${GREEN}PRODUCTION${NC}"
fi

echo ""
echo -e "${BLUE}Artifacts in dist/:${NC}"
ls -lh dist/

echo ""
print_success "All steps completed successfully!"
