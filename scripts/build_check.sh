#!/bin/bash
# Package building and validation
# This script builds the package and validates it can be installed and imported

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building and validating package...${NC}"

# Activate virtual environment if it exists
if [ -f "$HOME/hidden-regime-pyenv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$HOME/hidden-regime-pyenv/bin/activate"
fi

# Clean previous build artifacts
echo -e "${BLUE}🧹 Cleaning previous build artifacts...${NC}"
rm -rf build/ dist/ *.egg-info/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Check README renders correctly
echo -e "\n${BLUE}📄 Validating README...${NC}"
python -c "
import sys
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    if len(content) < 100:
        print('❌ README.md seems too short')
        sys.exit(1)
    print('✅ README.md is valid')
except Exception as e:
    print(f'❌ README.md validation failed: {e}')
    sys.exit(1)
"

# Build source distribution and wheel
echo -e "\n${BLUE}🔨 Building package...${NC}"
echo "Building source distribution and wheel..."
python -m build --sdist --wheel

# Check that build artifacts were created
if [ ! -d "dist" ] || [ -z "$(ls -A dist/)" ]; then
    echo -e "${RED}❌ Build failed - no artifacts in dist/${NC}"
    exit 1
fi

# List built artifacts
echo -e "\n${GREEN}📦 Built artifacts:${NC}"
ls -la dist/

# Validate package metadata and distribution
echo -e "\n${BLUE}🔍 Validating package with twine...${NC}"
twine check dist/*

# Test package installation in a temporary environment
echo -e "\n${BLUE}🧪 Testing package installation...${NC}"

# Create a temporary directory for testing installation
TEMP_DIR=$(mktemp -d)
echo "Testing installation in: $TEMP_DIR"

# Find the wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n1)
TARBALL_FILE=$(ls dist/*.tar.gz | head -n1)

if [ -z "$WHEEL_FILE" ]; then
    echo -e "${RED}❌ No wheel file found in dist/${NC}"
    exit 1
fi

# Test wheel installation
echo "Testing wheel installation: $WHEEL_FILE"
pip install "$WHEEL_FILE" --force-reinstall --no-deps --target "$TEMP_DIR" --quiet

# Test that the package can be imported from the installed location
echo "Testing package import..."
PYTHONPATH="$TEMP_DIR" python -c "
import sys
sys.path.insert(0, '$TEMP_DIR')

try:
    import hidden_regime
    print(f'✅ Package imported successfully')
    print(f'   Version: {hidden_regime.__version__}')
    print(f'   Author: {hidden_regime.__author__}')
    
    # Test basic functionality (v2.0.0 uses factory functions)
    from hidden_regime import create_financial_pipeline, Pipeline
    print('✅ Core factory functions imported successfully')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Import test failed: {e}')
    sys.exit(1)
"

# Clean up temporary directory
rm -rf "$TEMP_DIR"

# Check package size
echo -e "\n${BLUE}📊 Package size analysis:${NC}"
du -h dist/*

# Verify package contents
echo -e "\n${BLUE}📋 Package contents:${NC}"
if command -v unzip >/dev/null 2>&1; then
    echo "Wheel contents:"
    unzip -l "$WHEEL_FILE" | head -20
fi

echo -e "\n${GREEN}✅ Package build and validation completed successfully!${NC}"
echo -e "📦 Artifacts ready in dist/:"
echo -e "  • Source: ${BLUE}$(basename "$TARBALL_FILE")${NC}"
echo -e "  • Wheel:  ${BLUE}$(basename "$WHEEL_FILE")${NC}"
echo -e "\n💡 To publish to PyPI:"
echo -e "  twine upload dist/*"