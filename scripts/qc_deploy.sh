#!/bin/bash
# QuantConnect Deployment Script
# Usage: ./scripts/qc_deploy.sh [template_name]
# Example: ./scripts/qc_deploy.sh basic_regime_switching

set -e

# Configuration
TEMPLATE_NAME="${1:-basic_regime_switching}"
OUTPUT_DIR="qc_deploy"
TEMPLATE_FILE="quantconnect_templates/${TEMPLATE_NAME}.py"

# Validate template exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template not found: $TEMPLATE_FILE"
    echo ""
    echo "Available templates:"
    echo "  - basic_regime_switching"
    echo "  - advanced_crisis_hedging"
    echo "  - market_cycle_detection_bubble_fading"
    echo "  - market_cycle_detection_momentum_riding"
    exit 1
fi

# Clean previous deployment
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous deployment..."
    rm -rf "$OUTPUT_DIR"
fi

# Create output directory
echo "Creating deployment package..."
mkdir -p "$OUTPUT_DIR"

# Copy hidden_regime library (Python files only)
echo "Copying hidden_regime library (Python files only)..."
mkdir -p "$OUTPUT_DIR/hidden_regime"

# Copy all .py files from hidden_regime, preserving directory structure
find hidden_regime -name "*.py" -type f | while read file; do
    dir=$(dirname "$OUTPUT_DIR/$file")
    mkdir -p "$dir"
    cp "$file" "$OUTPUT_DIR/$file"
done

# Copy template as main.py
echo "Copying template: $TEMPLATE_NAME"
cp "$TEMPLATE_FILE" "$OUTPUT_DIR/main.py"

# Verify structure
echo ""
echo "Deployment package created successfully!"
echo ""
echo "Python files included:"
find "$OUTPUT_DIR" -type f -name "*.py" | wc -l
echo "files"
echo ""
echo "Sample structure:"
tree -L 2 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type d | head -20

echo ""
echo "Next steps:"
echo "1. Drag and drop the '$OUTPUT_DIR/' folder into QuantConnect web IDE"
echo "2. Click 'Run Backtest'"
echo "3. Check logs for 'HMM initialized' message"
echo ""
echo "Ready to upload to: https://www.quantconnect.com/terminal/"
