# QuantConnect LEAN with Hidden-Regime
#
# This Dockerfile creates a custom LEAN image with hidden-regime pre-installed
# for seamless regime-based algorithmic trading.
#
# Build from project root:
#   docker build -t lean-hidden-regime:latest -f docker/Dockerfile .
#
# Base: Official QuantConnect LEAN image
# Additions: hidden-regime and core dependencies

FROM quantconnect/lean:17389

LABEL maintainer="hidden-regime <contact@hiddenregime.com>"
LABEL description="QuantConnect LEAN with Hidden-Regime market regime detection"
LABEL version="2.0.0"

# Set working directory
WORKDIR /Lean

# Update pip to latest version
RUN pip install --upgrade pip setuptools wheel

# Install hidden-regime dependencies directly
# We install dependencies separately to avoid setuptools-scm issues in Docker
RUN pip install --no-cache-dir \
    numpy>=2.3.2 \
    pandas>=2.3.2 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    seaborn>=0.11.0 \
    tqdm>=4.50.0 \
    yfinance>=0.2.0 \
    ta>=0.10.2 \
    scikit-learn>=1.0.0 \
    plotly>=5.0.0 \
    Pillow>=9.0.0

# Copy hidden-regime module directly (no package build needed)
COPY hidden_regime/ /usr/local/lib/python3.11/site-packages/hidden_regime/
COPY hidden_regime_mcp/ /usr/local/lib/python3.11/site-packages/hidden_regime_mcp/

# Create __init__.py files if missing
RUN touch /usr/local/lib/python3.11/site-packages/hidden_regime/__init__.py && \
    touch /usr/local/lib/python3.11/site-packages/hidden_regime_mcp/__init__.py

# Verify hidden-regime modules are accessible
RUN python -c "import sys; sys.path.insert(0, '/usr/local/lib/python3.11/site-packages'); import hidden_regime; print('✓ Hidden-Regime imported successfully')"

# Verify visualization support
RUN python -c "import plotly; print('✓ Visualization support verified')"

# Create algorithm directory for strategy templates (mounted as volume at runtime)
RUN mkdir -p /Lean/Algorithm.Python

# Set Python path
# Using . ensures we can import modules from current directory
ENV PYTHONPATH="/usr/local/lib/python3.11/site-packages:/Lean:."

# Set working directory back to LEAN for execution
WORKDIR /Lean

# Health check to ensure hidden-regime is working
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import hidden_regime; import plotly" || exit 1

# Expose LEAN debugger port
EXPOSE 5678
