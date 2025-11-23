# QuantConnect LEAN with Hidden-Regime
#
# This Dockerfile creates a custom LEAN image with hidden-regime pre-installed
# for seamless regime-based algorithmic trading.
#
# Base: Official QuantConnect LEAN image
# Additions: hidden-regime and dependencies

FROM quantconnect/lean:latest

LABEL maintainer="hidden-regime <contact@hiddenregime.com>"
LABEL description="QuantConnect LEAN with Hidden-Regime market regime detection"
LABEL version="1.0.0"

# Set working directory
WORKDIR /Lean

# Update pip to latest version
RUN pip install --upgrade pip

# Install hidden-regime and its dependencies
# Pin versions for reproducibility
RUN pip install --no-cache-dir \
    numpy>=2.3.2 \
    pandas>=2.3.2 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    scikit-learn>=1.0.0 \
    yfinance>=0.2.0 \
    ta>=0.10.2

# Install hidden-regime
# For development, use local package
# For production, use: RUN pip install --no-cache-dir hidden-regime
COPY . /tmp/hidden-regime
RUN pip install --no-cache-dir /tmp/hidden-regime && \
    rm -rf /tmp/hidden-regime

# Pre-compile Python modules for faster startup
RUN python -c "import hidden_regime; print('Hidden-Regime imported successfully')"

# Verify installation
RUN python -c "from hidden_regime.quantconnect import HiddenRegimeAlgorithm; print('QC integration verified')"

# Copy template algorithms to Algorithm.Python directory
COPY quantconnect_templates/*.py /Lean/Algorithm.Python/

# Set Python path to include hidden-regime
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.11/site-packages"

# Default command (inherited from base image)
# ENTRYPOINT ["dotnet", "QuantConnect.Lean.Launcher.dll"]

# Expose default LEAN ports
EXPOSE 5678

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import hidden_regime" || exit 1

# Build metadata
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/hidden-regime/hidden-regime" \
      org.label-schema.schema-version="1.0"
