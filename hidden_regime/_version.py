"""
Version management for hidden-regime package.

This module provides centralized version information that can be imported
by setup.py, __init__.py, and other modules that need version data.
"""

# Version information
__version__ = "0.1.0"
__version_info__ = tuple(map(int, __version__.split(".")))

# Package metadata
__title__ = "hidden-regime"
__description__ = "Market regime detection using Hidden Markov Models with Bayesian uncertainty quantification"
__author__ = "aoaustin"
__author_email__ = "contact@hiddenregime.com"
__url__ = "https://github.com/hidden-regime/hidden-regime"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Hidden Regime"

# Development status
__status__ = "Alpha"  # Alpha, Beta, Production/Stable

# Build information (can be populated by CI/CD)
__build__ = None
__commit__ = None
__timestamp__ = None