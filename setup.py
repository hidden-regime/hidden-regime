'''Package setup for hidden-regime'''
import os
import sys
from setuptools import setup, find_packages

# Ensure we can import from the package directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hidden_regime'))

# Import version information
try:
    from _version import (
        __version__, __title__, __description__, __author__, 
        __author_email__, __url__, __license__
    )
except ImportError:
    # Fallback if _version.py is not available
    __version__ = "0.1.0"
    __title__ = "hidden-regime"
    __description__ = "Market regime detection using Hidden Markov Models with Bayesian uncertainty quantification"
    __author__ = "aoaustin"
    __author_email__ = "contact@hiddenregime.com"
    __url__ = "https://github.com/hidden-regime/hidden-regime"
    __license__ = "MIT"

# Read long description from README
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    try:
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return __description__

# Read requirements from requirements files if they exist
def read_requirements(filename):
    """Read requirements from a file."""
    req_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name=__title__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=__url__,
    project_urls={
        "Homepage": "https://hiddenregime.com",
        "Documentation": "https://docs.hiddenregime.com",
        "Source Code": "https://github.com/hidden-regime/hidden-regime",
        "Bug Reports": "https://github.com/hidden-regime/hidden-regime/issues",
        "Feature Requests": "https://github.com/hidden-regime/hidden-regime/issues",
        "Changelog": "https://github.com/hidden-regime/hidden-regime/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/hidden-regime",
        "Download": "https://pypi.org/project/hidden-regime/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence", 
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
        "Environment :: Console",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
            "bandit[toml]>=1.7.0",
            "tox>=4.0.0",
            "build>=0.8.0",
            "twine>=4.0.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0", 
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "all": [
            "scikit-learn>=1.0.0",  # ml
            "sphinx>=4.0.0",        # docs
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hidden-regime=hidden_regime.cli:main",
        ],
    },
    keywords=[
        "finance", "trading", "regimes", "hmm", "hidden-markov-models",
        "bayesian", "machine-learning", "market-data", "quantitative-finance",
        "algorithmic-trading", "financial-analysis", "time-series", 
        "regime-detection", "volatility", "risk-management"
    ],
    include_package_data=True,
    package_data={
        "hidden_regime": ["py.typed", "*.pyi"],
    },
    zip_safe=False,  # Allow access to package data
    platforms=["any"],
    license=__license__,
)
