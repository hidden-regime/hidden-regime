'''Package setup'''
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hidden-regime",
    version="0.1.0",
    author="aoaustin",
    author_email="contact@hiddenregime.com", 
    description="Market regime detection using Hidden Markov Models with Bayesian uncertainty quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hidden-regime/hidden-regime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
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
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
        ]
    },
    keywords="finance trading regimes hmm bayesian machine-learning market-data",
)
