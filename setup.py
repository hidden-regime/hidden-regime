'''Package setup'''
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hidden-regime",
    version="0.0.1",
    author="aoaustin",
    author_email="contact@hiddenregime.com", 
    description="Market regime detection using Hidden Markov Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hidden-regime/hidden-regime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
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
        "numpy",
        "pandas", 
        "scipy",
        "matplotlib",
    ],
    keywords="finance trading regimes hmm bayesian machine-learning",
)
