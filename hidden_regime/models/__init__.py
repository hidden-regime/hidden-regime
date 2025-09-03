"""
Hidden Markov Model implementation for market regime detection.

This module provides the core HMM functionality for detecting and analyzing
market regimes using sophisticated statistical modeling.
"""

from .base_hmm import HiddenMarkovModel
from .config import HMMConfig

__all__ = [
    'HiddenMarkovModel',
    'HMMConfig',
]