"""
Model components for hidden-regime pipeline.

Provides model components that implement the ModelComponent interface
for various regime detection algorithms including Hidden Markov Models
and multi-timeframe regime detection.
"""

from .hmm import HiddenMarkovModel
from .multitimeframe import MultiTimeframeRegime

# Re-export HMMConfig from canonical location for backward compatibility
from ..config.model import HMMConfig

__all__ = [
    "HiddenMarkovModel",
    "HMMConfig",
    "MultiTimeframeRegime",
]
