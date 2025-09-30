"""
Model components for hidden-regime pipeline.

Provides model components that implement the ModelComponent interface
for various regime detection algorithms including Hidden Markov Models.
"""

from .hmm import HiddenMarkovModel

__all__ = [
    "HiddenMarkovModel",
]
