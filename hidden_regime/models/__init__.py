"""
Hidden Markov Model implementation for market regime detection.

This module provides the core HMM functionality for detecting and analyzing
market regimes using sophisticated statistical modeling, including online
learning and real-time streaming capabilities.
"""

from .base_hmm import HiddenMarkovModel
from .config import HMMConfig
from .online_hmm import OnlineHMM, OnlineHMMConfig
from .streaming import (
    StreamingProcessor, StreamingConfig, StreamingMode,
    StreamingObservation, StreamingResult, 
    SimulatedDataSource, StreamingDataSource
)

__all__ = [
    'HiddenMarkovModel',
    'HMMConfig',
    'OnlineHMM',
    'OnlineHMMConfig', 
    'StreamingProcessor',
    'StreamingConfig',
    'StreamingMode',
    'StreamingObservation',
    'StreamingResult',
    'SimulatedDataSource',
    'StreamingDataSource'
]