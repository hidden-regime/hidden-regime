"""
QuantConnect LEAN Integration for Hidden-Regime

This package provides seamless integration between hidden-regime's market regime
detection and QuantConnect's LEAN algorithmic trading engine.

Key Components:
    - HiddenRegimeAlgorithm: Base algorithm class with regime detection
    - QuantConnectDataAdapter: Converts QC data to hidden-regime format
    - RegimeSignalAdapter: Converts regime signals to trading instructions
    - HiddenRegimeAlphaModel: QC Framework alpha model integration
    - RegimeIndicator: Custom QC indicators for regime detection

Quick Start:
    >>> from hidden_regime.quantconnect import HiddenRegimeAlgorithm
    >>>
    >>> class MyStrategy(HiddenRegimeAlgorithm):
    ...     def Initialize(self):
    ...         self.SetStartDate(2020, 1, 1)
    ...         self.SetCash(100000)
    ...         self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
    ...         self.initialize_regime_detection("SPY", n_states=3)
    ...
    ...     def OnData(self, data):
    ...         self.update_regime()
    ...         if self.current_regime == "Bull":
    ...             self.SetHoldings(self.symbol, 1.0)

For documentation and examples, visit: https://hiddenregime.com/docs/quantconnect
"""

from .algorithm import HiddenRegimeAlgorithm
from .optimized_algorithm import HiddenRegimeAlgorithmOptimized
from .data_adapter import (
    QuantConnectDataAdapter,
    RollingWindowDataAdapter,
    HistoryDataAdapter,
)
from .signal_adapter import (
    RegimeSignalAdapter,
    TradingSignal,
    MultiAssetSignalAdapter,
    SignalDirection,
    SignalStrength,
)
from .regime_label_adapter import RegimeLabelQuantConnectAdapter
from .config import (
    QuantConnectConfig,
    RegimeTradingConfig,
    MultiAssetRegimeConfig,
)
from .indicators import (
    RegimeIndicator,
    RegimeConfidenceIndicator,
    RegimeStrengthIndicator,
)
from .alpha_model import HiddenRegimeAlphaModel
from .universe_selection import (
    RegimeBasedUniverseSelection,
    MultiRegimeUniverseSelection,
)
from .performance import (
    RegimeModelCache,
    CachedRegimeDetector,
    PerformanceProfiler,
    BatchRegimeUpdater,
)

__all__ = [
    # Core algorithm classes
    "HiddenRegimeAlgorithm",
    "HiddenRegimeAlgorithmOptimized",
    # Data adapters
    "QuantConnectDataAdapter",
    "RollingWindowDataAdapter",
    "HistoryDataAdapter",
    # Signal generation (legacy)
    "RegimeSignalAdapter",
    "TradingSignal",
    "MultiAssetSignalAdapter",
    "SignalDirection",
    "SignalStrength",
    # Signal generation (new - RegimeLabel based)
    "RegimeLabelQuantConnectAdapter",
    # Configuration
    "QuantConnectConfig",
    "RegimeTradingConfig",
    "MultiAssetRegimeConfig",
    # Indicators
    "RegimeIndicator",
    "RegimeConfidenceIndicator",
    "RegimeStrengthIndicator",
    # QC Framework integration
    "HiddenRegimeAlphaModel",
    "RegimeBasedUniverseSelection",
    "MultiRegimeUniverseSelection",
    # Performance optimization (Phase 4)
    "RegimeModelCache",
    "CachedRegimeDetector",
    "PerformanceProfiler",
    "BatchRegimeUpdater",
]

__version__ = "1.0.0"
