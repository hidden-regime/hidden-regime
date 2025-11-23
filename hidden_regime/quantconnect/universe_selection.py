"""
Universe selection models using regime detection.

This module provides universe selection strategies based on market regimes,
allowing algorithms to dynamically select securities based on regime strength.
"""

from typing import List

try:
    from AlgorithmImports import (
        QCAlgorithm,
        Symbol,
        UniverseSelectionModel,
        CoarseFundamental,
    )
    QC_AVAILABLE = True
except ImportError:
    # Mocks for testing
    QC_AVAILABLE = False

    class UniverseSelectionModel:  # type: ignore
        """Mock UniverseSelectionModel."""
        pass


class RegimeBasedUniverseSelection(UniverseSelectionModel):  # type: ignore
    """
    Select securities based on favorable regime characteristics.

    This universe model:
    1. Evaluates regime for candidate securities
    2. Selects securities in favorable regimes (Bull, High)
    3. Filters by regime confidence
    4. Limits universe size

    Example:
        >>> algorithm.SetUniverseSelection(
        ...     RegimeBasedUniverseSelection(
        ...         favorable_regimes=["Bull", "High"],
        ...         min_confidence=0.7,
        ...         max_securities=10
        ...     )
        ... )
    """

    def __init__(
        self,
        favorable_regimes: List[str] = None,
        min_confidence: float = 0.6,
        max_securities: int = 10,
        n_states: int = 3,
    ):
        """
        Initialize regime-based universe selection.

        Args:
            favorable_regimes: List of regime names to select
            min_confidence: Minimum regime confidence threshold
            max_securities: Maximum number of securities to hold
            n_states: Number of HMM states for regime detection
        """
        self.favorable_regimes = favorable_regimes or ["Bull", "High"]
        self.min_confidence = min_confidence
        self.max_securities = max_securities
        self.n_states = n_states

        # Track regime state per symbol
        self._regime_data = {}

    def SelectCoarse(
        self, algorithm: any, coarse: List[any]
    ) -> List[any]:
        """
        Coarse universe selection.

        Args:
            algorithm: QCAlgorithm instance
            coarse: List of CoarseFundamental objects

        Returns:
            List of selected symbols
        """
        # For now, implement simple filter
        # In production, would evaluate regime for each security

        # Filter by basic criteria
        filtered = [
            x
            for x in coarse
            if x.HasFundamentalData and x.Price > 5 and x.DollarVolume > 1e7
        ]

        # Sort by dollar volume
        sorted_by_volume = sorted(
            filtered, key=lambda x: x.DollarVolume, reverse=True
        )

        # Take top N
        selected = sorted_by_volume[: self.max_securities]

        return [x.Symbol for x in selected]


class MultiRegimeUniverseSelection(UniverseSelectionModel):  # type: ignore
    """
    Advanced universe selection supporting multiple regime strategies.

    This model maintains separate universes for different regime types:
    - Bull regime universe: High-beta, growth stocks
    - Bear regime universe: Defensive, low-volatility stocks
    - Crisis regime universe: Safe havens (bonds, gold)
    """

    def __init__(
        self,
        universes: dict = None,
        rebalance_frequency: str = "weekly",
    ):
        """
        Initialize multi-regime universe selection.

        Args:
            universes: Dict mapping regimes to symbol lists
            rebalance_frequency: How often to rebalance
        """
        self.universes = universes or {
            "Bull": ["SPY", "QQQ", "IWM"],
            "Bear": ["TLT", "GLD", "SHY"],
            "Crisis": ["TLT", "GLD", "UUP"],
        }
        self.rebalance_frequency = rebalance_frequency
        self._current_regime = "Bull"

    def SelectCoarse(
        self, algorithm: any, coarse: List[any]
    ) -> List[any]:
        """
        Select universe based on current market regime.

        Args:
            algorithm: QCAlgorithm instance
            coarse: List of CoarseFundamental objects

        Returns:
            List of symbols for current regime
        """
        # Get current market regime (would integrate with regime detector)
        regime = self._current_regime

        # Get symbols for this regime
        regime_symbols = self.universes.get(regime, [])

        # Filter coarse to only include regime symbols
        return [
            x.Symbol
            for x in coarse
            if str(x.Symbol) in regime_symbols and x.Price > 0
        ]

    def update_regime(self, new_regime: str) -> None:
        """
        Update current market regime.

        Args:
            new_regime: New regime name
        """
        if new_regime in self.universes:
            self._current_regime = new_regime
