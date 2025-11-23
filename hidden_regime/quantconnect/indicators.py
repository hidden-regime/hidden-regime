"""
Custom QuantConnect indicators using Hidden Markov Models.

This module provides custom indicators that can be used in QuantConnect
algorithms to incorporate regime detection into technical analysis.
"""

from typing import Optional

try:
    from AlgorithmImports import PythonIndicator
    QC_AVAILABLE = True
except ImportError:
    # Mock for testing
    QC_AVAILABLE = False

    class PythonIndicator:  # type: ignore
        """Mock PythonIndicator for testing."""

        def __init__(self, name: str):
            self.Name = name
            self.Value = 0.0
            self.IsReady = False

        def Update(self, input_data: any) -> bool:
            return True


class RegimeIndicator(PythonIndicator):  # type: ignore
    """
    Custom indicator that outputs the current market regime as a numeric value.

    Regime mappings:
        - Bull/High: 1.0
        - Sideways/Medium: 0.0
        - Bear/Low: -1.0
        - Crisis: -2.0

    This indicator can be used in QuantConnect algorithms for charting
    and technical analysis.
    """

    def __init__(self, name: str = "Regime", window_size: int = 252):
        """
        Initialize regime indicator.

        Args:
            name: Indicator name
            window_size: Historical window size for regime detection
        """
        super().__init__(name)
        self.window_size = window_size
        self._data_buffer = []
        self._current_regime_value = 0.0
        self._regime_pipeline = None

    def Update(self, input_data: any) -> bool:
        """
        Update indicator with new price data.

        Args:
            input_data: IndicatorDataPoint or TradeBar

        Returns:
            True if update successful
        """
        # Extract price from input
        if hasattr(input_data, "Close"):
            price = float(input_data.Close)
        elif hasattr(input_data, "Value"):
            price = float(input_data.Value)
        else:
            return False

        # Add to buffer
        self._data_buffer.append(
            {
                "Close": price,
                "Time": getattr(input_data, "Time", None),
            }
        )

        # Maintain window size
        if len(self._data_buffer) > self.window_size:
            self._data_buffer = self._data_buffer[-self.window_size :]

        # Need minimum data for regime detection
        if len(self._data_buffer) < 30:
            return False

        # Run regime detection (simplified for indicator)
        try:
            self._update_regime()
            self.Value = self._current_regime_value
            self.IsReady = True
            return True
        except Exception:
            return False

    def _update_regime(self) -> None:
        """Update regime detection (simplified version)."""
        # This is a placeholder - in practice, would use full HMM
        # For now, use simple price momentum as proxy
        if len(self._data_buffer) < 2:
            return

        recent_prices = [b["Close"] for b in self._data_buffer[-20:]]
        recent_return = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Simple regime classification based on returns
        if recent_return > 0.05:  # 5% gain
            self._current_regime_value = 1.0  # Bull
        elif recent_return < -0.05:  # 5% loss
            self._current_regime_value = -1.0  # Bear
        else:
            self._current_regime_value = 0.0  # Sideways


class RegimeConfidenceIndicator(PythonIndicator):  # type: ignore
    """
    Custom indicator that outputs regime detection confidence.

    Values range from 0.0 (low confidence) to 1.0 (high confidence).
    Can be used to filter trades or adjust position sizing.
    """

    def __init__(self, name: str = "RegimeConfidence"):
        """
        Initialize confidence indicator.

        Args:
            name: Indicator name
        """
        super().__init__(name)
        self._confidence = 0.0

    def Update(self, input_data: any) -> bool:
        """
        Update confidence indicator.

        Args:
            input_data: IndicatorDataPoint with confidence value

        Returns:
            True if update successful
        """
        if hasattr(input_data, "Value"):
            self.Value = float(input_data.Value)
            self.IsReady = True
            return True
        return False

    def set_confidence(self, confidence: float) -> None:
        """
        Manually set confidence value.

        Args:
            confidence: Confidence value (0.0 to 1.0)
        """
        self.Value = max(0.0, min(1.0, confidence))
        self.IsReady = True


class RegimeStrengthIndicator(PythonIndicator):  # type: ignore
    """
    Custom indicator measuring regime strength.

    Combines regime persistence and confidence to measure how
    "strong" the current regime is. Higher values indicate stable,
    confident regime detection.
    """

    def __init__(self, name: str = "RegimeStrength", lookback: int = 20):
        """
        Initialize strength indicator.

        Args:
            name: Indicator name
            lookback: Lookback period for stability calculation
        """
        super().__init__(name)
        self.lookback = lookback
        self._regime_history = []

    def Update(self, input_data: any) -> bool:
        """
        Update strength indicator.

        Args:
            input_data: IndicatorDataPoint with regime value

        Returns:
            True if update successful
        """
        if not hasattr(input_data, "Value"):
            return False

        regime_value = float(input_data.Value)
        self._regime_history.append(regime_value)

        # Maintain lookback window
        if len(self._regime_history) > self.lookback:
            self._regime_history = self._regime_history[-self.lookback :]

        # Calculate strength (consistency of regime)
        if len(self._regime_history) < self.lookback:
            return False

        # Count how many recent periods are in same regime
        current_regime = self._regime_history[-1]
        same_regime_count = sum(
            1 for r in self._regime_history if abs(r - current_regime) < 0.1
        )

        # Strength is proportion of consistent regime
        self.Value = same_regime_count / len(self._regime_history)
        self.IsReady = True
        return True


def create_regime_indicator(
    name: str = "Regime", window_size: int = 252
) -> RegimeIndicator:
    """
    Factory function to create regime indicator.

    Args:
        name: Indicator name
        window_size: Historical window size

    Returns:
        RegimeIndicator instance
    """
    return RegimeIndicator(name=name, window_size=window_size)


def create_confidence_indicator(
    name: str = "RegimeConfidence",
) -> RegimeConfidenceIndicator:
    """
    Factory function to create confidence indicator.

    Args:
        name: Indicator name

    Returns:
        RegimeConfidenceIndicator instance
    """
    return RegimeConfidenceIndicator(name=name)


def create_strength_indicator(
    name: str = "RegimeStrength", lookback: int = 20
) -> RegimeStrengthIndicator:
    """
    Factory function to create strength indicator.

    Args:
        name: Indicator name
        lookback: Lookback period

    Returns:
        RegimeStrengthIndicator instance
    """
    return RegimeStrengthIndicator(name=name, lookback=lookback)
