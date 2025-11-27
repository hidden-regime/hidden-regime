"""
Alpha model for QuantConnect Framework integration.

This module provides an Alpha Model that generates insights based on
hidden-regime's market regime detection.
"""

from datetime import timedelta
from typing import List, Optional

try:
    from AlgorithmImports import (
        AlphaModel,
        Insight,
        InsightType,
        InsightDirection,
        Resolution,
    )
    QC_AVAILABLE = True
except ImportError:
    # Mocks for testing
    QC_AVAILABLE = False

    class AlphaModel:  # type: ignore
        """Mock AlphaModel."""
        pass

    class Insight:  # type: ignore
        """Mock Insight."""
        pass

    class InsightType:  # type: ignore
        """Mock InsightType."""
        Price = "Price"

    class InsightDirection:  # type: ignore
        """Mock InsightDirection."""
        Up = 1
        Flat = 0
        Down = -1


class HiddenRegimeAlphaModel(AlphaModel):  # type: ignore
    """
    Alpha model using hidden-regime for generating trading insights.

    This alpha model generates Insights based on detected market regimes:
    - Bull regime → Up insight
    - Bear regime → Down insight
    - Sideways → Flat insight
    - Crisis → Down insight (strong)

    Example:
        >>> algorithm = QCAlgorithm()
        >>> algorithm.SetAlpha(HiddenRegimeAlphaModel(
        ...     n_states=3,
        ...     lookback_days=252,
        ...     confidence_threshold=0.7
        ... ))
    """

    def __init__(
        self,
        n_states: int = 3,
        lookback_days: int = 252,
        confidence_threshold: float = 0.6,
        insight_period_days: int = 5,
        resolution: any = None,
    ):
        """
        Initialize Hidden-Regime alpha model.

        Args:
            n_states: Number of HMM states
            lookback_days: Historical data window
            confidence_threshold: Minimum confidence for insights
            insight_period_days: How long insights are valid
            resolution: Data resolution (defaults to Daily if not specified)
        """
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.confidence_threshold = confidence_threshold
        self.insight_period = timedelta(days=insight_period_days)

        if QC_AVAILABLE and resolution is None:
            from AlgorithmImports import Resolution
            self.resolution = Resolution.Daily
        else:
            self.resolution = resolution

        # Store regime detection components per symbol
        self._data_adapters = {}
        self._regime_pipelines = {}
        self._last_regimes = {}

    def Update(self, algorithm: any, data: any) -> List[any]:
        """
        Generate insights based on regime detection.

        Args:
            algorithm: The algorithm instance
            data: The data slice

        Returns:
            List of Insight objects
        """
        insights = []

        # For each symbol in the universe
        for symbol in algorithm.ActiveSecurities.Keys:
            if symbol not in data or data[symbol] is None:
                continue

            # Initialize regime detection for new symbols
            if symbol not in self._data_adapters:
                self._initialize_symbol(symbol)

            # Update data adapter
            bar = data[symbol]
            self._update_data(symbol, bar)

            # Generate regime insight
            insight = self._generate_insight(algorithm, symbol)
            if insight:
                insights.append(insight)

        return insights

    def _initialize_symbol(self, symbol: any) -> None:
        """
        Initialize regime detection for a symbol.

        Args:
            symbol: The symbol to initialize
        """
        from .data_adapter import QuantConnectDataAdapter

        self._data_adapters[symbol] = QuantConnectDataAdapter(
            lookback_days=self.lookback_days
        )
        self._regime_pipelines[symbol] = None
        self._last_regimes[symbol] = None

    def _update_data(self, symbol: any, bar: any) -> None:
        """
        Update data adapter with new bar.

        Args:
            symbol: The symbol
            bar: The price bar
        """
        adapter = self._data_adapters[symbol]

        if hasattr(bar, "Time"):
            adapter.add_bar(
                time=bar.Time,
                open_price=bar.Open,
                high=bar.High,
                low=bar.Low,
                close=bar.Close,
                volume=bar.Volume,
            )

    def _generate_insight(self, algorithm: any, symbol: any) -> Optional[any]:
        """
        Generate insight for symbol based on regime.

        Args:
            algorithm: The algorithm instance
            symbol: The symbol

        Returns:
            Insight object or None
        """
        adapter = self._data_adapters[symbol]

        # Need sufficient data
        if not adapter.is_ready():
            return None

        # Get DataFrame
        try:
            df = adapter.to_dataframe()
        except ValueError:
            return None

        # Create/update pipeline
        if self._regime_pipelines[symbol] is None:
            import hidden_regime as hr

            pipeline = hr.create_financial_pipeline(
                ticker=str(symbol),
                n_states=self.n_states,
            )
            self._regime_pipelines[symbol] = pipeline

        # Update regime
        pipeline = self._regime_pipelines[symbol]
        pipeline.data._data = df

        try:
            result = pipeline.update()
        except Exception:
            return None

        # Extract regime info
        latest = result.iloc[-1]
        regime_name = latest.get("regime_name", "Unknown")
        confidence = latest.get("confidence", 0.0)

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Only generate insight if regime changed
        if self._last_regimes[symbol] == regime_name:
            return None

        self._last_regimes[symbol] = regime_name

        # Map regime to insight direction
        direction = self._regime_to_direction(regime_name)

        if direction is None:
            return None

        # Create insight
        if QC_AVAILABLE:
            return Insight.Price(
                symbol,
                self.insight_period,
                direction,
                None,  # magnitude
                confidence,
            )
        else:
            # Mock for testing
            return {
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "regime": regime_name,
            }

    def _regime_to_direction(self, regime_name: str) -> Optional[any]:
        """
        Convert regime name to insight direction.

        Args:
            regime_name: Regime name (Bull, Bear, etc.)

        Returns:
            InsightDirection or None
        """
        if QC_AVAILABLE:
            mapping = {
                "Bull": InsightDirection.Up,
                "Bear": InsightDirection.Down,
                "Sideways": InsightDirection.Flat,
                "Crisis": InsightDirection.Down,
                "High": InsightDirection.Up,
                "Low": InsightDirection.Down,
                "Medium": InsightDirection.Flat,
            }
        else:
            # Mock directions
            mapping = {
                "Bull": 1,
                "Bear": -1,
                "Sideways": 0,
                "Crisis": -1,
                "High": 1,
                "Low": -1,
                "Medium": 0,
            }

        return mapping.get(regime_name)

    def OnSecuritiesChanged(self, algorithm: any, changes: any) -> None:
        """
        Handle universe changes.

        Args:
            algorithm: The algorithm instance
            changes: SecurityChanges object
        """
        # Clean up removed securities
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            self._data_adapters.pop(symbol, None)
            self._regime_pipelines.pop(symbol, None)
            self._last_regimes.pop(symbol, None)
