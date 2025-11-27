"""
Base algorithm class for QuantConnect LEAN integration.

This module provides HiddenRegimeAlgorithm, a base class that extends
QuantConnect's QCAlgorithm with integrated regime detection capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Any

import pandas as pd

try:
    # Try to import QuantConnect libraries (when running in LEAN)
    from AlgorithmImports import *  # noqa: F401, F403
    QC_AVAILABLE = True
except ImportError:
    # Mock QCAlgorithm for development/testing
    QC_AVAILABLE = False

    class QCAlgorithm:  # type: ignore
        """Mock QCAlgorithm for testing."""
        def Log(self, message: str) -> None:
            print(f"[LOG] {message}")

        def Debug(self, message: str) -> None:
            print(f"[DEBUG] {message}")

        def SetWarmUp(self, period: Any) -> None:
            pass


from .config import QuantConnectConfig, RegimeTradingConfig
from .data_adapter import QuantConnectDataAdapter
from .signal_adapter import RegimeSignalAdapter, TradingSignal
from .logging import RegimeDetectionLogger


class HiddenRegimeAlgorithm(QCAlgorithm):  # type: ignore
    """
    Base algorithm class with integrated regime detection.

    This class extends QuantConnect's QCAlgorithm with hidden-regime's
    market regime detection capabilities. It handles data collection,
    regime updates, and signal generation automatically.

    Example:
        >>> class MyStrategy(HiddenRegimeAlgorithm):
        ...     def Initialize(self):
        ...         self.SetStartDate(2020, 1, 1)
        ...         self.SetCash(100000)
        ...         self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        ...         self.initialize_regime_detection("SPY", n_states=3)
        ...
        ...     def OnData(self, data):
        ...         if not self.regime_is_ready():
        ...             return
        ...         self.update_regime()
        ...         if self.current_regime == "Bull":
        ...             self.SetHoldings(self.symbol, 1.0)
        ...         elif self.current_regime == "Bear":
        ...             self.Liquidate()
    """

    def __init__(self) -> None:
        """Initialize the algorithm with regime detection components."""
        super().__init__()

        # Regime detection components
        self._regime_pipelines: Dict[str, Any] = {}
        self._data_adapters: Dict[str, QuantConnectDataAdapter] = {}
        self._signal_adapters: Dict[str, RegimeSignalAdapter] = {}

        # Current regime state
        self._current_signals: Dict[str, TradingSignal] = {}
        self.current_regime: Optional[str] = None
        self.regime_confidence: float = 0.0
        self.regime_state: int = -1

        # Configuration
        self._qc_config: Optional[QuantConnectConfig] = None
        self._trading_config: Optional[RegimeTradingConfig] = None

        # Caching and retraining
        self._last_retrain: Dict[str, datetime] = {}
        self._retrain_enabled: bool = True

        # Comprehensive logging
        self.logger = RegimeDetectionLogger(self)

    def initialize_regime_detection(
        self,
        ticker: str,
        n_states: int = 3,
        lookback_days: int = 252,
        retrain_frequency: str = "weekly",
        regime_allocations: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.0,
        **pipeline_kwargs,
    ) -> None:
        """
        Initialize regime detection for a ticker.

        This method sets up the regime detection pipeline, data adapter,
        and signal generator for the specified asset.

        Args:
            ticker: Ticker symbol (e.g., "SPY")
            n_states: Number of regime states (2-5)
            lookback_days: Historical data window size
            retrain_frequency: How often to retrain ('daily', 'weekly', 'monthly', 'never')
            regime_allocations: Dict mapping regimes to allocations
            min_confidence: Minimum confidence for signals
            **pipeline_kwargs: Additional arguments for pipeline creation

        Example:
            >>> self.initialize_regime_detection(
            ...     ticker="SPY",
            ...     n_states=3,
            ...     lookback_days=252,
            ...     retrain_frequency="weekly"
            ... )
        """
        # Import here to avoid circular dependency
        import hidden_regime as hr

        # Create QC configuration
        self._qc_config = QuantConnectConfig(
            lookback_days=lookback_days,
            retrain_frequency=retrain_frequency,
            min_confidence=min_confidence,
        )

        # Create trading configuration
        if regime_allocations:
            self._trading_config = RegimeTradingConfig(
                regime_allocations=regime_allocations
            )
        else:
            self._trading_config = RegimeTradingConfig()

        # Create data adapter
        self._data_adapters[ticker] = QuantConnectDataAdapter(
            lookback_days=lookback_days
        )

        # Create signal adapter
        self._signal_adapters[ticker] = RegimeSignalAdapter(
            regime_allocations=self._trading_config.regime_allocations,
            min_confidence=min_confidence,
        )

        # Create regime detection pipeline
        # Note: Pipeline will be created when we have enough data
        self._regime_pipelines[ticker] = {
            "pipeline": None,
            "ticker": ticker,
            "n_states": n_states,
            "pipeline_kwargs": pipeline_kwargs,
        }

        # Set warm-up period
        warmup_period = timedelta(days=lookback_days)
        self.SetWarmUp(warmup_period)

        # Log initialization
        self.logger.log_regime_setup(
            ticker=ticker,
            n_states=n_states,
            lookback_days=lookback_days,
            retrain_frequency=retrain_frequency,
            allocations=regime_allocations or self._trading_config.regime_allocations,
        )

        self.Debug(
            f"Initialized regime detection for {ticker}: "
            f"{n_states} states, {lookback_days} day lookback"
        )

    def on_tradebar(self, ticker: str, bar: Any) -> None:
        """
        Update regime detection with new price bar.

        Call this method from OnData() to feed price data to regime detector.

        Args:
            ticker: Ticker symbol
            bar: TradeBar object from QuantConnect
        """
        if ticker not in self._data_adapters:
            return

        adapter = self._data_adapters[ticker]

        # Add bar to adapter (handle both QC TradeBars and test dicts)
        if bar is None:
            # Skip if no bar data
            return

        bar_time = None
        bar_close = None

        if hasattr(bar, "Time"):
            bar_time = bar.Time
            bar_close = bar.Close
            adapter.add_bar(
                time=bar.Time,
                open_price=bar.Open,
                high=bar.High,
                low=bar.Low,
                close=bar.Close,
                volume=bar.Volume,
            )
        elif isinstance(bar, dict):
            # For testing with dict objects
            bar_time = bar.get("Time", datetime.now())
            bar_close = bar.get("Close", 0)
            adapter.add_bar(
                time=bar_time,
                open_price=bar.get("Open", 0),
                high=bar.get("High", 0),
                low=bar.get("Low", 0),
                close=bar_close,
                volume=bar.get("Volume", 0),
            )
        else:
            self.Debug(f"Warning: Unexpected bar type: {type(bar)}")
            return

        # Log bar received
        if bar_time and bar_close is not None:
            self.logger.log_bar_received(ticker, bar_time, bar_close)

        # Check readiness after receiving bar
        if adapter.is_ready() and ticker in self._regime_pipelines:
            if self._regime_pipelines[ticker]["pipeline"] is None:
                self.logger.log_regime_readiness(
                    ticker, len(adapter), ready=True
                )

    def update_regime(self, ticker: Optional[str] = None) -> bool:
        """
        Update regime detection for ticker(s).

        This method:
        1. Converts buffered data to DataFrame
        2. Trains/updates HMM if needed
        3. Generates regime signal
        4. Updates current_regime attributes
        5. Calls on_regime_change if regime changed

        Args:
            ticker: Specific ticker to update (None = update all)

        Returns:
            True if update successful, False otherwise
        """
        import hidden_regime as hr

        tickers_to_update = (
            [ticker] if ticker else list(self._regime_pipelines.keys())
        )

        success = True
        for tick in tickers_to_update:
            if tick not in self._data_adapters:
                continue

            adapter = self._data_adapters[tick]

            # Check if we have enough data
            if not adapter.is_ready(min_bars=30):
                continue

            # Get DataFrame
            try:
                df = adapter.to_dataframe()
            except ValueError:
                continue

            # Check if pipeline needs to be created or retrained
            pipeline_info = self._regime_pipelines[tick]
            pipeline = pipeline_info["pipeline"]

            if pipeline is None or self._should_retrain(tick):
                # Log training event
                reason = "initial" if pipeline is None else "scheduled retrain"
                self.logger.log_pipeline_training(tick, datetime.now(), reason=reason)

                # Create/recreate pipeline
                pipeline = hr.create_financial_pipeline(
                    ticker=tick,
                    n_states=pipeline_info["n_states"],
                    **pipeline_info.get("pipeline_kwargs", {}),
                )
                pipeline_info["pipeline"] = pipeline
                self._last_retrain[tick] = datetime.now()

                self.Debug(f"Created/retrained regime pipeline for {tick}")

            # Update pipeline with data (inject data directly)
            try:
                # Inject data into pipeline's data component
                pipeline.data._data = df

                # Run pipeline update
                pipeline.update()

                # Get interpreter output (DataFrame with regime information)
                interpreter_output = pipeline.component_outputs.get("interpreter")
                if interpreter_output is None or interpreter_output.empty:
                    raise ValueError("No regime interpretation available")

                # Generate signal
                signal_adapter = self._signal_adapters[tick]
                signal = signal_adapter.from_pipeline_result(interpreter_output)

                # Log inference result
                self.logger.log_pipeline_inference(
                    ticker=tick,
                    timestamp=datetime.now(),
                    regime=signal.regime_name,
                    state=signal.regime_state,
                    confidence=signal.confidence,
                )

                # Log signal generation
                self.logger.log_signal_generation(
                    ticker=tick,
                    timestamp=datetime.now(),
                    regime=signal.regime_name,
                    allocation=signal.allocation,
                    direction=signal.direction,
                    strength=signal.strength,
                    confidence=signal.confidence,
                )

                # Store signal
                self._current_signals[tick] = signal

                # Update current regime attributes (for single-ticker case)
                if ticker or len(tickers_to_update) == 1:
                    old_regime = self.current_regime
                    self.current_regime = signal.regime_name
                    self.regime_confidence = signal.confidence
                    self.regime_state = signal.regime_state

                    # Call regime change handler
                    if old_regime != self.current_regime and old_regime is not None:
                        self.logger.log_regime_change(
                            ticker=tick,
                            timestamp=datetime.now(),
                            old_regime=old_regime,
                            new_regime=self.current_regime,
                            confidence=self.regime_confidence,
                        )
                        self.on_regime_change(
                            old_regime=old_regime,
                            new_regime=self.current_regime,
                            confidence=self.regime_confidence,
                            ticker=tick,
                        )

            except Exception as e:
                self.Debug(f"Error updating regime for {tick}: {str(e)}")
                success = False

        return success

    def on_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float,
        ticker: str,
    ) -> None:
        """
        Called when regime changes.

        Override this method to implement custom logic on regime transitions.

        Args:
            old_regime: Previous regime name
            new_regime: New regime name
            confidence: Confidence in new regime (0.0 to 1.0)
            ticker: Ticker symbol

        Example:
            >>> def on_regime_change(self, old_regime, new_regime, confidence, ticker):
            ...     self.Log(f"{ticker}: {old_regime} → {new_regime} ({confidence:.1%})")
            ...     if new_regime == "Bear":
            ...         self.Liquidate()
        """
        # Log position update if we're trading
        signal = self.get_regime_signal(ticker)
        if signal and self._qc_config and self._qc_config.log_regime_changes:
            old_allocation = self.Portfolio.Positions.get(ticker, None)
            old_alloc_pct = (
                old_allocation.Percentage if old_allocation else 0.0
            )
            self.logger.log_position_update(
                ticker=ticker,
                timestamp=datetime.now(),
                old_allocation=old_alloc_pct,
                new_allocation=signal.allocation,
                portfolio_value=self.Portfolio.TotalPortfolioValue,
            )

        if self._qc_config and self._qc_config.log_regime_changes:
            self.Log(
                f"Regime change [{ticker}]: {old_regime} → {new_regime} "
                f"(confidence: {confidence:.1%})"
            )

    def get_regime_signal(self, ticker: str) -> Optional[TradingSignal]:
        """
        Get current regime signal for ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            TradingSignal or None if not available
        """
        return self._current_signals.get(ticker)

    def get_regime_allocation(self, ticker: str) -> float:
        """
        Get recommended allocation for ticker based on regime.

        Args:
            ticker: Ticker symbol

        Returns:
            Allocation value (0.0 to 1.0 or negative for shorts)
        """
        signal = self.get_regime_signal(ticker)
        return signal.allocation if signal else 0.0

    def regime_is_ready(self, ticker: Optional[str] = None) -> bool:
        """
        Check if regime detection is ready for trading.

        Args:
            ticker: Specific ticker to check (None = check all)

        Returns:
            True if regime detection has sufficient data
        """
        if ticker:
            adapter = self._data_adapters.get(ticker)
            return adapter.is_ready() if adapter else False
        else:
            # All adapters must be ready
            return all(
                adapter.is_ready() for adapter in self._data_adapters.values()
            )

    def _should_retrain(self, ticker: str) -> bool:
        """
        Check if pipeline should be retrained.

        Args:
            ticker: Ticker symbol

        Returns:
            True if retraining is needed
        """
        if not self._retrain_enabled or not self._qc_config:
            return False

        if ticker not in self._last_retrain:
            return True

        last_train = self._last_retrain[ticker]
        days_since = (datetime.now() - last_train).days

        frequency = self._qc_config.retrain_frequency
        if frequency == "never":
            return False
        elif frequency == "daily":
            return days_since >= 1
        elif frequency == "weekly":
            return days_since >= 7
        elif frequency == "monthly":
            return days_since >= 30

        return False
