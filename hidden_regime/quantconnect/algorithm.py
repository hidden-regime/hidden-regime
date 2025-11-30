"""
Base algorithm class for QuantConnect LEAN integration.

This module provides HiddenRegimeAlgorithm, a base class that extends
QuantConnect's QCAlgorithm with integrated regime detection capabilities.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

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


from .config import QuantConnectConfig, RegimeTradingConfig, RegimeTypeAllocations
from .data_adapter import QuantConnectDataAdapter
from .debug import DebugDataAccumulator
from .logging import RegimeDetectionLogger
from .signal_adapter import RegimeSignalAdapter, TradingSignal


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

        # Debug data accumulators (for CSV export)
        self._debug_accumulators: Dict[str, DebugDataAccumulator] = {}
        self._debug_enabled = True  # Enable by default, can be overridden via parameter
        self._debug_output_dir: Optional[str] = (
            None  # Set in Initialize() from parameter
        )

    def initialize_regime_detection(
        self,
        ticker: str,
        n_states: int = 3,
        lookback_days: int = 252,
        retrain_frequency: str = "weekly",
        regime_type_allocations: Optional[RegimeTypeAllocations] = None,
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
            regime_type_allocations: RegimeTypeAllocations mapping RegimeType enum to floats
            min_confidence: Minimum confidence for signals
            **pipeline_kwargs: Additional arguments for pipeline creation

        Raises:
            ValueError: If passed old string-based regime_allocations dict

        Example:
            >>> from hidden_regime.quantconnect.config import RegimeTypeAllocations
            >>> allocations = RegimeTypeAllocations(bullish=1.0, bearish=0.0, sideways=0.5)
            >>> self.initialize_regime_detection(
            ...     ticker="SPY",
            ...     n_states=3,
            ...     lookback_days=252,
            ...     retrain_frequency="weekly",
            ...     regime_type_allocations=allocations
            ... )
        """
        # Reject old string-based allocations (breaking change)
        if "regime_allocations" in pipeline_kwargs:
            raise ValueError(
                "String-based regime_allocations are no longer supported. "
                "Use RegimeTypeAllocations instead:\n"
                "  from hidden_regime.quantconnect.config import RegimeTypeAllocations\n"
                "  allocations = RegimeTypeAllocations(bullish=1.0, bearish=0.0, ...)\n"
                "See migration guide: docs/guides/quantconnect_migration.md"
            )

        # Import here to avoid circular dependency
        import hidden_regime as hr

        # Use default allocations if not specified
        if regime_type_allocations is None:
            regime_type_allocations = RegimeTypeAllocations()

        # Create QC configuration
        self._qc_config = QuantConnectConfig(
            lookback_days=lookback_days,
            retrain_frequency=retrain_frequency,
            min_confidence=min_confidence,
        )

        # Store trading allocations (for logging/reporting)
        self._trading_config = RegimeTradingConfig()

        # Create data adapter
        self._data_adapters[ticker] = QuantConnectDataAdapter(
            lookback_days=lookback_days
        )

        # Create signal adapter with enum-based allocations
        self._signal_adapters[ticker] = RegimeSignalAdapter(
            regime_type_allocations=regime_type_allocations,
            min_confidence=min_confidence,
        )

        # Create debug data accumulator
        if self._debug_enabled:
            self._debug_accumulators[ticker] = DebugDataAccumulator(
                ticker=ticker, n_states=n_states
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
            allocations=regime_type_allocations,
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
                self.logger.log_regime_readiness(ticker, len(adapter), ready=True)

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

        tickers_to_update = [ticker] if ticker else list(self._regime_pipelines.keys())

        success = True
        for tick in tickers_to_update:
            if tick not in self._data_adapters:
                continue

            adapter = self._data_adapters[tick]

            # Check if we have enough data
            if not adapter.is_ready():
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
                self.logger.log_pipeline_training(tick, self.Time, reason=reason)

                # Create/recreate pipeline with safeguard against external data fetch
                # In QuantConnect backtest context, only QuantConnect-provided data should be used
                kwargs = pipeline_info.get("pipeline_kwargs", {})
                kwargs["allow_external_data_fetch"] = False

                pipeline = hr.create_financial_pipeline(
                    ticker=tick,
                    n_states=pipeline_info["n_states"],
                    **kwargs,
                )
                pipeline_info["pipeline"] = pipeline
                # Use backtest time (self.Time) not wall-clock time (datetime.now())
                self._last_retrain[tick] = self.Time

                self.Debug(f"Created/retrained regime pipeline for {tick}")

            # Update pipeline with data (use public interface)
            try:
                # Run pipeline update with new data
                # Pipeline will validate, process through mandatory pipeline, and accumulate
                pipeline.update(data=df)

                # Get interpreter output (DataFrame with regime information)
                interpreter_output = pipeline.component_outputs.get("interpreter")
                if interpreter_output is None or interpreter_output.empty:
                    raise ValueError("No regime interpretation available")

                # Generate signal
                signal_adapter = self._signal_adapters[tick]
                signal = signal_adapter.from_pipeline_result(interpreter_output)

                # Capture debug data (after successful pipeline update)
                if self._debug_enabled and tick in self._debug_accumulators:
                    self._capture_debug_data(tick, df, interpreter_output, signal)

                # Log inference result
                self.logger.log_pipeline_inference(
                    ticker=tick,
                    timestamp=self.Time,
                    regime=signal.regime_name,
                    state=signal.regime_state,
                    confidence=signal.confidence,
                )

                # Log signal generation
                self.logger.log_signal_generation(
                    ticker=tick,
                    timestamp=self.Time,
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
                            timestamp=self.Time,
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
            old_alloc_pct = old_allocation.Percentage if old_allocation else 0.0
            self.logger.log_position_update(
                ticker=ticker,
                timestamp=self.Time,
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

        Applies quick-win improvements from debug analysis:
        1. BEARISH filter: Reduce or skip BEARISH trades (Sharpe: -1.34)
        2. Day 0 transition skip: Reduce size on regime change day (Sharpe: -3.53)
        3. Volatility targeting: Scale position by current volatility

        Args:
            ticker: Ticker symbol

        Returns:
            Allocation value (0.0 to 1.0 or negative for shorts)
        """
        signal = self.get_regime_signal(ticker)
        if not signal:
            return 0.0

        allocation = signal.allocation

        # QUICK WIN 1: Filter BEARISH regime
        # Analysis shows BEARISH has Sharpe of -1.34 (negative)
        # Reduce position size to 25% in BEARISH, or 0 to skip entirely
        if signal.regime_name and "BEARISH" in signal.regime_name.upper():
            allocation *= 0.25  # Reduce to 25% of normal size
            self.Debug(
                f"BEARISH regime detected: reducing allocation from "
                f"{signal.allocation:.2f} to {allocation:.2f}"
            )

        # QUICK WIN 2: Skip Day 0 regime transitions
        # Analysis shows Day 0 transitions have Sharpe of -3.53 (terrible)
        # Wait 2-3 days for regime stability before full position
        if ticker in self._data_adapters:
            adapter = self._data_adapters[ticker]
            try:
                df = adapter.to_dataframe()
                if len(df) > 1:
                    # Check if regime changed between last 2 bars
                    current_regime = df.iloc[-1].get("regime_label")
                    previous_regime = df.iloc[-2].get("regime_label")

                    if current_regime != previous_regime:
                        # This is Day 0 (transition day) - reduce size
                        allocation *= 0.5
                        self.Debug(
                            f"Regime transition detected ({previous_regime} → {current_regime}): "
                            f"reducing allocation to {allocation:.2f} for stability"
                        )
            except Exception:
                pass  # If we can't detect transitions, continue with normal sizing

        # QUICK WIN 3: Volatility targeting
        # Scale position size by current volatility vs target volatility
        # This naturally reduces size during volatile periods (when losses are bigger)
        if ticker in self._data_adapters:
            adapter = self._data_adapters[ticker]
            try:
                df = adapter.to_dataframe()
                if len(df) >= 20:
                    # Calculate 20-day rolling volatility
                    returns = df["Close"].pct_change()
                    current_vol = returns.tail(20).std()

                    # Annualize volatility
                    annualized_vol = current_vol * (252**0.5)

                    # Target volatility (15% annual)
                    target_vol = 0.15

                    # Scale allocation to maintain constant risk
                    if annualized_vol > 0.001:  # Avoid division by zero
                        vol_adjustment = target_vol / annualized_vol
                        allocation *= vol_adjustment
                        self.Debug(
                            f"Volatility targeting: vol={annualized_vol:.1%}, "
                            f"adj={vol_adjustment:.2f}, allocation={allocation:.2f}"
                        )
            except Exception:
                pass  # If we can't calculate volatility, continue with normal sizing

        return allocation

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
            return all(adapter.is_ready() for adapter in self._data_adapters.values())

    def _should_retrain(self, ticker: str) -> bool:
        """
        Check if pipeline should be retrained.

        Uses backtest time (self.Time) not wall-clock time to ensure
        retraining happens at the right frequency during backtesting.

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
        # Use self.Time (backtest time) not datetime.now() (wall-clock time)
        days_since = (self.Time.date() - last_train.date()).days

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

    def _capture_debug_data(
        self,
        ticker: str,
        raw_df: pd.DataFrame,
        interpreter_output: pd.DataFrame,
        signal: TradingSignal,
    ) -> None:
        """
        Capture internal state for debugging and analysis.

        Extracts data from all pipeline stages and accumulates in DebugDataAccumulator.
        Called after each successful pipeline update.

        Args:
            ticker: Asset ticker
            raw_df: Raw OHLCV data from QuantConnectDataAdapter
            interpreter_output: Regime interpretation DataFrame from pipeline
            signal: TradingSignal generated from interpretation
        """
        if ticker not in self._debug_accumulators:
            return

        accumulator = self._debug_accumulators[ticker]
        latest_row = interpreter_output.iloc[-1]
        latest_bar = raw_df.iloc[-1]

        # Get n_states from pipeline info
        n_states = self._regime_pipelines.get(ticker, {}).get("n_states", 3)

        # Extract bar data (OHLCV)
        bar_data = {
            "Open": float(latest_bar.get("Open", 0.0)),
            "High": float(latest_bar.get("High", 0.0)),
            "Low": float(latest_bar.get("Low", 0.0)),
            "Close": float(latest_bar.get("Close", 0.0)),
            "Volume": int(latest_bar.get("Volume", 0)),
        }

        # Extract observation data (log returns, features)
        observation_data = {}
        if "log_return" in latest_row:
            observation_data["log_return"] = float(latest_row["log_return"])

        # Extract model output (HMM predictions, state probs, parameters)
        model_output = {
            "predicted_state": int(latest_row.get("predicted_state", -1)),
            "confidence": float(latest_row.get("confidence", 0.0)),
        }

        # Add per-state probabilities
        for state_idx in range(n_states):
            prob_key = f"state_{state_idx}_prob"
            if prob_key in latest_row:
                model_output[prob_key] = float(latest_row[prob_key])

        # Extract interpreter output (regime labels and metrics)
        interpreter_data = {
            "regime_label": str(latest_row.get("regime_label", "Unknown")),
            "regime_type": str(latest_row.get("regime_type", "unknown")),
            "regime_strength": float(latest_row.get("regime_strength", 0.0)),
        }

        # Add regime characteristics
        char_keys = [
            "mean_return",
            "volatility",
            "win_rate",
            "max_drawdown",
            "sharpe_ratio",
            "expected_return",
            "expected_volatility",
            "avg_regime_duration",
        ]
        for key in char_keys:
            if key in latest_row:
                try:
                    interpreter_data[key] = float(latest_row[key])
                except (ValueError, TypeError):
                    interpreter_data[key] = None

        # Extract signal data
        signal_data = {
            "signal_direction": signal.direction.value if signal else 0,
            "signal_strength": signal.strength.value if signal else 0,
            "signal_allocation": float(signal.allocation) if signal else 0.0,
            "signal_confidence": float(signal.confidence) if signal else 0.0,
        }

        # Add timestep to accumulator
        accumulator.add_timestep(
            timestamp=latest_row.name,  # Index is timestamp
            bar_data=bar_data,
            observation_data=observation_data,
            model_output=model_output,
            interpreter_output=interpreter_data,
            signal_data=signal_data,
        )

        # Add state probabilities
        state_probs = {}
        for state_idx in range(n_states):
            prob_key = f"state_{state_idx}_prob"
            state_probs[state_idx] = latest_row.get(prob_key, 0.0)
        accumulator.add_state_probabilities(
            timestamp=latest_row.name, state_probs=state_probs
        )

        # Record HMM params on first timestep
        if len(accumulator.timesteps) == 1:
            pipeline = self._regime_pipelines[ticker]["pipeline"]
            if pipeline and hasattr(pipeline, "model"):
                model = pipeline.model
                accumulator.record_hmm_params(model)

    def _get_results_directory(self) -> str:
        """
        Detect the QuantConnect backtest results directory.

        Strategies (in order):
        1. Use explicitly provided debug_output_dir parameter
        2. QC's /Lean/Results (when running via QC docker scripts)
        3. Find most recent backtest_results/*/directory
        4. Fall back to other common locations

        Returns:
            Path to results directory where debug CSVs will be written
        """
        # Use explicitly provided directory if set
        if self._debug_output_dir:
            return self._debug_output_dir

        # Check for QC's standard Results directory (mounted during docker backtest)
        if os.path.exists("/Lean/Results") and os.path.isdir("/Lean/Results"):
            return "/Lean/Results"

        # Try to find the MOST RECENT backtest results directory
        # (they have timestamps like basic_regime_switching_20251129_074427)
        backtest_results_dir = None

        # Check parent directories for backtest_results folder
        current = os.getcwd()
        for _ in range(5):  # Try up to 5 levels up
            test_path = os.path.join(current, "backtest_results")
            if os.path.exists(test_path) and os.path.isdir(test_path):
                backtest_results_dir = test_path
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        if backtest_results_dir and os.path.exists(backtest_results_dir):
            # Find the most recently modified subdirectory
            try:
                subdirs = [
                    os.path.join(backtest_results_dir, d)
                    for d in os.listdir(backtest_results_dir)
                    if os.path.isdir(os.path.join(backtest_results_dir, d))
                ]

                if subdirs:
                    # Sort by modification time, get most recent
                    most_recent = max(subdirs, key=os.path.getmtime)
                    return most_recent
            except (OSError, ValueError):
                pass

        # Fallback to other common locations
        possible_dirs = [
            "Results",
            "backtest_results",
            os.path.join(os.getcwd(), "Results"),
            os.path.join(os.getcwd(), "backtest_results"),
        ]

        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                return dir_path

        # Last resort: use current directory
        return os.getcwd()

    def _export_debug_data(self) -> None:
        """
        Export all accumulated debug data to CSV files.

        Called at end of backtest (OnEndOfAlgorithm hook).
        Attempts to write to multiple locations in this order:
        1. Explicitly provided debug_output_dir
        2. Most recent backtest results directory
        3. ObjectStore (QuantConnect's standard storage)
        4. Current working directory
        """
        if not self._debug_accumulators:
            return

        for ticker, accumulator in self._debug_accumulators.items():
            # Try multiple output paths
            output_paths_to_try = []

            # 1. Explicitly provided directory
            if self._debug_output_dir:
                output_paths_to_try.append(self._debug_output_dir)

            # 2. Most recent backtest directory
            try:
                base_dir = self._get_results_directory()
                output_paths_to_try.append(base_dir)
            except Exception:
                pass

            # 3. ObjectStore (QC's native storage)
            try:
                objectstore_path = self.ObjectStore.GetFilePath("")
                if objectstore_path and objectstore_path != "":
                    output_paths_to_try.append(objectstore_path)
            except Exception:
                pass

            # 4. Current working directory
            output_paths_to_try.append(os.getcwd())

            # Try to export to the first accessible directory
            exported = False
            last_error = None

            for base_output_dir in output_paths_to_try:
                try:
                    # Create ticker-specific debug subdirectory
                    output_dir = os.path.join(base_output_dir, f"debug_{ticker}")
                    os.makedirs(output_dir, exist_ok=True)

                    # Try to write a test file to verify directory is writable
                    test_file = os.path.join(output_dir, ".write_test")
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)

                    # Export CSVs
                    csv_files = accumulator.export_to_csv(output_dir)

                    # Log export
                    self.Log(f"DEBUG: {ticker} analysis exported to {output_dir}")
                    for filename, filepath in csv_files.items():
                        self.Log(f"  - {filename}")

                    # Log summary
                    summary = accumulator.summary()
                    for line in summary.split("\n"):
                        self.Log(f"DEBUG: {line}")

                    exported = True
                    break

                except (OSError, IOError, PermissionError) as e:
                    last_error = e
                    continue

            if not exported:
                error_msg = f"Could not export debug data for {ticker}"
                if last_error:
                    error_msg += f": {str(last_error)}"
                self.Log(f"DEBUG: WARNING - {error_msg}")
                self.Log(f"DEBUG: Tried paths: {output_paths_to_try}")

    def OnEndOfAlgorithm(self) -> None:
        """
        QuantConnect hook called at end of backtest.

        Exports accumulated debug data to CSV files.
        """
        if self._debug_enabled:
            self._export_debug_data()
