"""
Basic Regime Switching Strategy

This template demonstrates the simplest regime-based trading strategy using
RegimeType enum for type-safe allocation mappings:

Strategy Logic:
- BULLISH regime: 100% long position
- BEARISH regime: Move to cash
- SIDEWAYS regime: 50% long position
- CRISIS regime: Cash
- MIXED regime: Defensive 25% long position

Why RegimeType Enum?
For 4+ state HMMs, the interpreter discovers regime names dynamically from data
(e.g., "Uptrend", "Flat", "Crash Bear"). Using the stable RegimeType enum ensures
allocations remain consistent even if discovered names change after retraining.

Author: hidden-regime
License: MIT
"""

# Import QuantConnect API
from AlgorithmImports import *

# Import Hidden-Regime components
from hidden_regime.quantconnect import HiddenRegimeAlgorithm
from hidden_regime.quantconnect.config import RegimeTypeAllocations


class BasicRegimeSwitching(HiddenRegimeAlgorithm):
    """
    Simple regime-based strategy for Given Ticker.

    Strategy Logic:
        1. Detect market regime using 3-state HMM
        2. Allocate based on regime:
           - Bull: 100% long
           - Bear: 0% (cash)
           - Sideways: 50% long
        3. Rebalance on regime changes
    """

    def Initialize(self):
        """
        Initialize the algorithm.

        Sets up:
        - Backtest period
        - Initial cash
        - Equity
        - Regime detection with adaptive parameter updating

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (default: "SPY")
        start_date : str
            Start date in format YYYY-MM-DD (default: "2020-01-01")
        end_date : str
            End date in format YYYY-MM-DD (default: "2022-01-01")
        cash : float
            Initial portfolio cash (default: 100000)
        n_states : int
            Number of HMM states (default: 3)
        lookback_days : int
            Historical data window for HMM training (default: 252)
        min_confidence : float
            Minimum confidence threshold for regime signals (default: 0.7)

        bullish_allocation : float
            Portfolio allocation for BULLISH regime (0.0-1.0, default: 1.0)
        bearish_allocation : float
            Portfolio allocation for BEARISH regime (0.0-1.0, default: 0.0)
        sideways_allocation : float
            Portfolio allocation for SIDEWAYS regime (0.0-1.0, default: 0.5)
        crisis_allocation : float
            Portfolio allocation for CRISIS regime (0.0-1.0, default: 0.0)
        mixed_allocation : float
            Portfolio allocation for MIXED regime (0.0-1.0, default: 0.25)

        retrain_frequency : str
            Schedule-based retraining safety net ('daily', 'weekly', 'monthly', 'never')
            Works alongside automatic adaptive re-fitting which continuously updates
            parameters based on drift detection. Default: "never"

            NOTE: Adaptive re-fitting happens automatically when drift is detected.
            retrain_frequency is a safety mechanism to ensure model refresh on a schedule
            regardless of drift signals. Recommended: "weekly" for production trading.

        debug_output_dir : str, optional
            Directory to export debug CSVs (HMM params, state probs, signals, etc.)
            Default: None (auto-detects backtest_results/ or Results/ directory)

            Debug CSVs exported to: {output_dir}/debug_{ticker}/
            Files created:
            - timesteps.csv: One row per bar with all state data
            - state_probabilities.csv: Per-state probability tracking
            - hmm_params.csv: HMM parameters in long format
            - training_history.csv: Model training events
            - regime_changes.csv: Regime transitions only

        Architecture Note (4+ State HMMs):
            For 4+ state models, the interpreter discovers regime names dynamically from data.
            For example, it might discover "Uptrend", "Flat", "Downtrend", "Crisis" instead of
            the standard labels. Using RegimeType enum (BULLISH, BEARISH, SIDEWAYS, CRISIS, MIXED)
            ensures allocations remain stable even if discovered names change after retraining.
        """
        # === EASY PARAMETERS TO MODIFY ===
        ticker = self.GetParameter("ticker", "SPY")
        start_year = int(self.GetParameter("start_year", 2015))
        start_month = int(self.GetParameter("start_month", 1))
        start_day = int(self.GetParameter("start_day", 1))
        end_year = int(self.GetParameter("end_year", 2016))
        end_month = int(self.GetParameter("end_month", 1))
        end_day = int(self.GetParameter("end_day", 1))
        initial_cash = float(self.GetParameter("cash", 100000))

        n_states = int(self.GetParameter("n_states", 4))
        lookback_days = int(self.GetParameter("lookback_days", 252))
        min_confidence = float(self.GetParameter("min_confidence", 0.7))
        random_seed = int(self.GetParameter("random_seed", 4242))

        # Regime allocations using RegimeType enum
        # These map to stable financial regime types independent of discovered names
        regime_allocations = RegimeTypeAllocations(
            bullish=float(self.GetParameter("bullish_allocation", 1.0)),
            bearish=float(self.GetParameter("bearish_allocation", 0.0)),
            sideways=float(self.GetParameter("sideways_allocation", 0.5)),
            crisis=float(self.GetParameter("crisis_allocation", 0.0)),
            mixed=float(self.GetParameter("mixed_allocation", 0.25)),
        )

        retrain_frequency = self.GetParameter("retrain_frequency", "never")

        # Debug output (CSV export of internal state)
        debug_output_dir = self.GetParameter("debug_output_dir", None)

        # === BACKTEST CONFIGURATION ===
        self.SetStartDate(start_year, start_month, start_day)
        self.SetEndDate(end_year, end_month, end_day)
        self.SetCash(initial_cash)

        # Store debug output directory (used at end of backtest)
        if debug_output_dir:
            self._debug_output_dir = debug_output_dir

        # Add equity
        self.symbol = self.AddEquity(ticker, Resolution.Daily).Symbol  # noqa: F405

        # Initialize regime detection with enum-based allocations
        # The system uses TWO mechanisms for parameter updates:
        # 1. Adaptive re-fitting (automatic): Updates parameters in real-time when drift detected
        #    - Emission-only updates (~1% cost) for volatility changes
        #    - Transition-only updates (~5% cost) for persistence changes
        #    - Full retrains (~100% cost) for major structural changes
        # 2. Retraining schedule (safety net): Ensures model refresh regardless of drift signals
        #    - Prevents over-reliance on drift detection alone
        #    - Provides baseline refresh frequency (daily/weekly/monthly/never)
        #
        # RegimeType enum allocations ensure temporal stability: Even if the interpreter
        # discovers different regime names after retraining (e.g., "Uptrend" vs "Bull"),
        # the allocation for BULLISH regime remains constant.
        # See: hidden_regime/quantconnect/config.py for RegimeTypeAllocations
        self.initialize_regime_detection(
            ticker=ticker,
            n_states=n_states,
            lookback_days=lookback_days,
            retrain_frequency=retrain_frequency,
            regime_type_allocations=regime_allocations,
            min_confidence=min_confidence,
            random_seed=random_seed,
        )

        self.Debug(
            f"BasicRegimeSwitching initialized: {ticker} ({start_year}-{start_month}-{start_day} to {end_year}-{end_month}-{end_day})"
        )

    def OnWarmupFinished(self):
        """
        Called when the warmup period completes.

        Fetch historical data from the warmup period and pre-populate
        the regime detector's buffer so training can begin immediately.
        """
        ticker = self.GetParameter("ticker", "SPY")
        lookback_days = int(self.GetParameter("lookback_days", 252))

        # Fetch history from warmup period
        history = self.History(
            self.symbol, lookback_days, Resolution.Daily
        )  # noqa: F405

        if not history.empty:
            # Add each bar to the regime detector
            for index, row in history.iterrows():
                bar = {
                    "Time": index if not isinstance(index, tuple) else index[1],
                    "Open": row["open"],
                    "High": row["high"],
                    "Low": row["low"],
                    "Close": row["close"],
                    "Volume": row["volume"],
                }
                self.on_tradebar(ticker, bar)

            self.Debug(f"Loaded {len(history)} warmup bars for {ticker}")

    def OnData(self, data):
        """
        Handle new market data.

        Args:
            data: Slice object containing market data
        """
        # Skip if no data
        if not data.ContainsKey(self.symbol):
            return

        # Update regime detector with new bar
        bar = data[self.symbol]
        ticker = self.GetParameter("ticker", "SPY")
        self.on_tradebar(ticker, bar)

        # Wait until regime detection is ready
        if not self.regime_is_ready():
            return

        # Update regime
        self.update_regime()

        # Get recommended allocation
        allocation = self.get_regime_allocation(ticker)

        # Trade to target allocation
        self.SetHoldings(self.symbol, allocation)

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """
        Called when regime transitions.

        Args:
            old_regime: Previous regime name
            new_regime: New regime name
            confidence: Confidence in new regime (0.0 to 1.0)
            ticker: Asset ticker
        """
        self.Log(
            f"[{ticker}] Regime Change: {old_regime} → {new_regime} "
            f"(confidence: {confidence:.1%})"
        )

        # Log portfolio value
        portfolio_value = self.Portfolio.TotalPortfolioValue
        self.Log(f"Portfolio Value: ${portfolio_value:,.2f}")


# For QuantConnect cloud/local LEAN deployment, the algorithm class
# is automatically detected and instantiated.
