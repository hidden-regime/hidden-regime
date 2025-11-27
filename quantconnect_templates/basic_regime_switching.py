"""
Basic Regime Switching Strategy

This template demonstrates the simplest regime-based trading strategy:
- Bull regime: 100% long position
- Bear regime: Move to cash
- Sideways: 50% long position

Author: hidden-regime
License: MIT
"""

# Import QuantConnect API
from AlgorithmImports import *

# Import Hidden-Regime components
from hidden_regime.quantconnect import HiddenRegimeAlgorithm

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
        - Regime detection

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
            Minimum confidence threshold for regime signals (default: 0.6)
        """
        # === EASY PARAMETERS TO MODIFY ===
        ticker = self.GetParameter("ticker", "SPY")
        start_year = int(self.GetParameter("start_year", 2017))
        start_month = int(self.GetParameter("start_month", 1))
        start_day = int(self.GetParameter("start_day", 1))
        end_year = int(self.GetParameter("end_year", 2023))
        end_month = int(self.GetParameter("end_month", 1))
        end_day = int(self.GetParameter("end_day", 1))
        initial_cash = float(self.GetParameter("cash", 100000))

        n_states = int(self.GetParameter("n_states", 3))
        lookback_days = int(self.GetParameter("lookback_days", 252))
        min_confidence = float(self.GetParameter("min_confidence", 0.6))

        # Regime allocations (can be customized per regime)
        bull_allocation = float(self.GetParameter("bull_allocation", 1.0))
        bear_allocation = float(self.GetParameter("bear_allocation", 0.0))
        sideways_allocation = float(self.GetParameter("sideways_allocation", 0.5))

        # === BACKTEST CONFIGURATION ===
        self.SetStartDate(start_year, start_month, start_day)
        self.SetEndDate(end_year, end_month, end_day)
        self.SetCash(initial_cash)

        # Add equity
        self.symbol = self.AddEquity(ticker, Resolution.Daily).Symbol  # noqa: F405

        # Initialize regime detection
        self.initialize_regime_detection(
            ticker=ticker,
            n_states=n_states,
            lookback_days=lookback_days,
            retrain_frequency="never",
            regime_allocations={
                "Bull": bull_allocation,
                "Bear": bear_allocation,
                "Sideways": sideways_allocation,
            },
            min_confidence=min_confidence,
        )

        self.Debug(f"BasicRegimeSwitching initialized: {ticker} ({start_year}-{start_month}-{start_day} to {end_year}-{end_month}-{end_day})")

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
