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
    Simple regime-based strategy for SPY.

    Strategy Logic:
        1. Detect market regime using 3-state HMM
        2. Allocate based on regime:
           - Bull: 100% long SPY
           - Bear: 0% (cash)
           - Sideways: 50% long SPY
        3. Rebalance on regime changes
    """

    def Initialize(self):
        """
        Initialize the algorithm.

        Sets up:
        - Backtest period
        - Initial cash
        - SPY equity
        - Regime detection
        """
        # Backtest configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetCash(100000)

        # Add SPY
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Initialize regime detection
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,  # Bull, Bear, Sideways
            lookback_days=252,  # 1 year of data
            retrain_frequency="weekly",
            regime_allocations={
                "Bull": 1.0,  # 100% long
                "Bear": 0.0,  # Cash
                "Sideways": 0.5,  # 50% long
            },
            min_confidence=0.6,  # Only act on confident signals
        )

        self.Debug("BasicRegimeSwitching initialized")

    def OnData(self, data):
        """
        Handle new market data.

        Args:
            data: Slice object containing market data
        """
        # Skip if no data for SPY
        if not data.ContainsKey(self.symbol):
            return

        # Update regime detector with new bar
        bar = data[self.symbol]
        self.on_tradebar("SPY", bar)

        # Wait until regime detection is ready
        if not self.regime_is_ready():
            return

        # Update regime
        self.update_regime()

        # Get recommended allocation
        allocation = self.get_regime_allocation("SPY")

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
            f"Regime Change: {old_regime} → {new_regime} "
            f"(confidence: {confidence:.1%})"
        )

        # Log portfolio value
        portfolio_value = self.Portfolio.TotalPortfolioValue
        self.Log(f"Portfolio Value: ${portfolio_value:,.2f}")


# For QuantConnect cloud/local LEAN deployment, the algorithm class
# is automatically detected and instantiated.
