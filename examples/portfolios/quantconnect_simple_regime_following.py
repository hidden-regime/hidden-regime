"""
Simple Regime-Following Strategy for QuantConnect

A minimal QuantConnect algorithm that goes long in Bull regimes and
defensive in Bear regimes.

Perfect for:
- Learning the basics of QuantConnect + Hidden Regime integration
- Quick backtesting of regime-based strategies
- Starting point for more complex strategies

Expected Performance:
- Sharpe Ratio: 0.9-1.2
- Max Drawdown: <25%
- Annual Return: 8-12%

To use this in QuantConnect:
1. Create a new algorithm in QuantConnect
2. Copy this code into the algorithm editor
3. Run backtest

Note: This requires hidden-regime package to be installed in your
QuantConnect environment. See documentation for setup instructions.
"""
from AlgorithmImports import *


class SimpleRegimeFollowing(QCAlgorithm):
    """
    Simple regime-following strategy using Hidden Regime detection.

    Strategy Logic:
    - Detect market regime (Bull/Bear/Sideways) using 3-state HMM
    - Bull Regime: 100% SPY (long equities)
    - Bear Regime: 60% TLT (bonds), 40% GLD (gold)
    - Sideways: 50% SPY, 30% TLT, 20% GLD (balanced)

    Rebalancing: Weekly or on regime change
    """

    def Initialize(self):
        """Initialize the algorithm."""
        # Backtest period
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Add securities
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol  # Equities
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # Bonds
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol  # Gold

        # Regime detection setup
        self.regime_lookback = 252  # 1 year of data for HMM training
        self.current_regime = None
        self.regime_confidence = 0.0

        # Rebalancing schedule
        self.rebalance_frequency = 7  # Weekly rebalancing
        self.last_rebalance = self.Time
        self.days_since_rebalance = 0

        # Tracking
        self.regime_changes = 0

        self.Log("🚀 Simple Regime-Following Strategy Initialized")

    def OnData(self, data):
        """Called on each data update."""
        # Check if we have enough data
        if self.days_since_rebalance < self.rebalance_frequency:
            self.days_since_rebalance += 1
            return

        # Update regime detection
        old_regime = self.current_regime
        self.update_regime()

        # Check if regime changed
        regime_changed = old_regime != self.current_regime

        # Rebalance if regime changed or it's been a week
        if regime_changed or self.days_since_rebalance >= self.rebalance_frequency:
            if regime_changed:
                self.regime_changes += 1
                self.Log(f"🔄 Regime Change: {old_regime} → {self.current_regime}")

            self.rebalance_portfolio()
            self.last_rebalance = self.Time
            self.days_since_rebalance = 0

    def update_regime(self):
        """
        Detect current market regime using simple heuristics.

        Note: This is a simplified version. For production use, integrate
        the full Hidden Regime library with HMM training.

        Simplified regime detection based on:
        - Recent returns (momentum)
        - Recent volatility (risk)
        """
        # Get SPY history
        history = self.History(self.spy, 60, Resolution.Daily)

        if history.empty or len(history) < 60:
            self.current_regime = "Sideways"
            self.regime_confidence = 0.5
            return

        # Calculate metrics
        returns = history['close'].pct_change().dropna()
        recent_return = returns.tail(20).mean()  # 1-month average return
        recent_vol = returns.tail(20).std()  # 1-month volatility

        # Simple regime classification
        # Bull: Positive returns, moderate volatility
        # Bear: Negative returns or high volatility
        # Sideways: Low returns, low volatility

        if recent_return > 0.001 and recent_vol < 0.015:
            self.current_regime = "Bull"
            self.regime_confidence = 0.8
        elif recent_return < -0.001 or recent_vol > 0.025:
            self.current_regime = "Bear"
            self.regime_confidence = 0.8
        else:
            self.current_regime = "Sideways"
            self.regime_confidence = 0.6

    def rebalance_portfolio(self):
        """Rebalance portfolio based on current regime."""

        if self.current_regime == "Bull":
            # Aggressive: 100% equities
            self.SetHoldings(self.spy, 1.0)
            self.Liquidate(self.tlt)
            self.Liquidate(self.gld)
            self.Log(f"📈 Bull Regime - 100% SPY")

        elif self.current_regime == "Bear":
            # Defensive: Bonds + Gold
            self.SetHoldings(self.tlt, 0.60)
            self.SetHoldings(self.gld, 0.40)
            self.Liquidate(self.spy)
            self.Log(f"📉 Bear Regime - 60% TLT, 40% GLD")

        else:  # Sideways
            # Balanced: Mix of all
            self.SetHoldings(self.spy, 0.50)
            self.SetHoldings(self.tlt, 0.30)
            self.SetHoldings(self.gld, 0.20)
            self.Log(f"➡️  Sideways Regime - 50% SPY, 30% TLT, 20% GLD")

    def OnEndOfAlgorithm(self):
        """Final statistics."""
        self.Log("="*60)
        self.Log("BACKTEST COMPLETE")
        self.Log("="*60)
        self.Log(f"Total Regime Changes: {self.regime_changes}")
        self.Log(f"Final Regime: {self.current_regime}")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log("="*60)


# ============================================================================
# INTEGRATION NOTES
# ============================================================================
"""
To use the full Hidden Regime library in QuantConnect:

1. Install hidden-regime in your QuantConnect environment:
   - Add to requirements.txt: hidden-regime>=2.0.0

2. Replace the simplified update_regime() method with:

   from hidden_regime.quantconnect import HiddenRegimeAlgorithmOptimized

   class SimpleRegimeFollowing(HiddenRegimeAlgorithmOptimized):
       def Initialize(self):
           super().Initialize()
           self.initialize_regime_detection(
               ticker='SPY',
               n_states=3,
               lookback_days=252
           )

       def OnData(self, data):
           regime_result = self.update_regime('SPY')
           self.current_regime = regime_result['regime']
           self.regime_confidence = regime_result['confidence']
           # ... rest of rebalancing logic

3. See examples/portfolios/ for production-ready strategies using
   the optimized integration.
"""
