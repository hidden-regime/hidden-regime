"""
Crisis Detection and Defensive Allocation Strategy

This template demonstrates advanced crisis regime detection:
- 4-state HMM to capture crisis regimes
- Automatic defensive allocation during crisis
- Flight to safety: bonds and gold
- Quick regime adaptation with shorter lookback

Author: hidden-regime
License: MIT
"""

# Note: When running in QuantConnect LEAN, uncomment this line:
# from AlgorithmImports import *

# For local testing/development:
import sys
sys.path.insert(0, '..')
from hidden_regime.quantconnect import HiddenRegimeAlgorithm


class CrisisDetectionStrategy(HiddenRegimeAlgorithm):
    """
    Crisis detection strategy with defensive positioning.

    Strategy Logic:
        1. Use 4-state HMM to detect: Bull, Sideways, Bear, Crisis
        2. Monitor SPY for market-wide regime
        3. Crisis regime triggers immediate defensive allocation
        4. Normal regimes use moderate risk positioning
        5. Fast adaptation with 90-day lookback

    Defensive Assets:
        - TLT: Long-term Treasury bonds (flight to quality)
        - GLD: Gold (crisis hedge)
        - SHY: Short-term Treasury (capital preservation)
    """

    def Initialize(self):
        """Initialize the algorithm."""
        # Backtest configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Add securities
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol  # Market proxy
        self.tlt = self.AddEquity("TLT", Resolution.Daily).Symbol  # Long bonds
        self.gld = self.AddEquity("GLD", Resolution.Daily).Symbol  # Gold
        self.shy = self.AddEquity("SHY", Resolution.Daily).Symbol  # Short bonds

        # Initialize regime detection with 4 states for crisis detection
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=4,  # Bull, Sideways, Bear, Crisis
            lookback_days=90,  # Shorter lookback for faster crisis detection
            retrain_frequency="weekly",
            regime_allocations={
                "Bull": 1.0,      # 100% stocks (SPY)
                "Sideways": 0.6,  # 60% stocks, 40% defensive
                "Bear": 0.0,      # Move to defensive
                "Crisis": 0.0,    # Full defensive mode
            },
            min_confidence=0.65,
        )

        # Track regime history for crisis analysis
        self.regime_history = []
        self.crisis_mode = False
        self.days_in_crisis = 0

        # Alert settings
        self.crisis_confidence_threshold = 0.75

        self.Debug("CrisisDetectionStrategy initialized")
        self.Debug(f"Monitoring for crisis regimes with {self.crisis_confidence_threshold:.0%} confidence")

    def OnData(self, data):
        """Handle new market data."""
        # Check for required data
        if not all(data.ContainsKey(s) for s in [self.spy, self.tlt, self.gld, self.shy]):
            return

        # Update regime detection
        spy_bar = data[self.spy]
        self.on_tradebar("SPY", spy_bar)

        # Wait for regime detection to be ready
        if not self.regime_is_ready():
            return

        # Update regime
        self.update_regime()

        # Record regime history
        self.regime_history.append({
            "time": self.Time,
            "regime": self.current_regime,
            "confidence": self.regime_confidence,
        })

        # Trim history to last 30 days
        if len(self.regime_history) > 30:
            self.regime_history = self.regime_history[-30:]

        # Check for crisis regime
        is_crisis = self.current_regime in ["Crisis", "Bear"] and \
                    self.regime_confidence >= self.crisis_confidence_threshold

        # Handle crisis transitions
        if is_crisis and not self.crisis_mode:
            self.enter_crisis_mode()
        elif not is_crisis and self.crisis_mode:
            self.exit_crisis_mode()

        # Execute allocation based on regime
        self.allocate_portfolio()

    def enter_crisis_mode(self):
        """Enter crisis defensive mode."""
        self.crisis_mode = True
        self.days_in_crisis = 0

        self.Log(f"⚠️  CRISIS DETECTED - Entering defensive mode")
        self.Log(f"   Regime: {self.current_regime}, Confidence: {self.regime_confidence:.1%}")

        # Immediate defensive allocation
        self.Liquidate(self.spy)  # Exit stocks immediately

    def exit_crisis_mode(self):
        """Exit crisis mode and return to normal allocation."""
        self.crisis_mode = False

        self.Log(f"✓ Crisis mode ended after {self.days_in_crisis} days")
        self.Log(f"   New regime: {self.current_regime}, Confidence: {self.regime_confidence:.1%}")

    def allocate_portfolio(self):
        """
        Allocate portfolio based on current regime.

        Allocation Strategy:
            Crisis/Bear (High Confidence):
                - 50% TLT (long-term bonds)
                - 30% GLD (gold)
                - 20% SHY (short-term bonds)

            Bear (Low Confidence):
                - 40% TLT
                - 30% SHY
                - 30% GLD

            Sideways:
                - 60% SPY
                - 25% TLT
                - 15% GLD

            Bull:
                - 100% SPY
        """
        if self.crisis_mode:
            # Crisis allocation: Flight to safety
            self.SetHoldings(self.tlt, 0.50)
            self.SetHoldings(self.gld, 0.30)
            self.SetHoldings(self.shy, 0.20)
            self.SetHoldings(self.spy, 0.00)

            self.days_in_crisis += 1

        elif self.current_regime == "Bear":
            # Bear market: Defensive but not crisis
            self.SetHoldings(self.tlt, 0.40)
            self.SetHoldings(self.shy, 0.30)
            self.SetHoldings(self.gld, 0.30)
            self.SetHoldings(self.spy, 0.00)

        elif self.current_regime == "Sideways":
            # Sideways: Moderate exposure
            self.SetHoldings(self.spy, 0.60)
            self.SetHoldings(self.tlt, 0.25)
            self.SetHoldings(self.gld, 0.15)
            self.SetHoldings(self.shy, 0.00)

        else:  # Bull
            # Bull market: Full equity exposure
            self.SetHoldings(self.spy, 1.00)
            self.SetHoldings(self.tlt, 0.00)
            self.SetHoldings(self.gld, 0.00)
            self.SetHoldings(self.shy, 0.00)

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """Called when regime transitions."""
        self.Log(f"Regime Transition: {old_regime} → {new_regime} ({confidence:.1%})")

        # Analyze regime persistence
        if len(self.regime_history) >= 5:
            recent_regimes = [r["regime"] for r in self.regime_history[-5:]]
            regime_changes = len(set(recent_regimes))

            if regime_changes == 1:
                self.Debug(f"Stable regime detected: {new_regime}")
            elif regime_changes >= 4:
                self.Debug(f"High volatility: {regime_changes} regime changes in 5 days")

    def OnEndOfDay(self):
        """End of day summary."""
        if self.Time.day % 7 == 0:  # Weekly summary
            # Calculate portfolio metrics
            portfolio_value = self.Portfolio.TotalPortfolioValue
            cash_pct = self.Portfolio.Cash / portfolio_value

            self.Log("="*50)
            self.Log(f"Weekly Summary - {self.Time.strftime('%Y-%m-%d')}")
            self.Log(f"Portfolio Value: ${portfolio_value:,.2f}")
            self.Log(f"Current Regime: {self.current_regime} ({self.regime_confidence:.1%})")
            self.Log(f"Crisis Mode: {self.crisis_mode}")
            if self.crisis_mode:
                self.Log(f"Days in Crisis: {self.days_in_crisis}")
            self.Log(f"Cash: {cash_pct:.1%}")
            self.Log("="*50)
