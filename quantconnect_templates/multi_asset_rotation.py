"""
Multi-Asset Regime Rotation Strategy

This template demonstrates a more sophisticated strategy that:
- Detects regimes for multiple assets
- Rotates allocation to assets in favorable regimes
- Implements risk management through diversification

Author: hidden-regime
License: MIT
"""

# Note: When running in QuantConnect LEAN, uncomment this line:
# from AlgorithmImports import *

# For local testing/development:
import sys
sys.path.insert(0, '..')
from hidden_regime.quantconnect import HiddenRegimeAlgorithm


class MultiAssetRegimeRotation(HiddenRegimeAlgorithm):
    """
    Multi-asset rotation based on individual asset regimes.

    Strategy Logic:
        1. Monitor regimes for: SPY (stocks), QQQ (tech), TLT (bonds), GLD (gold)
        2. Allocate to assets in Bull regimes
        3. Reduce exposure to assets in Bear/Crisis regimes
        4. Rebalance weekly or on significant regime changes
    """

    def Initialize(self):
        """Initialize the algorithm."""
        # Backtest configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Define asset universe
        self.assets = {
            "SPY": "Stocks (S&P 500)",
            "QQQ": "Tech (Nasdaq)",
            "TLT": "Long-Term Bonds",
            "GLD": "Gold",
        }

        # Add securities
        self.symbols = {}
        for ticker, description in self.assets.items():
            self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.Debug(f"Added {ticker}: {description}")

        # Initialize regime detection for each asset
        for ticker in self.assets.keys():
            self.initialize_regime_detection(
                ticker=ticker,
                n_states=4,  # More granular regime detection
                lookback_days=180,  # 6 months
                retrain_frequency="weekly",
                min_confidence=0.65,
            )

        # Rebalancing schedule
        self.rebalance_days = 0
        self.rebalance_frequency = 5  # Rebalance every 5 days

        # Track allocations
        self.current_allocations = {ticker: 0.0 for ticker in self.assets.keys()}

        self.Debug("MultiAssetRegimeRotation initialized")

    def OnData(self, data):
        """
        Handle new market data.

        Args:
            data: Slice object containing market data
        """
        # Update regime detection for all assets
        for ticker, symbol in self.symbols.items():
            if data.ContainsKey(symbol):
                bar = data[symbol]
                self.on_tradebar(ticker, bar)

        # Check if all regimes are ready
        if not all(self.regime_is_ready(ticker) for ticker in self.assets.keys()):
            return

        # Update all regimes
        for ticker in self.assets.keys():
            self.update_regime(ticker)

        # Rebalance on schedule or significant changes
        self.rebalance_days += 1
        if self.rebalance_days >= self.rebalance_frequency:
            self.rebalance_portfolio()
            self.rebalance_days = 0

    def rebalance_portfolio(self):
        """
        Rebalance portfolio based on current regimes.
        """
        # Get signals for all assets
        signals = {}
        for ticker in self.assets.keys():
            signal = self.get_regime_signal(ticker)
            if signal:
                signals[ticker] = signal

        if not signals:
            return

        # Calculate allocations
        allocations = self.calculate_allocations(signals)

        # Execute trades
        for ticker, allocation in allocations.items():
            if ticker in self.symbols:
                symbol = self.symbols[ticker]
                self.SetHoldings(symbol, allocation)

        # Update tracking
        self.current_allocations = allocations

        # Log rebalancing
        self.log_portfolio_state(signals, allocations)

    def calculate_allocations(self, signals):
        """
        Calculate portfolio allocations based on regime signals.

        Strategy:
        - Allocate to assets in Bull/High regimes
        - Reduce exposure to Bear/Low regimes
        - Equal weight among favorable assets
        - Max 40% per asset (concentration limit)

        Args:
            signals: Dict of ticker -> TradingSignal

        Returns:
            Dict of ticker -> allocation
        """
        favorable_regimes = ["Bull", "High", "Sideways", "Medium"]
        max_allocation_per_asset = 0.40

        # Filter to favorable regimes
        favorable_assets = {
            ticker: signal
            for ticker, signal in signals.items()
            if signal.regime_name in favorable_regimes
            and signal.confidence >= 0.6
        }

        if not favorable_assets:
            # No favorable assets - go to defensive allocation
            return self.defensive_allocation()

        # Equal weight among favorable assets
        num_assets = len(favorable_assets)
        base_allocation = min(1.0 / num_assets, max_allocation_per_asset)

        allocations = {}
        for ticker in self.assets.keys():
            if ticker in favorable_assets:
                # Weight by confidence
                signal = favorable_assets[ticker]
                allocations[ticker] = base_allocation * signal.confidence
            else:
                allocations[ticker] = 0.0

        # Normalize to sum to 1.0
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}

        return allocations

    def defensive_allocation(self):
        """
        Defensive allocation when no assets in favorable regimes.

        Returns:
            Dict with allocation to bonds and gold
        """
        return {
            "SPY": 0.0,
            "QQQ": 0.0,
            "TLT": 0.6,  # 60% bonds
            "GLD": 0.4,  # 40% gold
        }

    def log_portfolio_state(self, signals, allocations):
        """
        Log current portfolio state.

        Args:
            signals: Current regime signals
            allocations: Current allocations
        """
        self.Log("="*50)
        self.Log("Portfolio Rebalance")
        self.Log(f"Date: {self.Time}")
        self.Log(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")

        for ticker in self.assets.keys():
            if ticker in signals:
                signal = signals[ticker]
                allocation = allocations.get(ticker, 0.0)
                self.Log(
                    f"{ticker}: {signal.regime_name} "
                    f"(conf: {signal.confidence:.1%}) → {allocation:.1%}"
                )

        self.Log("="*50)

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """
        Called when an asset's regime changes.

        Args:
            old_regime: Previous regime
            new_regime: New regime
            confidence: Confidence in new regime
            ticker: Asset ticker
        """
        self.Debug(
            f"{ticker} regime change: {old_regime} → {new_regime} "
            f"({confidence:.1%})"
        )

        # Consider immediate rebalance on significant changes
        if new_regime in ["Crisis", "Bear"] and confidence > 0.8:
            self.Debug(f"High-confidence bearish signal for {ticker}, rebalancing")
            self.rebalance_portfolio()
            self.rebalance_days = 0
