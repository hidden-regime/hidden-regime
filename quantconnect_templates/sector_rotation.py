"""
Sector Rotation Based on Market Regimes

This template demonstrates sector-based regime rotation:
- Detect overall market regime (SPY)
- Rotate into sectors that perform best in each regime
- Bull regime: Technology, Consumer Discretionary
- Bear regime: Utilities, Consumer Staples
- Crisis regime: Healthcare, Utilities

Author: hidden-regime
License: MIT
"""

# Note: When running in QuantConnect LEAN, uncomment this line:
# from AlgorithmImports import *

# For local testing/development:
import sys
sys.path.insert(0, '..')
from hidden_regime.quantconnect import HiddenRegimeAlgorithm


class SectorRotationStrategy(HiddenRegimeAlgorithm):
    """
    Sector rotation strategy based on market regime detection.

    Strategy Logic:
        1. Detect market regime using SPY
        2. Rotate into sectors that historically outperform in each regime
        3. Equal weight within selected sectors
        4. Rebalance on regime changes or monthly

    Sector ETFs:
        - XLK: Technology
        - XLY: Consumer Discretionary
        - XLF: Financials
        - XLP: Consumer Staples
        - XLU: Utilities
        - XLV: Healthcare
        - XLE: Energy
        - XLI: Industrials
    """

    def Initialize(self):
        """Initialize the algorithm."""
        # Backtest configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Market proxy for regime detection
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Sector ETFs
        self.sectors = {
            "XLK": {  # Technology
                "symbol": self.AddEquity("XLK", Resolution.Daily).Symbol,
                "name": "Technology",
                "regimes": ["Bull", "Sideways"],
            },
            "XLY": {  # Consumer Discretionary
                "symbol": self.AddEquity("XLY", Resolution.Daily).Symbol,
                "name": "Consumer Discretionary",
                "regimes": ["Bull"],
            },
            "XLF": {  # Financials
                "symbol": self.AddEquity("XLF", Resolution.Daily).Symbol,
                "name": "Financials",
                "regimes": ["Bull", "Sideways"],
            },
            "XLP": {  # Consumer Staples
                "symbol": self.AddEquity("XLP", Resolution.Daily).Symbol,
                "name": "Consumer Staples",
                "regimes": ["Bear", "Crisis", "Sideways"],
            },
            "XLU": {  # Utilities
                "symbol": self.AddEquity("XLU", Resolution.Daily).Symbol,
                "name": "Utilities",
                "regimes": ["Bear", "Crisis"],
            },
            "XLV": {  # Healthcare
                "symbol": self.AddEquity("XLV", Resolution.Daily).Symbol,
                "name": "Healthcare",
                "regimes": ["Crisis", "Sideways"],
            },
            "XLE": {  # Energy
                "symbol": self.AddEquity("XLE", Resolution.Daily).Symbol,
                "name": "Energy",
                "regimes": ["Bull"],
            },
            "XLI": {  # Industrials
                "symbol": self.AddEquity("XLI", Resolution.Daily).Symbol,
                "name": "Industrials",
                "regimes": ["Bull", "Sideways"],
            },
        }

        # Initialize market regime detection
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=4,  # Bull, Sideways, Bear, Crisis
            lookback_days=252,
            retrain_frequency="monthly",
            min_confidence=0.60,
        )

        # Rebalancing settings
        self.days_since_rebalance = 0
        self.rebalance_frequency = 30  # Monthly or on regime change
        self.max_sectors = 4  # Maximum sectors to hold

        self.Debug("SectorRotationStrategy initialized")
        self.Debug(f"Monitoring {len(self.sectors)} sector ETFs")

    def OnData(self, data):
        """Handle new market data."""
        # Update regime detection
        if data.ContainsKey(self.spy):
            self.on_tradebar("SPY", data[self.spy])

        if not self.regime_is_ready():
            return

        # Update regime
        self.update_regime()

        # Increment rebalance counter
        self.days_since_rebalance += 1

        # Rebalance on schedule
        if self.days_since_rebalance >= self.rebalance_frequency:
            self.rebalance_portfolio()
            self.days_since_rebalance = 0

    def rebalance_portfolio(self):
        """Rebalance portfolio based on current regime."""
        # Get current regime
        regime = self.current_regime
        confidence = self.regime_confidence

        self.Log(f"Rebalancing for {regime} regime ({confidence:.1%} confidence)")

        # Select sectors for current regime
        selected_sectors = self.select_sectors_for_regime(regime)

        if not selected_sectors:
            # No sectors match - go to cash
            self.Liquidate()
            self.Log("No suitable sectors - moving to cash")
            return

        # Calculate equal weight allocation
        num_sectors = min(len(selected_sectors), self.max_sectors)
        weight = 1.0 / num_sectors

        # Log selected sectors
        sector_names = [self.sectors[ticker]["name"] for ticker in selected_sectors[:num_sectors]]
        self.Log(f"Selected sectors: {', '.join(sector_names)}")

        # Allocate to selected sectors
        allocated_tickers = []
        for ticker in selected_sectors[:num_sectors]:
            symbol = self.sectors[ticker]["symbol"]
            self.SetHoldings(symbol, weight)
            allocated_tickers.append(ticker)

        # Liquidate unselected sectors
        for ticker, info in self.sectors.items():
            if ticker not in allocated_tickers:
                symbol = info["symbol"]
                if self.Portfolio[symbol].Invested:
                    self.Liquidate(symbol)

    def select_sectors_for_regime(self, regime):
        """
        Select sectors that perform well in given regime.

        Args:
            regime: Current market regime

        Returns:
            List of ticker symbols for selected sectors
        """
        selected = []

        for ticker, info in self.sectors.items():
            if regime in info["regimes"]:
                selected.append(ticker)

        # Sort by recent performance (optional - using alphabetical for now)
        selected.sort()

        return selected

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """Called when regime transitions."""
        self.Log(f"REGIME CHANGE: {old_regime} → {new_regime} ({confidence:.1%})")

        # Immediate rebalance on regime change
        if confidence >= 0.70:  # High confidence regime change
            self.Log("High-confidence regime change - rebalancing immediately")
            self.rebalance_portfolio()
            self.days_since_rebalance = 0

    def OnEndOfDay(self):
        """End of day reporting."""
        # Monthly performance summary
        if self.Time.day == 1:
            portfolio_value = self.Portfolio.TotalPortfolioValue

            # Get current holdings
            holdings = []
            for ticker, info in self.sectors.items():
                symbol = info["symbol"]
                if self.Portfolio[symbol].Invested:
                    holding_pct = self.Portfolio[symbol].HoldingsValue / portfolio_value
                    holdings.append(f"{ticker} ({holding_pct:.1%})")

            self.Log("="*60)
            self.Log(f"Monthly Summary - {self.Time.strftime('%Y-%m-%d')}")
            self.Log(f"Portfolio Value: ${portfolio_value:,.2f}")
            self.Log(f"Current Regime: {self.current_regime} ({self.regime_confidence:.1%})")
            self.Log(f"Holdings: {', '.join(holdings) if holdings else 'Cash'}")
            self.Log("="*60)


class AdvancedSectorRotation(HiddenRegimeAlgorithm):
    """
    Advanced sector rotation with individual sector regime detection.

    Instead of detecting market regime and rotating sectors,
    this strategy detects regime for EACH sector independently
    and allocates to sectors in their best regimes.
    """

    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Sector ETFs
        self.sector_tickers = ["XLK", "XLY", "XLF", "XLP", "XLU", "XLV", "XLE", "XLI"]
        self.sectors = {}

        for ticker in self.sector_tickers:
            symbol = self.AddEquity(ticker, Resolution.Daily).Symbol
            self.sectors[ticker] = {
                "symbol": symbol,
                "regime": None,
                "confidence": 0.0,
            }

            # Initialize regime detection for each sector
            self.initialize_regime_detection(
                ticker=ticker,
                n_states=3,
                lookback_days=180,
                retrain_frequency="monthly",
            )

        # Rebalancing
        self.rebalance_days = 0
        self.rebalance_frequency = 7  # Weekly

        self.Debug("AdvancedSectorRotation initialized")
        self.Debug(f"Individual regime detection for {len(self.sectors)} sectors")

    def OnData(self, data):
        """Handle new market data."""
        # Update all sector regimes
        for ticker, info in self.sectors.items():
            if data.ContainsKey(info["symbol"]):
                self.on_tradebar(ticker, data[info["symbol"]])

        # Check if all regimes are ready
        if not all(self.regime_is_ready(t) for t in self.sector_tickers):
            return

        # Update all regimes
        for ticker in self.sector_tickers:
            self.update_regime(ticker)
            signal = self.get_regime_signal(ticker)
            if signal:
                self.sectors[ticker]["regime"] = signal.regime_name
                self.sectors[ticker]["confidence"] = signal.confidence

        # Rebalance on schedule
        self.rebalance_days += 1
        if self.rebalance_days >= self.rebalance_frequency:
            self.rebalance_by_sector_regimes()
            self.rebalance_days = 0

    def rebalance_by_sector_regimes(self):
        """Allocate to sectors in Bull regimes with high confidence."""
        # Find sectors in Bull regime
        bull_sectors = []
        for ticker, info in self.sectors.items():
            if info["regime"] == "Bull" and info["confidence"] >= 0.65:
                bull_sectors.append((ticker, info["confidence"]))

        if not bull_sectors:
            # No bull sectors - use defensive allocation
            self.defensive_allocation()
            return

        # Sort by confidence (highest first)
        bull_sectors.sort(key=lambda x: x[1], reverse=True)

        # Allocate to top sectors
        max_holdings = min(4, len(bull_sectors))
        weight = 1.0 / max_holdings

        for i, (ticker, confidence) in enumerate(bull_sectors[:max_holdings]):
            symbol = self.sectors[ticker]["symbol"]
            self.SetHoldings(symbol, weight)

        # Liquidate others
        for ticker, info in self.sectors.items():
            if ticker not in [t for t, _ in bull_sectors[:max_holdings]]:
                if self.Portfolio[info["symbol"]].Invested:
                    self.Liquidate(info["symbol"])

    def defensive_allocation(self):
        """Defensive allocation when no bull sectors."""
        # Allocate to defensive sectors (XLP, XLU, XLV)
        defensive = ["XLP", "XLU", "XLV"]
        weight = 1.0 / len(defensive)

        for ticker in defensive:
            if ticker in self.sectors:
                self.SetHoldings(self.sectors[ticker]["symbol"], weight)

        # Liquidate non-defensive
        for ticker, info in self.sectors.items():
            if ticker not in defensive and self.Portfolio[info["symbol"]].Invested:
                self.Liquidate(info["symbol"])
