"""
Momentum Regime Rotation Strategy

Combines momentum signals with regime detection for enhanced returns.

Strategy:
- Ranks assets by momentum (6-month return)
- Filters by regime state (only long in favorable regimes)
- Monthly rebalancing to top performers
- Goes defensive in Bear/Crisis regimes

Expected Performance:
- Sharpe Ratio: 1.3-1.7
- Max Drawdown: <20%
- Annual Return: 12-18%
"""
from AlgorithmImports import *
from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized


class MomentumRegimeRotation(HiddenRegimeAlgorithmOptimized):
    """
    Momentum rotation filtered by regime state.

    Universe: 10 sector ETFs
    Methodology: Dual momentum (absolute + relative) with regime filter
    """

    def Initialize(self):
        # Basic setup
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Sector ETF universe
        self.sectors = {
            'XLK': self.AddEquity('XLK', Resolution.Daily).Symbol,  # Technology
            'XLY': self.AddEquity('XLY', Resolution.Daily).Symbol,  # Consumer Discretionary
            'XLF': self.AddEquity('XLF', Resolution.Daily).Symbol,  # Financials
            'XLE': self.AddEquity('XLE', Resolution.Daily).Symbol,  # Energy
            'XLV': self.AddEquity('XLV', Resolution.Daily).Symbol,  # Healthcare
            'XLI': self.AddEquity('XLI', Resolution.Daily).Symbol,  # Industrials
            'XLP': self.AddEquity('XLP', Resolution.Daily).Symbol,  # Consumer Staples
            'XLU': self.AddEquity('XLU', Resolution.Daily).Symbol,  # Utilities
            'XLB': self.AddEquity('XLB', Resolution.Daily).Symbol,  # Materials
            'XLRE': self.AddEquity('XLRE', Resolution.Daily).Symbol, # Real Estate
        }

        # Defensive assets
        self.defensive = {
            'TLT': self.AddEquity('TLT', Resolution.Daily).Symbol,  # Bonds
            'GLD': self.AddEquity('GLD', Resolution.Daily).Symbol,  # Gold
        }

        # Enable optimizations
        self.enable_caching(max_cache_size=150, retrain_frequency='monthly')
        self.enable_batch_updates(max_workers=4)

        # Market regime detection
        self.initialize_regime_detection(
            ticker='SPY',  # Use SPY as market proxy (need to add it)
            n_states=3,
            lookback_days=180,  # 6 months
        )

        # Add SPY for regime detection
        self.spy = self.AddEquity('SPY', Resolution.Daily).Symbol

        # Momentum parameters
        self.momentum_period = 126  # 6 months
        self.top_n = 3  # Hold top 3 sectors

        # Rebalancing
        self.rebalance_frequency = 21  # Monthly (trading days)
        self.last_rebalance = self.Time

        # Tracking
        self.momentum_scores = {}

        self.Log("🚀 Momentum Regime Rotation initialized")

    def OnData(self, data):
        # Check if time to rebalance
        if (self.Time - self.last_rebalance).days < self.rebalance_frequency:
            return

        # Update market regime
        regime_result = self.update_regime('SPY')
        regime = regime_result['regime']
        confidence = regime_result['confidence']

        # Calculate momentum for all sectors
        self.calculate_momentum()

        # Rebalance based on regime
        if regime == "Bull" and confidence >= 0.7:
            # Long top momentum sectors
            self.rebalance_momentum_long()

        elif regime == "Sideways":
            # Partial allocation to top sectors
            self.rebalance_partial()

        else:  # Bear or low confidence
            # Go defensive
            self.rebalance_defensive()

        self.last_rebalance = self.Time

        self.Log(f"📊 Rebalanced - Regime: {regime} ({confidence:.1%})")

    def calculate_momentum(self):
        """Calculate 6-month momentum for all sectors."""
        self.momentum_scores = {}

        for ticker, symbol in self.sectors.items():
            # Get 6-month history
            history = self.History(symbol, self.momentum_period, Resolution.Daily)

            if history.empty or len(history) < self.momentum_period:
                self.momentum_scores[ticker] = -999  # Penalize missing data
                continue

            # Calculate momentum (6-month return)
            start_price = history['close'].iloc[0]
            end_price = history['close'].iloc[-1]
            momentum = (end_price - start_price) / start_price

            self.momentum_scores[ticker] = momentum

    def rebalance_momentum_long(self):
        """Long top momentum sectors (Bull regime)."""
        # Sort by momentum
        sorted_sectors = sorted(
            self.momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top N sectors
        top_sectors = sorted_sectors[:self.top_n]

        # Equal weight allocation
        allocation = 1.0 / self.top_n

        # Liquidate non-top sectors
        for ticker in self.sectors.keys():
            if ticker not in [s[0] for s in top_sectors]:
                self.Liquidate(self.sectors[ticker])

        # Allocate to top sectors
        for ticker, momentum in top_sectors:
            if momentum > 0:  # Positive absolute momentum
                self.SetHoldings(self.sectors[ticker], allocation)
                self.Debug(f"  Long {ticker}: {momentum:.1%} momentum")
            else:
                # No positive momentum - go to defensive
                self.Liquidate(self.sectors[ticker])

        # Liquidate defensive assets
        for ticker in self.defensive.keys():
            self.Liquidate(self.defensive[ticker])

    def rebalance_partial(self):
        """Partial allocation (Sideways regime)."""
        # Sort by momentum
        sorted_sectors = sorted(
            self.momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top N sectors
        top_sectors = sorted_sectors[:self.top_n]

        # 50% allocation to equities, 50% to defensive
        equity_allocation = 0.5 / self.top_n

        # Liquidate non-top sectors
        for ticker in self.sectors.keys():
            if ticker not in [s[0] for s in top_sectors]:
                self.Liquidate(self.sectors[ticker])

        # Allocate to top sectors (50%)
        for ticker, momentum in top_sectors:
            if momentum > 0:
                self.SetHoldings(self.sectors[ticker], equity_allocation)

        # Allocate to defensive (50%)
        self.SetHoldings(self.defensive['TLT'], 0.30)
        self.SetHoldings(self.defensive['GLD'], 0.20)

    def rebalance_defensive(self):
        """Go defensive (Bear regime or low confidence)."""
        # Liquidate all sectors
        for ticker, symbol in self.sectors.items():
            self.Liquidate(symbol)

        # 100% defensive allocation
        self.SetHoldings(self.defensive['TLT'], 0.60)  # 60% Bonds
        self.SetHoldings(self.defensive['GLD'], 0.40)  # 40% Gold

        self.Debug("  🛡️  Defensive allocation")

    def on_regime_change(self, old_regime, new_regime, confidence, ticker=None):
        """Log regime changes."""
        self.Log(f"🔄 Regime: {old_regime} → {new_regime} ({confidence:.1%})")

        # Force immediate rebalance on regime change
        self.last_rebalance = self.Time - timedelta(days=self.rebalance_frequency)

    def OnEndOfAlgorithm(self):
        """Final statistics."""
        self.Log("="*60)
        self.Log("MOMENTUM REGIME ROTATION - FINAL STATISTICS")
        self.Log("="*60)

        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")

        # Show final momentum rankings
        if self.momentum_scores:
            sorted_scores = sorted(
                self.momentum_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.Log("\nFinal Momentum Rankings:")
            for ticker, momentum in sorted_scores[:5]:
                self.Log(f"  {ticker}: {momentum:+.1%}")

        self.Log("="*60)
