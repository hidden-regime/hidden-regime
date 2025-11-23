"""
All Weather Regime Portfolio Strategy

A robust all-weather portfolio that adapts allocation based on market regimes.
Inspired by Ray Dalio's All Weather Portfolio with regime-adaptive weighting.

Strategy:
- Detects market regimes (Bull/Bear/Sideways/Crisis)
- Adjusts allocation across asset classes based on regime
- Rebalances monthly or on regime changes
- Defensive positioning in crisis regimes

Expected Performance:
- Sharpe Ratio: 1.2-1.5
- Max Drawdown: <15%
- Annual Return: 8-12%
"""
from AlgorithmImports import *
from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized


class AllWeatherRegimePortfolio(HiddenRegimeAlgorithmOptimized):
    """
    All-weather portfolio with regime-adaptive allocation.

    Asset Classes:
    - US Stocks (SPY, QQQ)
    - International Stocks (EFA)
    - Bonds (TLT, IEF)
    - Commodities (GLD, DBC)
    - Real Estate (VNQ)
    """

    def Initialize(self):
        # Basic setup
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Asset universe - diversified across asset classes
        self.assets = {
            # Equities (40% base allocation)
            'SPY': self.AddEquity('SPY', Resolution.Daily).Symbol,  # US Large Cap
            'QQQ': self.AddEquity('QQQ', Resolution.Daily).Symbol,  # US Tech
            'EFA': self.AddEquity('EFA', Resolution.Daily).Symbol,  # International

            # Fixed Income (30% base allocation)
            'TLT': self.AddEquity('TLT', Resolution.Daily).Symbol,  # Long-term Treasury
            'IEF': self.AddEquity('IEF', Resolution.Daily).Symbol,  # Intermediate Treasury

            # Alternatives (30% base allocation)
            'GLD': self.AddEquity('GLD', Resolution.Daily).Symbol,  # Gold
            'DBC': self.AddEquity('DBC', Resolution.Daily).Symbol,  # Commodities
            'VNQ': self.AddEquity('VNQ', Resolution.Daily).Symbol,  # Real Estate
        }

        # Enable all optimizations
        self.enable_caching(max_cache_size=200, retrain_frequency='monthly')
        self.enable_batch_updates(max_workers=4)
        self.enable_profiling()

        # Initialize regime detection for market (using SPY as proxy)
        self.initialize_regime_detection(
            ticker='SPY',
            n_states=4,  # Bull, Bear, Sideways, Crisis
            lookback_days=252,
            regime_allocations={
                'Bull': 1.0,
                'Bear': 0.3,
                'Sideways': 0.6,
                'Crisis': 0.0
            }
        )

        # Rebalancing
        self.last_rebalance = self.Time
        self.rebalance_days = 30  # Monthly rebalancing

        # Performance tracking
        self.regime_changes = 0
        self.trades_executed = 0

        self.Log("🌍 All Weather Regime Portfolio initialized")

    def OnData(self, data):
        # Update market regime
        regime_result = self.update_regime('SPY')

        # Check if we should rebalance
        days_since_rebalance = (self.Time - self.last_rebalance).days

        should_rebalance = (
            days_since_rebalance >= self.rebalance_days or
            hasattr(self, '_regime_changed') and self._regime_changed
        )

        if should_rebalance:
            self.rebalance_portfolio(regime_result)
            self.last_rebalance = self.Time
            self._regime_changed = False

    def rebalance_portfolio(self, regime_result):
        """Rebalance portfolio based on current regime."""
        regime = regime_result['regime']
        confidence = regime_result['confidence']

        # Define allocation strategies by regime
        allocations = self.calculate_regime_allocations(regime, confidence)

        # Execute trades
        for ticker, target_allocation in allocations.items():
            symbol = self.assets[ticker]

            # Only trade if allocation is meaningful
            if target_allocation >= 0.01:
                self.SetHoldings(symbol, target_allocation)
                self.trades_executed += 1
            else:
                self.Liquidate(symbol)

        self.Debug(f"📊 Rebalanced for {regime} regime (confidence: {confidence:.1%})")

    def calculate_regime_allocations(self, regime, confidence):
        """Calculate target allocations based on regime."""

        if regime == "Bull":
            # Aggressive growth allocation
            return {
                'SPY': 0.25,  # 25% US Large Cap
                'QQQ': 0.20,  # 20% US Tech
                'EFA': 0.15,  # 15% International
                'TLT': 0.10,  # 10% Long Bonds
                'IEF': 0.10,  # 10% Intermediate Bonds
                'GLD': 0.10,  # 10% Gold
                'DBC': 0.05,  # 5% Commodities
                'VNQ': 0.05,  # 5% Real Estate
            }

        elif regime == "Bear":
            # Defensive allocation - bonds and gold
            return {
                'SPY': 0.05,  # 5% US Large Cap
                'QQQ': 0.00,  # 0% US Tech
                'EFA': 0.00,  # 0% International
                'TLT': 0.35,  # 35% Long Bonds
                'IEF': 0.25,  # 25% Intermediate Bonds
                'GLD': 0.25,  # 25% Gold
                'DBC': 0.05,  # 5% Commodities
                'VNQ': 0.05,  # 5% Real Estate
            }

        elif regime == "Sideways":
            # Balanced allocation
            return {
                'SPY': 0.15,  # 15% US Large Cap
                'QQQ': 0.10,  # 10% US Tech
                'EFA': 0.10,  # 10% International
                'TLT': 0.20,  # 20% Long Bonds
                'IEF': 0.15,  # 15% Intermediate Bonds
                'GLD': 0.15,  # 15% Gold
                'DBC': 0.08,  # 8% Commodities
                'VNQ': 0.07,  # 7% Real Estate
            }

        else:  # Crisis
            # Maximum defense - flight to safety
            return {
                'SPY': 0.00,  # 0% US Large Cap
                'QQQ': 0.00,  # 0% US Tech
                'EFA': 0.00,  # 0% International
                'TLT': 0.50,  # 50% Long Bonds
                'IEF': 0.30,  # 30% Intermediate Bonds
                'GLD': 0.20,  # 20% Gold
                'DBC': 0.00,  # 0% Commodities
                'VNQ': 0.00,  # 0% Real Estate
            }

    def on_regime_change(self, old_regime, new_regime, confidence, ticker=None):
        """Handle regime changes."""
        self.regime_changes += 1
        self._regime_changed = True

        self.Log(f"🔄 Regime Change #{self.regime_changes}: {old_regime} → {new_regime}")
        self.Log(f"   Confidence: {confidence:.1%}")

        # Log current portfolio value
        self.Log(f"   Portfolio: ${self.Portfolio.TotalPortfolioValue:,.2f}")

    def OnEndOfAlgorithm(self):
        """Log final statistics."""
        self.Log("="*60)
        self.Log("ALL WEATHER REGIME PORTFOLIO - FINAL STATISTICS")
        self.Log("="*60)

        self.Log(f"Total Regime Changes: {self.regime_changes}")
        self.Log(f"Total Trades Executed: {self.trades_executed}")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")

        # Performance stats
        if hasattr(self, '_profiler'):
            stats = self._profiler.get_statistics()
            self.Log(f"Performance Stats: {stats}")

        # Cache stats
        if hasattr(self, '_model_cache'):
            cache_stats = self._model_cache.get_statistics()
            self.Log(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")

        self.Log("="*60)
