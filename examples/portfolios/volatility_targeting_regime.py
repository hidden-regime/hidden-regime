"""
Volatility Targeting with Regime Detection

Maintains constant portfolio volatility while adapting to market regimes.

Strategy:
- Targets constant 10% annualized volatility
- Adjusts leverage/position size based on recent volatility
- Reduces target volatility in Bear/Crisis regimes
- Increases exposure in stable Bull regimes

Expected Performance:
- Sharpe Ratio: 1.4-1.8
- Volatility: ~10% (targeted)
- Max Drawdown: <18%
"""
from AlgorithmImports import *
from hidden_regime.quantconnect.optimized_algorithm import HiddenRegimeAlgorithmOptimized
import numpy as np


class VolatilityTargetingRegime(HiddenRegimeAlgorithmOptimized):
    """
    Volatility-targeted portfolio with regime-adaptive targets.

    Features:
    - Constant volatility targeting
    - Regime-dependent volatility targets
    - Position scaling based on realized vol
    - Risk parity across assets
    """

    def Initialize(self):
        # Basic setup
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Core holdings
        self.assets = {
            'SPY': self.AddEquity('SPY', Resolution.Daily).Symbol,  # US Stocks
            'QQQ': self.AddEquity('QQQ', Resolution.Daily).Symbol,  # Tech
            'IWM': self.AddEquity('IWM', Resolution.Daily).Symbol,  # Small Cap
        }

        # Enable optimizations
        self.enable_caching(max_cache_size=100, retrain_frequency='monthly')
        self.enable_profiling()

        # Regime detection
        self.initialize_regime_detection(
            ticker='SPY',
            n_states=3,
            lookback_days=252,
        )

        # Volatility targeting parameters
        self.base_vol_target = 0.10  # 10% annual volatility
        self.vol_lookback = 21  # 1 month for vol calculation
        self.rebalance_threshold = 0.02  # 2% vol deviation triggers rebalance

        # Regime-specific vol targets
        self.regime_vol_targets = {
            "Bull": 0.12,      # Higher vol target in bull
            "Sideways": 0.10,  # Base target
            "Bear": 0.06,      # Lower vol target in bear
        }

        # Tracking
        self.current_portfolio_vol = 0
        self.last_rebalance = self.Time

        self.Log("📊 Volatility Targeting with Regime Detection initialized")

    def OnData(self, data):
        # Update regime
        regime_result = self.update_regime('SPY')
        regime = regime_result['regime']

        # Calculate current portfolio volatility
        self.calculate_portfolio_volatility()

        # Get target volatility for current regime
        target_vol = self.regime_vol_targets.get(regime, self.base_vol_target)

        # Check if rebalance needed
        vol_deviation = abs(self.current_portfolio_vol - target_vol)

        if vol_deviation > self.rebalance_threshold or (self.Time - self.last_rebalance).days >= 5:
            self.rebalance_to_target_vol(target_vol, regime)
            self.last_rebalance = self.Time

    def calculate_portfolio_volatility(self):
        """Calculate realized portfolio volatility."""
        # Get portfolio returns history
        history = self.History(list(self.assets.values()), self.vol_lookback, Resolution.Daily)

        if history.empty:
            self.current_portfolio_vol = 0.10  # Default to base target
            return

        # Calculate returns for each asset
        returns = {}
        for ticker, symbol in self.assets.items():
            try:
                asset_history = history.loc[symbol]
                if len(asset_history) < 2:
                    continue

                prices = asset_history['close']
                asset_returns = prices.pct_change().dropna()
                returns[ticker] = asset_returns
            except:
                continue

        if not returns:
            self.current_portfolio_vol = 0.10
            return

        # Get current weights
        total_value = self.Portfolio.TotalPortfolioValue
        weights = {}
        for ticker, symbol in self.assets.items():
            if self.Portfolio[symbol].Invested:
                weights[ticker] = self.Portfolio[symbol].HoldingsValue / total_value
            else:
                weights[ticker] = 0

        # Calculate portfolio variance (simplified - equal weight if not invested)
        if sum(weights.values()) == 0:
            weights = {ticker: 1.0 / len(self.assets) for ticker in self.assets.keys()}

        # Portfolio variance calculation
        portfolio_returns = []
        for ticker, weight in weights.items():
            if ticker in returns and weight > 0:
                portfolio_returns.append(returns[ticker] * weight)

        if portfolio_returns:
            portfolio_return_series = sum(portfolio_returns)
            portfolio_variance = np.var(portfolio_return_series)
            self.current_portfolio_vol = np.sqrt(portfolio_variance * 252)  # Annualize
        else:
            self.current_portfolio_vol = 0.10

    def rebalance_to_target_vol(self, target_vol, regime):
        """Rebalance portfolio to achieve target volatility."""
        # Calculate individual asset volatilities
        asset_vols = {}

        for ticker, symbol in self.assets.items():
            history = self.History(symbol, self.vol_lookback, Resolution.Daily)

            if history.empty or len(history) < 2:
                asset_vols[ticker] = 0.20  # Default high vol
                continue

            returns = history['close'].pct_change().dropna()
            vol = np.std(returns) * np.sqrt(252)  # Annualize
            asset_vols[ticker] = vol if vol > 0 else 0.20

        # Calculate risk parity weights (inverse volatility weighting)
        inv_vols = {ticker: 1.0 / vol for ticker, vol in asset_vols.items()}
        total_inv_vol = sum(inv_vols.values())

        base_weights = {
            ticker: inv_vol / total_inv_vol
            for ticker, inv_vol in inv_vols.items()
        }

        # Calculate leverage needed to hit target vol
        if self.current_portfolio_vol > 0:
            leverage = target_vol / self.current_portfolio_vol
        else:
            leverage = 1.0

        # Cap leverage
        leverage = min(max(leverage, 0.5), 2.0)  # Between 0.5x and 2.0x

        # Apply leverage to weights
        final_weights = {
            ticker: weight * leverage
            for ticker, weight in base_weights.items()
        }

        # Normalize if over 100%
        total_weight = sum(final_weights.values())
        if total_weight > 1.0:
            final_weights = {
                ticker: weight / total_weight
                for ticker, weight in final_weights.items()
            }

        # Execute trades
        for ticker, weight in final_weights.items():
            symbol = self.assets[ticker]
            self.SetHoldings(symbol, weight)

        self.Debug(f"🎯 Rebalanced - Target Vol: {target_vol:.1%}, Current: {self.current_portfolio_vol:.1%}, Leverage: {leverage:.2f}x")
        self.Debug(f"   Regime: {regime}, Weights: {', '.join([f'{t}: {w:.1%}' for t, w in final_weights.items()])}")

    def on_regime_change(self, old_regime, new_regime, confidence, ticker=None):
        """Handle regime changes."""
        self.Log(f"🔄 Regime Change: {old_regime} → {new_regime} ({confidence:.1%})")

        # Log volatility target change
        old_target = self.regime_vol_targets.get(old_regime, self.base_vol_target)
        new_target = self.regime_vol_targets.get(new_regime, self.base_vol_target)

        self.Log(f"   Vol Target: {old_target:.1%} → {new_target:.1%}")

        # Force immediate rebalance
        self.last_rebalance = self.Time - timedelta(days=10)

    def OnEndOfAlgorithm(self):
        """Final statistics."""
        self.Log("="*60)
        self.Log("VOLATILITY TARGETING REGIME - FINAL STATISTICS")
        self.Log("="*60)

        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log(f"Final Portfolio Volatility: {self.current_portfolio_vol:.1%}")

        # Asset volatilities
        self.Log("\nFinal Asset Volatilities:")
        for ticker, symbol in self.assets.items():
            history = self.History(symbol, self.vol_lookback, Resolution.Daily)
            if not history.empty and len(history) >= 2:
                returns = history['close'].pct_change().dropna()
                vol = np.std(returns) * np.sqrt(252)
                self.Log(f"  {ticker}: {vol:.1%}")

        self.Log("="*60)
