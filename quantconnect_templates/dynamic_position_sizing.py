"""
Dynamic Position Sizing Based on Regime Confidence

This template demonstrates intelligent position sizing:
- Base allocation on regime type
- Scale position size by confidence level
- Increase leverage in high-confidence Bull regimes
- Reduce exposure in low-confidence or Bear regimes
- Risk management with stop-loss and position limits

Author: hidden-regime
License: MIT
"""

# Note: When running in QuantConnect LEAN, uncomment this line:
# from AlgorithmImports import *

# For local testing/development:
import sys
sys.path.insert(0, '..')
from hidden_regime.quantconnect import HiddenRegimeAlgorithm


class DynamicPositionSizing(HiddenRegimeAlgorithm):
    """
    Dynamic position sizing strategy based on regime confidence.

    Position Sizing Rules:
        Bull Regime:
            - High confidence (>80%): 100% allocation
            - Medium confidence (60-80%): 70% allocation
            - Low confidence (<60%): 40% allocation

        Sideways Regime:
            - High confidence: 50% allocation
            - Medium confidence: 30% allocation
            - Low confidence: 20% allocation

        Bear/Crisis Regime:
            - Any confidence: 0% allocation (cash)

    Risk Management:
        - Maximum drawdown: 15%
        - Stop-loss: 5% from recent high
        - Trailing stop in Bull regimes
    """

    def Initialize(self):
        """Initialize the algorithm."""
        # Backtest configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # Add SPY
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Initialize regime detection
        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,
            lookback_days=252,
            retrain_frequency="weekly",
            min_confidence=0.0,  # Accept all confidence levels
        )

        # Risk management
        self.max_drawdown = 0.15  # 15% max drawdown
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.recent_high = self.Portfolio.TotalPortfolioValue
        self.stop_loss_level = None

        # Performance tracking
        self.daily_returns = []

        self.Debug("DynamicPositionSizing initialized")
        self.Debug(f"Max drawdown: {self.max_drawdown:.0%}, Stop loss: {self.stop_loss_pct:.0%}")

    def OnData(self, data):
        """Handle new market data."""
        if not data.ContainsKey(self.symbol):
            return

        # Update regime detection
        self.on_tradebar("SPY", data[self.symbol])

        if not self.regime_is_ready():
            return

        # Update regime
        self.update_regime()

        # Check risk limits
        if self.check_risk_limits():
            self.Liquidate()
            return

        # Calculate target allocation
        target_allocation = self.calculate_position_size()

        # Execute trades
        current_allocation = 0.0
        if self.Portfolio[self.symbol].Invested:
            current_allocation = self.Portfolio[self.symbol].HoldingsValue / \
                               self.Portfolio.TotalPortfolioValue

        # Only rebalance if change is significant (>5%)
        allocation_change = abs(target_allocation - current_allocation)
        if allocation_change >= 0.05:
            self.SetHoldings(self.symbol, target_allocation)

            self.Debug(
                f"Rebalance: {current_allocation:.1%} → {target_allocation:.1%} "
                f"(Regime: {self.current_regime}, Conf: {self.regime_confidence:.1%})"
            )

            # Update stop loss
            if target_allocation > 0:
                current_price = data[self.symbol].Close
                self.stop_loss_level = current_price * (1 - self.stop_loss_pct)

        # Update tracking
        self.update_risk_tracking()

    def calculate_position_size(self):
        """
        Calculate position size based on regime and confidence.

        Returns:
            float: Target allocation (0.0 to 1.0)
        """
        regime = self.current_regime
        confidence = self.regime_confidence

        # Bear or Crisis: Go to cash
        if regime in ["Bear", "Crisis", "Low"]:
            return 0.0

        # Bull regime: Scale by confidence
        if regime in ["Bull", "High"]:
            if confidence >= 0.80:
                return 1.00  # High confidence - full allocation
            elif confidence >= 0.60:
                return 0.70  # Medium confidence
            else:
                return 0.40  # Low confidence

        # Sideways regime: Moderate exposure
        if regime in ["Sideways", "Medium"]:
            if confidence >= 0.80:
                return 0.50
            elif confidence >= 0.60:
                return 0.30
            else:
                return 0.20

        # Default: conservative
        return 0.20

    def check_risk_limits(self):
        """
        Check if risk limits are breached.

        Returns:
            bool: True if should exit positions
        """
        portfolio_value = self.Portfolio.TotalPortfolioValue

        # Update recent high
        if portfolio_value > self.recent_high:
            self.recent_high = portfolio_value

        # Check max drawdown
        drawdown = (self.recent_high - portfolio_value) / self.recent_high
        if drawdown >= self.max_drawdown:
            self.Log(f"⚠️  MAX DRAWDOWN BREACHED: {drawdown:.1%}")
            return True

        # Check stop loss
        if self.Portfolio[self.symbol].Invested and self.stop_loss_level:
            current_price = self.Securities[self.symbol].Price
            if current_price <= self.stop_loss_level:
                self.Log(f"⚠️  STOP LOSS HIT: ${current_price:.2f} <= ${self.stop_loss_level:.2f}")
                self.stop_loss_level = None
                return True

        return False

    def update_risk_tracking(self):
        """Update risk metrics and tracking."""
        portfolio_value = self.Portfolio.TotalPortfolioValue

        # Track daily returns
        if len(self.daily_returns) > 0:
            prev_value = self.daily_returns[-1]["value"]
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = 0.0

        self.daily_returns.append({
            "time": self.Time,
            "value": portfolio_value,
            "return": daily_return,
        })

        # Keep last 252 days
        if len(self.daily_returns) > 252:
            self.daily_returns = self.daily_returns[-252:]

    def on_regime_change(self, old_regime, new_regime, confidence, ticker):
        """Called when regime transitions."""
        self.Log(
            f"Regime Change: {old_regime} → {new_regime} "
            f"({confidence:.1%} confidence)"
        )

        # Reset stop loss on regime change
        if new_regime in ["Bear", "Crisis"]:
            self.stop_loss_level = None

    def OnEndOfDay(self):
        """End of day summary."""
        # Weekly summary
        if self.Time.weekday() == 4:  # Friday
            portfolio_value = self.Portfolio.TotalPortfolioValue
            drawdown = (self.recent_high - portfolio_value) / self.recent_high

            # Calculate weekly metrics
            if len(self.daily_returns) >= 5:
                week_returns = [r["return"] for r in self.daily_returns[-5:]]
                week_return = sum(week_returns)
                week_volatility = (sum((r - (week_return/5))**2 for r in week_returns) / 5) ** 0.5
            else:
                week_return = 0.0
                week_volatility = 0.0

            # Current position
            allocation = 0.0
            if self.Portfolio[self.symbol].Invested:
                allocation = self.Portfolio[self.symbol].HoldingsValue / portfolio_value

            self.Log("="*60)
            self.Log(f"Weekly Summary - {self.Time.strftime('%Y-%m-%d')}")
            self.Log(f"Portfolio Value: ${portfolio_value:,.2f}")
            self.Log(f"Current Drawdown: {drawdown:.2%}")
            self.Log(f"Weekly Return: {week_return:.2%}")
            self.Log(f"Weekly Volatility: {week_volatility:.2%}")
            self.Log(f"Regime: {self.current_regime} ({self.regime_confidence:.1%})")
            self.Log(f"Allocation: {allocation:.1%}")
            if self.stop_loss_level:
                self.Log(f"Stop Loss: ${self.stop_loss_level:.2f}")
            self.Log("="*60)


class KellyPositionSizing(HiddenRegimeAlgorithm):
    """
    Kelly Criterion position sizing based on regime statistics.

    Uses Kelly formula: f = (p*b - q) / b
    Where:
        f = fraction of capital to bet
        p = probability of win (regime win rate)
        q = probability of loss (1 - p)
        b = win/loss ratio

    Adjusts position size based on historical regime performance.
    """

    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.initialize_regime_detection(
            ticker="SPY",
            n_states=3,
            lookback_days=252,
            retrain_frequency="monthly",
        )

        # Track regime performance
        self.regime_stats = {
            "Bull": {"wins": 0, "losses": 0, "avg_win": 0.0, "avg_loss": 0.0},
            "Sideways": {"wins": 0, "losses": 0, "avg_win": 0.0, "avg_loss": 0.0},
            "Bear": {"wins": 0, "losses": 0, "avg_win": 0.0, "avg_loss": 0.0},
        }

        self.last_price = None
        self.kelly_fraction = 0.0

    def OnData(self, data):
        """Handle new market data."""
        if not data.ContainsKey(self.symbol):
            return

        current_price = data[self.symbol].Close

        # Update regime
        self.on_tradebar("SPY", data[self.symbol])

        if not self.regime_is_ready():
            self.last_price = current_price
            return

        # Update regime statistics
        if self.last_price and self.current_regime:
            self.update_regime_stats(current_price)

        self.update_regime()

        # Calculate Kelly fraction
        self.kelly_fraction = self.calculate_kelly()

        # Apply Kelly sizing
        self.SetHoldings(self.symbol, self.kelly_fraction)

        self.last_price = current_price

    def update_regime_stats(self, current_price):
        """Update regime performance statistics."""
        if not self.last_price:
            return

        daily_return = (current_price - self.last_price) / self.last_price
        regime = self.current_regime

        if regime not in self.regime_stats:
            return

        stats = self.regime_stats[regime]

        if daily_return > 0:
            stats["wins"] += 1
            # Update average win
            n = stats["wins"]
            stats["avg_win"] = ((n-1) * stats["avg_win"] + daily_return) / n
        else:
            stats["losses"] += 1
            # Update average loss
            n = stats["losses"]
            stats["avg_loss"] = ((n-1) * stats["avg_loss"] + abs(daily_return)) / n

    def calculate_kelly(self):
        """
        Calculate Kelly fraction for current regime.

        Returns:
            float: Kelly fraction (capped at 1.0 for safety)
        """
        regime = self.current_regime

        if regime not in self.regime_stats:
            return 0.25  # Default conservative allocation

        stats = self.regime_stats[regime]
        total_trades = stats["wins"] + stats["losses"]

        if total_trades < 20:  # Not enough data
            return 0.25

        # Calculate win probability
        win_prob = stats["wins"] / total_trades

        # Calculate win/loss ratio
        if stats["avg_loss"] > 0:
            win_loss_ratio = stats["avg_win"] / stats["avg_loss"]
        else:
            win_loss_ratio = 2.0  # Default

        # Kelly formula: f = (p*b - q) / b
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        # Cap Kelly fraction for safety (half Kelly)
        kelly = max(0.0, min(kelly * 0.5, 1.0))

        return kelly
