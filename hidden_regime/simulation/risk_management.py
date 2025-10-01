"""
Risk management system for trading simulation.

Provides stop-loss calculation, position sizing limits, and portfolio-level
risk controls as specified in simulation.md.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    max_position_pct: float = 0.1  # Maximum 10% of portfolio per position
    max_portfolio_risk: float = 0.02  # Maximum 2% portfolio risk per trade
    stop_loss_pct: float = 0.05  # 5% stop-loss by default
    max_total_exposure: float = 1.0  # Maximum 100% portfolio exposure
    max_drawdown_pct: float = 0.2  # Maximum 20% drawdown before reducing size


class RiskManager:
    """
    Portfolio risk management system.

    Handles position sizing, stop-loss calculation, and portfolio-level
    risk controls to prevent catastrophic losses from faulty signals.
    """

    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.

        Args:
            risk_limits: Risk limits configuration (uses defaults if None)
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.strategy_risk_multipliers: Dict[str, float] = {}

    def set_strategy_risk_multiplier(
        self, strategy_name: str, multiplier: float
    ) -> None:
        """
        Set risk multiplier for specific strategy.

        Args:
            strategy_name: Name of the trading strategy
            multiplier: Risk multiplier (1.0 = normal, 0.5 = half risk, 2.0 = double risk)
        """
        self.strategy_risk_multipliers[strategy_name] = multiplier

    def get_max_position_value(
        self, total_portfolio_value: float, strategy_name: str
    ) -> float:
        """
        Calculate maximum position value for a strategy.

        Args:
            total_portfolio_value: Current total portfolio value
            strategy_name: Name of the trading strategy

        Returns:
            Maximum dollar value for a single position
        """
        # Base position limit
        base_limit = total_portfolio_value * self.risk_limits.max_position_pct

        # Apply strategy-specific multiplier
        multiplier = self.strategy_risk_multipliers.get(strategy_name, 1.0)
        adjusted_limit = base_limit * multiplier

        # Ensure we don't exceed absolute limits
        return min(
            adjusted_limit, total_portfolio_value * self.risk_limits.max_total_exposure
        )

    def calculate_stop_loss(
        self, entry_price: float, strategy_name: str
    ) -> Optional[float]:
        """
        Calculate stop-loss price for a position.

        Args:
            entry_price: Entry price of the position
            strategy_name: Name of the trading strategy

        Returns:
            Stop-loss price (None if stop-loss disabled for strategy)
        """
        # Check if strategy has stop-loss disabled
        if strategy_name in ["buy_and_hold"]:
            return None

        # Calculate stop-loss based on percentage
        stop_loss_pct = self.risk_limits.stop_loss_pct

        # Apply strategy-specific adjustments
        if strategy_name.startswith("hmm_"):
            # HMM strategies might need tighter stops due to regime changes
            stop_loss_pct *= 0.8
        elif strategy_name.startswith("ta_"):
            # Technical indicators might need looser stops for noise
            stop_loss_pct *= 1.2

        # Calculate stop-loss price (assuming long position)
        stop_loss_price = entry_price * (1 - stop_loss_pct)

        return stop_loss_price

    def should_reduce_position_size(self, current_drawdown_pct: float) -> bool:
        """
        Check if position sizes should be reduced due to drawdown.

        Args:
            current_drawdown_pct: Current portfolio drawdown percentage

        Returns:
            True if position sizes should be reduced
        """
        return current_drawdown_pct > self.risk_limits.max_drawdown_pct

    def get_drawdown_adjustment_factor(self, current_drawdown_pct: float) -> float:
        """
        Get position size adjustment factor based on current drawdown.

        Args:
            current_drawdown_pct: Current portfolio drawdown percentage

        Returns:
            Adjustment factor (1.0 = no adjustment, 0.5 = half size, etc.)
        """
        if current_drawdown_pct <= self.risk_limits.max_drawdown_pct:
            return 1.0

        # Linear reduction: at 40% drawdown, reduce to 50% size
        max_reduction_drawdown = self.risk_limits.max_drawdown_pct * 2
        if current_drawdown_pct >= max_reduction_drawdown:
            return 0.5

        # Linear interpolation between no reduction and 50% reduction
        excess_drawdown = current_drawdown_pct - self.risk_limits.max_drawdown_pct
        reduction_range = max_reduction_drawdown - self.risk_limits.max_drawdown_pct
        reduction_factor = excess_drawdown / reduction_range * 0.5

        return 1.0 - reduction_factor

    def validate_trade(
        self, position_value: float, portfolio_value: float, strategy_name: str
    ) -> Tuple[bool, str]:
        """
        Validate if a trade meets risk management criteria.

        Args:
            position_value: Dollar value of proposed position
            portfolio_value: Current portfolio value
            strategy_name: Name of the trading strategy

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check position size limit
        max_position = self.get_max_position_value(portfolio_value, strategy_name)
        if position_value > max_position:
            return (
                False,
                f"Position size ${position_value:.0f} exceeds limit ${max_position:.0f}",
            )

        # Check portfolio exposure
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        if position_pct > self.risk_limits.max_total_exposure:
            return False, f"Position would exceed maximum exposure limit"

        return True, ""


class StopLossManager:
    """
    Specialized stop-loss management system.

    Handles different types of stop-losses including trailing stops,
    volatility-based stops, and time-based exits.
    """

    def __init__(self):
        """Initialize stop-loss manager."""
        self.trailing_stops: Dict[str, float] = (
            {}
        )  # position_key -> trailing_stop_price

    def update_trailing_stop(
        self,
        position_key: str,
        current_price: float,
        entry_price: float,
        trailing_pct: float = 0.05,
    ) -> Optional[float]:
        """
        Update trailing stop-loss for a position.

        Args:
            position_key: Unique identifier for the position
            current_price: Current market price
            entry_price: Original entry price
            trailing_pct: Trailing percentage (default 5%)

        Returns:
            Updated trailing stop price (None if no trailing stop)
        """
        # Only trail if position is profitable
        if current_price <= entry_price:
            return None

        # Calculate new trailing stop
        new_trailing_stop = current_price * (1 - trailing_pct)

        # Update if this is higher than current trailing stop
        current_trailing = self.trailing_stops.get(position_key, 0)
        if new_trailing_stop > current_trailing:
            self.trailing_stops[position_key] = new_trailing_stop

        return self.trailing_stops.get(position_key)

    def should_exit_on_trail(self, position_key: str, current_price: float) -> bool:
        """
        Check if position should be exited due to trailing stop.

        Args:
            position_key: Unique identifier for the position
            current_price: Current market price

        Returns:
            True if position should be exited
        """
        trailing_stop = self.trailing_stops.get(position_key)
        if trailing_stop is None:
            return False

        return current_price <= trailing_stop

    def remove_trailing_stop(self, position_key: str) -> None:
        """Remove trailing stop for closed position."""
        self.trailing_stops.pop(position_key, None)

    def calculate_volatility_stop(
        self, price_history: list, multiplier: float = 2.0, lookback: int = 20
    ) -> float:
        """
        Calculate volatility-based stop-loss.

        Args:
            price_history: Recent price history
            multiplier: Volatility multiplier (default 2.0)
            lookback: Number of periods for volatility calculation

        Returns:
            Volatility-based stop distance
        """
        if len(price_history) < lookback:
            return 0.05  # Default 5% if insufficient data

        recent_prices = price_history[-lookback:]
        returns = [
            recent_prices[i] / recent_prices[i - 1] - 1
            for i in range(1, len(recent_prices))
        ]

        if not returns:
            return 0.05

        volatility = np.std(returns)
        return volatility * multiplier
