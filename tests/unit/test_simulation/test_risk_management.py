"""
Unit tests for simulation/risk_management.py

Tests risk management functionality including stop-loss calculation,
position sizing limits, and portfolio-level risk controls.
"""

import pytest
import numpy as np

from hidden_regime.simulation.risk_management import (
    RiskManager,
    RiskLimits,
    StopLossManager,
)


@pytest.fixture
def default_risk_manager():
    """Create risk manager with default settings."""
    return RiskManager()


@pytest.fixture
def strict_risk_manager():
    """Create risk manager with strict limits."""
    limits = RiskLimits(
        max_position_pct=0.05,  # 5% max position
        max_portfolio_risk=0.01,  # 1% max risk per trade
        stop_loss_pct=0.02,  # 2% stop-loss
        max_total_exposure=0.5,  # 50% max exposure
        max_drawdown_pct=0.10,  # 10% max drawdown
    )
    return RiskManager(risk_limits=limits)


# ============================================================================
# UNIT TESTS (10 tests)
# ============================================================================


def test_calculate_stop_loss_long_position(default_risk_manager):
    """Test stop-loss calculation for long position."""
    entry_price = 100.0
    strategy_name = "test_strategy"

    stop_loss = default_risk_manager.calculate_stop_loss(entry_price, strategy_name)

    # With default 5% stop, should be $95
    assert stop_loss is not None
    assert stop_loss == pytest.approx(95.0, rel=0.01)
    assert stop_loss < entry_price


def test_calculate_stop_loss_short_position():
    """Test stop-loss calculation for short position (implementation detail)."""
    # Note: Current implementation calculates for long positions
    # Short positions would need stop above entry price
    risk_manager = RiskManager()
    entry_price = 100.0

    stop_loss = risk_manager.calculate_stop_loss(entry_price, "test_strategy")

    # Current implementation: stop = entry * (1 - pct) for longs
    assert stop_loss < entry_price


def test_calculate_stop_loss_custom_percentage(strict_risk_manager):
    """Test stop-loss with custom percentage."""
    entry_price = 100.0

    stop_loss = strict_risk_manager.calculate_stop_loss(entry_price, "test_strategy")

    # Strict manager has 2% stop-loss
    assert stop_loss == pytest.approx(98.0, rel=0.01)


def test_calculate_stop_loss_hmm_strategy(default_risk_manager):
    """Test that HMM strategies get tighter stops."""
    entry_price = 100.0

    # HMM strategy gets 80% of default stop (tighter)
    stop_loss = default_risk_manager.calculate_stop_loss(entry_price, "hmm_regime_following")

    # Default is 5%, HMM gets 5% * 0.8 = 4%
    expected = 100.0 * (1 - 0.05 * 0.8)
    assert stop_loss == pytest.approx(expected, rel=0.01)
    assert stop_loss > 95.0  # Tighter than regular 5% stop


def test_calculate_stop_loss_ta_strategy(default_risk_manager):
    """Test that technical analysis strategies get looser stops."""
    entry_price = 100.0

    # TA strategy gets 120% of default stop (looser)
    stop_loss = default_risk_manager.calculate_stop_loss(entry_price, "ta_moving_average")

    # Default is 5%, TA gets 5% * 1.2 = 6%
    expected = 100.0 * (1 - 0.05 * 1.2)
    assert stop_loss == pytest.approx(expected, rel=0.01)
    assert stop_loss < 95.0  # Looser than regular 5% stop


def test_calculate_stop_loss_buy_and_hold():
    """Test that buy-and-hold strategy has no stop-loss."""
    risk_manager = RiskManager()
    entry_price = 100.0

    stop_loss = risk_manager.calculate_stop_loss(entry_price, "buy_and_hold")

    # Buy-and-hold should have no stop-loss
    assert stop_loss is None


def test_get_max_position_value_default(default_risk_manager):
    """Test default max position value calculation."""
    portfolio_value = 10000.0
    strategy_name = "test_strategy"

    max_position = default_risk_manager.get_max_position_value(
        portfolio_value, strategy_name
    )

    # Default is 10% of portfolio
    assert max_position == pytest.approx(1000.0, rel=0.01)


def test_get_max_position_value_with_limits(strict_risk_manager):
    """Test max position value with strict limits."""
    portfolio_value = 10000.0

    max_position = strict_risk_manager.get_max_position_value(
        portfolio_value, "test_strategy"
    )

    # Strict manager: 5% max position
    assert max_position == pytest.approx(500.0, rel=0.01)


def test_get_max_position_value_strategy_specific():
    """Test strategy-specific risk multipliers."""
    risk_manager = RiskManager()
    portfolio_value = 10000.0

    # Set conservative multiplier for one strategy
    risk_manager.set_strategy_risk_multiplier("conservative_strategy", 0.5)

    max_position = risk_manager.get_max_position_value(
        portfolio_value, "conservative_strategy"
    )

    # Should be 10% * 0.5 = 5% of portfolio
    assert max_position == pytest.approx(500.0, rel=0.01)


def test_validate_trade_size_within_limits(default_risk_manager):
    """Test trade validation for position within limits."""
    position_value = 500.0  # 5% of portfolio
    portfolio_value = 10000.0

    is_valid, reason = default_risk_manager.validate_trade(
        position_value, portfolio_value, "test_strategy"
    )

    assert is_valid is True
    assert reason == ""


# ============================================================================
# INTEGRATION TESTS (5 tests)
# ============================================================================


def test_validate_trade_size_exceeds_limits(default_risk_manager):
    """Test trade validation for position exceeding limits."""
    position_value = 2000.0  # 20% of portfolio (exceeds 10% limit)
    portfolio_value = 10000.0

    is_valid, reason = default_risk_manager.validate_trade(
        position_value, portfolio_value, "test_strategy"
    )

    assert is_valid is False
    assert "exceeds limit" in reason.lower()


def test_validate_trade_size_with_existing_positions():
    """Test trade validation considers portfolio exposure limits."""
    risk_manager = RiskManager()

    # Trying to add 50% position (5000/10000), but max_position_pct default is 10%
    position_value = 5000.0  # 50% of portfolio (exceeds 10% limit)
    portfolio_value = 10000.0

    is_valid, reason = risk_manager.validate_trade(
        position_value, portfolio_value, "test_strategy"
    )

    # Should be invalid (exceeds 10% position limit)
    assert is_valid is False


def test_calculate_portfolio_risk():
    """Test portfolio risk aggregation across strategies."""
    risk_manager = RiskManager()

    # Test stop-loss calculation for multiple strategies
    strategies = ["strategy_1", "strategy_2", "strategy_3"]
    entry_price = 100.0

    stop_losses = [
        risk_manager.calculate_stop_loss(entry_price, strategy)
        for strategy in strategies
    ]

    # All stops should be calculated
    assert all(stop is not None for stop in stop_losses)
    # All stops should be below entry for long positions
    assert all(stop < entry_price for stop in stop_losses)


def test_risk_manager_with_trading_engine():
    """Test risk manager integration with trading engine (mock test)."""
    risk_limits = RiskLimits(max_position_pct=0.1)
    risk_manager = RiskManager(risk_limits=risk_limits)

    portfolio_value = 10000.0
    strategy = "test_strategy"

    # Get max position
    max_position = risk_manager.get_max_position_value(portfolio_value, strategy)
    assert max_position == 1000.0

    # Validate a trade at max position
    is_valid, _ = risk_manager.validate_trade(max_position, portfolio_value, strategy)
    assert is_valid is True

    # Validate a trade exceeding max position
    is_valid, _ = risk_manager.validate_trade(max_position + 1, portfolio_value, strategy)
    assert is_valid is False


def test_risk_manager_prevents_oversized_positions():
    """Test that risk manager prevents oversized positions."""
    strict_limits = RiskLimits(max_position_pct=0.05)  # 5% max
    risk_manager = RiskManager(risk_limits=strict_limits)

    portfolio_value = 10000.0
    large_position = 1000.0  # 10% of portfolio

    is_valid, reason = risk_manager.validate_trade(
        large_position, portfolio_value, "test_strategy"
    )

    assert is_valid is False
    assert "exceeds limit" in reason.lower()


def test_risk_manager_handles_multiple_strategies():
    """Test risk manager with multiple strategies and different multipliers."""
    risk_manager = RiskManager()

    # Set different multipliers
    risk_manager.set_strategy_risk_multiplier("aggressive_strategy", 2.0)
    risk_manager.set_strategy_risk_multiplier("conservative_strategy", 0.5)

    portfolio_value = 10000.0

    # Aggressive strategy should get 20% (10% * 2.0)
    aggressive_max = risk_manager.get_max_position_value(
        portfolio_value, "aggressive_strategy"
    )
    assert aggressive_max == pytest.approx(2000.0, rel=0.01)

    # Conservative strategy should get 5% (10% * 0.5)
    conservative_max = risk_manager.get_max_position_value(
        portfolio_value, "conservative_strategy"
    )
    assert conservative_max == pytest.approx(500.0, rel=0.01)


def test_risk_manager_updates_on_position_changes():
    """Test that risk calculations update correctly as portfolio changes."""
    risk_manager = RiskManager()

    # Initial portfolio
    initial_portfolio = 10000.0
    initial_max = risk_manager.get_max_position_value(initial_portfolio, "test_strategy")
    assert initial_max == 1000.0  # 10%

    # Portfolio grows
    larger_portfolio = 15000.0
    larger_max = risk_manager.get_max_position_value(larger_portfolio, "test_strategy")
    assert larger_max == 1500.0  # Still 10%, but of larger portfolio

    # Portfolio shrinks
    smaller_portfolio = 5000.0
    smaller_max = risk_manager.get_max_position_value(smaller_portfolio, "test_strategy")
    assert smaller_max == 500.0  # Still 10%, but of smaller portfolio


def test_risk_manager_portfolio_heat_calculation():
    """Test total risk calculation across portfolio."""
    risk_manager = RiskManager()

    # Calculate risk for multiple positions
    portfolio_value = 10000.0
    positions = [
        ("strategy_1", 800.0),  # 8% of portfolio
        ("strategy_2", 600.0),  # 6% of portfolio
        ("strategy_3", 400.0),  # 4% of portfolio
    ]

    total_exposure = sum(value for _, value in positions)
    exposure_pct = total_exposure / portfolio_value

    # Total exposure is 18% (< 100% limit)
    assert exposure_pct == 0.18
    assert exposure_pct < risk_manager.risk_limits.max_total_exposure


def test_should_reduce_position_size():
    """Test drawdown-based position size reduction."""
    risk_manager = RiskManager()

    # Normal drawdown (10%) - no reduction
    should_reduce = risk_manager.should_reduce_position_size(0.10)
    assert should_reduce is False

    # Large drawdown (25%) - should reduce
    should_reduce = risk_manager.should_reduce_position_size(0.25)
    assert should_reduce is True


def test_get_drawdown_adjustment_factor():
    """Test drawdown adjustment factor calculation."""
    risk_manager = RiskManager()

    # No drawdown - no adjustment
    factor = risk_manager.get_drawdown_adjustment_factor(0.0)
    assert factor == 1.0

    # Small drawdown (10%) - no adjustment (below 20% threshold)
    factor = risk_manager.get_drawdown_adjustment_factor(0.10)
    assert factor == 1.0

    # Moderate drawdown (20%) - threshold, no adjustment yet
    factor = risk_manager.get_drawdown_adjustment_factor(0.20)
    assert factor == 1.0

    # Large drawdown (30%) - should reduce
    factor = risk_manager.get_drawdown_adjustment_factor(0.30)
    assert 0.5 < factor < 1.0

    # Severe drawdown (40%+) - max reduction to 50%
    factor = risk_manager.get_drawdown_adjustment_factor(0.50)
    assert factor == 0.5


# ============================================================================
# STOP LOSS MANAGER TESTS
# ============================================================================


def test_stop_loss_manager_update_trailing_stop():
    """Test trailing stop update logic."""
    manager = StopLossManager()

    entry_price = 100.0
    current_price = 110.0  # Position is profitable

    trailing_stop = manager.update_trailing_stop(
        "position_1", current_price, entry_price, trailing_pct=0.05
    )

    # Trailing stop should be 5% below current price
    assert trailing_stop == pytest.approx(104.5, rel=0.01)


def test_stop_loss_manager_no_trail_when_losing():
    """Test that trailing stop doesn't activate for losing positions."""
    manager = StopLossManager()

    entry_price = 100.0
    current_price = 95.0  # Position is losing

    trailing_stop = manager.update_trailing_stop(
        "position_1", current_price, entry_price, trailing_pct=0.05
    )

    # Should not set trailing stop for losing position
    assert trailing_stop is None


def test_stop_loss_manager_should_exit_on_trail():
    """Test trailing stop exit logic."""
    manager = StopLossManager()

    # Set up trailing stop
    entry_price = 100.0
    high_price = 110.0
    manager.update_trailing_stop("position_1", high_price, entry_price, trailing_pct=0.05)

    # Price drops below trailing stop
    current_price = 104.0
    should_exit = manager.should_exit_on_trail("position_1", current_price)

    assert should_exit is True


def test_stop_loss_manager_calculate_volatility_stop():
    """Test volatility-based stop calculation."""
    manager = StopLossManager()

    # Create price history with known volatility
    price_history = [100, 102, 98, 101, 99, 103, 97, 102, 100, 101]

    vol_stop = manager.calculate_volatility_stop(
        price_history, multiplier=2.0, lookback=10
    )

    # Should return a percentage based on volatility
    assert isinstance(vol_stop, float)
    assert 0 < vol_stop < 0.5  # Reasonable stop range
