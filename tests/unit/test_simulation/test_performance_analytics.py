"""
Unit tests for simulation/performance_analytics.py

Tests performance analytics including Sharpe ratio, drawdown metrics,
win rate, profit factor, and trade journaling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hidden_regime.simulation.performance_analytics import (
    PerformanceAnalyzer,
    TradeJournal,
    TradeMetrics,
)
from hidden_regime.simulation.trading_engine import Trade


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    base_date = datetime(2023, 1, 1)
    trades = [
        Trade(
            symbol="AAPL",
            shares=10,
            entry_price=100.0,
            exit_price=110.0,
            entry_date=base_date,
            exit_date=base_date + timedelta(days=5),
            pnl=100.0,
            pnl_pct=10.0,
            hold_days=5,
            exit_reason="signal",
        ),
        Trade(
            symbol="AAPL",
            shares=10,
            entry_price=110.0,
            exit_price=105.0,
            entry_date=base_date + timedelta(days=6),
            exit_date=base_date + timedelta(days=10),
            pnl=-50.0,
            pnl_pct=-4.55,
            hold_days=4,
            exit_reason="stop_loss",
        ),
        Trade(
            symbol="AAPL",
            shares=10,
            entry_price=105.0,
            exit_price=115.0,
            entry_date=base_date + timedelta(days=11),
            exit_date=base_date + timedelta(days=18),
            pnl=100.0,
            pnl_pct=9.52,
            hold_days=7,
            exit_reason="signal",
        ),
    ]
    return trades


@pytest.fixture
def trade_journal(sample_trades):
    """Create trade journal with sample trades."""
    journal = TradeJournal()
    for trade in sample_trades:
        journal.add_trade(trade)
    return journal


@pytest.fixture
def performance_analyzer():
    """Create performance analyzer."""
    return PerformanceAnalyzer()


# ============================================================================
# TRADE JOURNAL TESTS (12 tests)
# ============================================================================


def test_trade_journal_add_trade():
    """Test adding trades to journal."""
    journal = TradeJournal()
    trade = Trade(
        symbol="AAPL",
        shares=10,
        entry_price=100.0,
        exit_price=110.0,
        entry_date=datetime(2023, 1, 1),
        exit_date=datetime(2023, 1, 10),
        pnl=100.0,
        pnl_pct=10.0,
        hold_days=9,
        exit_reason="signal",
    )

    journal.add_trade(trade)

    assert len(journal.get_all_trades()) == 1
    assert journal.get_all_trades()[0] == trade


def test_trade_journal_get_trades_by_strategy():
    """Test filtering trades by strategy (via DataFrame)."""
    journal = TradeJournal()

    # Add trades with different symbols (can be used as strategy proxy)
    trade1 = Trade(
        symbol="AAPL",
        shares=10,
        entry_price=100.0,
        exit_price=110.0,
        entry_date=datetime(2023, 1, 1),
        exit_date=datetime(2023, 1, 10),
        pnl=100.0,
        pnl_pct=10.0,
        hold_days=9,
        exit_reason="signal",
    )
    trade2 = Trade(
        symbol="MSFT",
        shares=5,
        entry_price=200.0,
        exit_price=210.0,
        entry_date=datetime(2023, 1, 1),
        exit_date=datetime(2023, 1, 10),
        pnl=50.0,
        pnl_pct=5.0,
        hold_days=9,
        exit_reason="signal",
    )

    journal.add_trade(trade1)
    journal.add_trade(trade2)

    # Get as DataFrame and filter
    df = journal.get_trades_df()
    aapl_trades = df[df["symbol"] == "AAPL"]

    assert len(aapl_trades) == 1
    assert aapl_trades.iloc[0]["pnl"] == 100.0


def test_trade_journal_summary_statistics(trade_journal):
    """Test trade journal summary statistics."""
    metrics = trade_journal.get_trade_metrics()

    assert isinstance(metrics, TradeMetrics)
    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2  # 2 wins, 1 loss
    assert metrics.losing_trades == 1
    assert 60 < metrics.win_rate < 70  # ~66.67%


def test_performance_analyzer_empty_trades():
    """Test performance analyzer with empty trades list."""
    analyzer = PerformanceAnalyzer()

    metrics = analyzer._calculate_trade_based_metrics([])

    assert metrics["num_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["avg_trade_pnl"] == 0.0


def test_performance_analyzer_single_trade():
    """Test performance analyzer with single trade."""
    analyzer = PerformanceAnalyzer()

    trade = Trade(
        symbol="AAPL",
        shares=10,
        entry_price=100.0,
        exit_price=110.0,
        entry_date=datetime(2023, 1, 1),
        exit_date=datetime(2023, 1, 10),
        pnl=100.0,
        pnl_pct=10.0,
        hold_days=9,
        exit_reason="signal",
    )

    metrics = analyzer._calculate_trade_based_metrics([trade])

    assert metrics["num_trades"] == 1
    assert metrics["win_rate"] == 100.0
    assert metrics["avg_trade_pnl"] == 100.0
    assert metrics["largest_win"] == 100.0


def test_calculate_total_return():
    """Test total return calculation."""
    analyzer = PerformanceAnalyzer()

    # Portfolio grows from $10,000 to $12,000
    portfolio_values = [10000, 10500, 11000, 11500, 12000]
    daily_returns = [0.0, 0.05, 0.0476, 0.0455, 0.0435]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=[],
    )

    # Total return should be (12000 - 10000) / 10000 = 0.20 (20%)
    assert metrics["total_return"] == pytest.approx(0.20, rel=0.01)
    assert metrics["total_return_pct"] == pytest.approx(20.0, rel=0.01)


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    analyzer = PerformanceAnalyzer()

    # Create synthetic returns with known properties
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.01, 252)  # 252 trading days
    portfolio_values = [10000 * (1 + r) for r in np.cumsum(daily_returns)]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns.tolist(),
        portfolio_values=portfolio_values,
        trades=[],
        risk_free_rate=0.02,
    )

    # Sharpe ratio should be calculated
    assert "sharpe_ratio" in metrics
    assert isinstance(metrics["sharpe_ratio"], (int, float))


def test_calculate_max_drawdown():
    """Test maximum drawdown calculation."""
    analyzer = PerformanceAnalyzer()

    # Portfolio with clear drawdown: 10000 -> 12000 -> 9000 -> 13000
    portfolio_values = [10000, 11000, 12000, 10000, 9000, 11000, 13000]
    daily_returns = [0.0, 0.10, 0.09, -0.167, -0.10, 0.22, 0.18]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=[],
    )

    # Max drawdown should be from peak of 12000 to trough of 9000 = 3000
    assert metrics["max_drawdown"] == pytest.approx(3000, abs=100)
    # Max drawdown % should be (12000 - 9000) / 12000 = 25%
    assert metrics["max_drawdown_pct"] == pytest.approx(25.0, rel=0.1)


def test_calculate_win_rate(sample_trades):
    """Test win rate calculation."""
    analyzer = PerformanceAnalyzer()

    metrics = analyzer._calculate_trade_based_metrics(sample_trades)

    # 2 wins out of 3 trades = 66.67%
    assert metrics["win_rate"] == pytest.approx(66.67, rel=0.01)
    assert metrics["num_trades"] == 3


def test_calculate_profit_factor(sample_trades):
    """Test profit factor calculation."""
    analyzer = PerformanceAnalyzer()

    metrics = analyzer._calculate_trade_based_metrics(sample_trades)

    # Gross profit: 100 + 100 = 200
    # Gross loss: |-50| = 50
    # Profit factor: 200 / 50 = 4.0
    assert metrics["profit_factor"] == pytest.approx(4.0, rel=0.01)


def test_calculate_average_trade_duration(sample_trades):
    """Test average trade duration calculation."""
    analyzer = PerformanceAnalyzer()

    metrics = analyzer._calculate_trade_based_metrics(sample_trades)

    # Hold days: 5, 4, 7 -> average = 5.33
    assert metrics["avg_hold_days"] == pytest.approx(5.33, rel=0.01)


def test_trade_journal_get_trades_df(trade_journal):
    """Test getting trades as DataFrame."""
    df = trade_journal.get_trades_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "pnl" in df.columns
    assert "entry_date" in df.columns
    assert "exit_date" in df.columns
    assert "hold_days" in df.columns


# ============================================================================
# INTEGRATION TESTS (8 tests)
# ============================================================================


def test_performance_analytics_with_real_trades(sample_trades):
    """Test performance analytics with realistic trade data."""
    analyzer = PerformanceAnalyzer()

    # Create realistic portfolio values
    portfolio_values = [10000, 10100, 10050, 10100, 10200]
    daily_returns = [0.0, 0.01, -0.0049, 0.0049, 0.0098]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=sample_trades,
    )

    # Should have all key metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics

    # Trade metrics should match sample_trades
    assert metrics["num_trades"] == 3


def test_performance_analytics_multi_strategy():
    """Test performance analytics with multiple strategies."""
    analyzer = PerformanceAnalyzer()

    # Trades from different strategies
    strategy1_trades = [
        Trade(
            symbol="AAPL_strategy1",
            shares=10,
            entry_price=100.0,
            exit_price=110.0,
            entry_date=datetime(2023, 1, 1),
            exit_date=datetime(2023, 1, 10),
            pnl=100.0,
            pnl_pct=10.0,
            hold_days=9,
            exit_reason="signal",
        ),
    ]

    strategy2_trades = [
        Trade(
            symbol="MSFT_strategy2",
            shares=5,
            entry_price=200.0,
            exit_price=220.0,
            entry_date=datetime(2023, 1, 1),
            exit_date=datetime(2023, 1, 10),
            pnl=100.0,
            pnl_pct=10.0,
            hold_days=9,
            exit_reason="signal",
        ),
    ]

    all_trades = strategy1_trades + strategy2_trades

    portfolio_values = [10000, 11000, 12000]
    daily_returns = [0.0, 0.10, 0.09]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=all_trades,
    )

    assert metrics["num_trades"] == 2
    assert metrics["win_rate"] == 100.0


def test_performance_analytics_handles_losses():
    """Test performance analytics with losing scenario."""
    analyzer = PerformanceAnalyzer()

    # All losing trades
    losing_trades = [
        Trade(
            symbol="AAPL",
            shares=10,
            entry_price=100.0,
            exit_price=90.0,
            entry_date=datetime(2023, 1, 1),
            exit_date=datetime(2023, 1, 10),
            pnl=-100.0,
            pnl_pct=-10.0,
            hold_days=9,
            exit_reason="stop_loss",
        ),
        Trade(
            symbol="AAPL",
            shares=10,
            entry_price=90.0,
            exit_price=85.0,
            entry_date=datetime(2023, 1, 11),
            exit_date=datetime(2023, 1, 20),
            pnl=-50.0,
            pnl_pct=-5.56,
            hold_days=9,
            exit_reason="stop_loss",
        ),
    ]

    portfolio_values = [10000, 9900, 9850]
    daily_returns = [0.0, -0.01, -0.0051]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=losing_trades,
    )

    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0  # No wins
    assert metrics["total_return"] < 0


def test_performance_analytics_calculates_metrics_correctly():
    """Test that all performance metrics are calculated correctly."""
    analyzer = PerformanceAnalyzer()

    # Known data for validation
    portfolio_values = [10000, 10100, 10200, 10300, 10400]
    daily_returns = [0.0, 0.01, 0.0099, 0.0098, 0.0097]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=[],
    )

    # Verify key calculations
    assert metrics["initial_value"] == 10000
    assert metrics["final_value"] == 10400
    assert metrics["peak_value"] == 10400
    assert metrics["trading_days"] == 5

    # Positive return metrics
    assert metrics["total_return"] > 0
    assert metrics["annualized_return"] > 0


def test_performance_analytics_time_weighted_returns():
    """Test time-weighted return calculations."""
    analyzer = PerformanceAnalyzer()

    # Simulate 1 year of trading (252 days)
    np.random.seed(42)
    daily_returns = np.random.normal(0.0008, 0.01, 252).tolist()
    initial_value = 10000
    portfolio_values = [initial_value]
    for ret in daily_returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=[],
    )

    # Annualized return should be calculated
    assert "annualized_return" in metrics
    assert isinstance(metrics["annualized_return"], (int, float))


def test_performance_analytics_benchmark_comparison():
    """Test performance metrics can be used for benchmark comparison."""
    analyzer = PerformanceAnalyzer()

    # Strategy performance
    strategy_values = [10000, 10500, 11000, 11500, 12000]
    strategy_returns = [0.0, 0.05, 0.0476, 0.0455, 0.0435]

    strategy_metrics = analyzer.calculate_metrics(
        daily_returns=strategy_returns,
        portfolio_values=strategy_values,
        trades=[],
    )

    # Benchmark performance (worse than strategy)
    benchmark_values = [10000, 10200, 10400, 10600, 10800]
    benchmark_returns = [0.0, 0.02, 0.0196, 0.0192, 0.0189]

    benchmark_metrics = analyzer.calculate_metrics(
        daily_returns=benchmark_returns,
        portfolio_values=benchmark_values,
        trades=[],
    )

    # Strategy should outperform benchmark
    assert strategy_metrics["total_return"] > benchmark_metrics["total_return"]


def test_performance_analytics_export_to_dataframe(trade_journal):
    """Test exporting performance data to DataFrame."""
    df = trade_journal.get_trades_df()

    # Should be exportable DataFrame
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Should have all trade data
    assert "pnl" in df.columns
    assert "pnl_pct" in df.columns
    assert "entry_date" in df.columns
    assert "exit_date" in df.columns


def test_performance_analytics_rolling_metrics(trade_journal):
    """Test calculating rolling performance metrics."""
    df = trade_journal.get_trades_df()

    # Get monthly performance
    monthly_perf = trade_journal.get_monthly_performance()

    # Should return DataFrame (may be empty if trades within same month)
    assert isinstance(monthly_perf, pd.DataFrame)


def test_trade_metrics_dataclass():
    """Test TradeMetrics dataclass structure."""
    metrics = TradeMetrics(
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=60.0,
        avg_win=150.0,
        avg_loss=-75.0,
        largest_win=300.0,
        largest_loss=-200.0,
        profit_factor=2.0,
        avg_hold_days=7.5,
    )

    assert metrics.total_trades == 10
    assert metrics.win_rate == 60.0
    assert metrics.profit_factor == 2.0


def test_drawdown_duration_calculation():
    """Test maximum drawdown duration calculation."""
    analyzer = PerformanceAnalyzer()

    # Portfolio: Peak -> Drawdown (3 days) -> Recovery -> Peak
    portfolio_values = np.array([10000, 11000, 12000, 11000, 10000, 9500, 11000, 12500])

    peaks = np.maximum.accumulate(portfolio_values)
    duration = analyzer._calculate_max_drawdown_duration(portfolio_values, peaks)

    # Longest drawdown should be from index 2 (peak 12000) to index 5 (trough 9500) = 3 days
    assert duration >= 3


def test_sortino_ratio_calculation():
    """Test Sortino ratio calculation (downside deviation)."""
    analyzer = PerformanceAnalyzer()

    # Mix of positive and negative returns
    daily_returns = [0.01, 0.02, -0.015, 0.01, -0.01, 0.015, 0.01]
    portfolio_values = [10000 * (1 + r) for r in np.cumsum(daily_returns)]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=[],
    )

    # Sortino ratio should be calculated
    assert "sortino_ratio" in metrics
    # Sortino should typically be higher than Sharpe (only penalizes downside)
    # But with small sample, relationship may vary
    assert isinstance(metrics["sortino_ratio"], (int, float))


def test_positive_day_percentage():
    """Test calculation of positive trading days percentage."""
    analyzer = PerformanceAnalyzer()

    # 7 days, 5 positive, 2 negative
    daily_returns = [0.01, 0.02, -0.01, 0.01, -0.005, 0.015, 0.01]
    portfolio_values = [10000 * (1 + r) for r in np.cumsum(daily_returns)]

    metrics = analyzer.calculate_metrics(
        daily_returns=daily_returns,
        portfolio_values=portfolio_values,
        trades=[],
    )

    # 5 positive days out of 7 = ~71.4%
    assert metrics["positive_day_pct"] == pytest.approx(71.4, rel=0.1)
    assert metrics["positive_days"] == 5
    assert metrics["negative_days"] == 2
