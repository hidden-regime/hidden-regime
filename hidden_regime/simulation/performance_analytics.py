"""
Performance analytics for trading simulation.

Provides comprehensive performance metrics, trade journaling, and
capital-based analytics as specified in simulation.md.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeMetrics:
    """Trade-level performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_hold_days: float


class TradeJournal:
    """
    Comprehensive trade journaling system.

    Tracks all trades with detailed entry/exit information for analysis.
    """

    def __init__(self):
        """Initialize trade journal."""
        self.trades: List = []

    def add_trade(self, trade) -> None:
        """Add a trade to the journal."""
        self.trades.append(trade)

    def get_all_trades(self) -> List:
        """Get all recorded trades."""
        return self.trades.copy()

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as a pandas DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        trade_data = []
        for trade in self.trades:
            trade_data.append(
                {
                    "symbol": trade.symbol,
                    "shares": trade.shares,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "entry_date": trade.entry_date,
                    "exit_date": trade.exit_date,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "hold_days": trade.hold_days,
                    "exit_reason": trade.exit_reason,
                }
            )

        return pd.DataFrame(trade_data)

    def get_trade_metrics(self) -> TradeMetrics:
        """Calculate comprehensive trade-level metrics."""
        if not self.trades:
            return TradeMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                profit_factor=0.0,
                avg_hold_days=0.0,
            )

        # Separate winning and losing trades
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        # Calculate metrics
        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

        # Win/Loss statistics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0

        # Profit factor (gross profit / gross loss)
        gross_profit = sum([t.pnl for t in winning_trades])
        gross_loss = abs(sum([t.pnl for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average holding period
        avg_hold_days = np.mean([t.hold_days for t in self.trades])

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days,
        )

    def get_monthly_performance(self) -> pd.DataFrame:
        """Get monthly performance breakdown."""
        if not self.trades:
            return pd.DataFrame()

        df = self.get_trades_df()
        df["exit_month"] = pd.to_datetime(df["exit_date"]).dt.to_period("M")

        monthly_stats = (
            df.groupby("exit_month")
            .agg(
                {
                    "pnl": ["sum", "count", "mean"],
                    "pnl_pct": "mean",
                    "hold_days": "mean",
                }
            )
            .round(2)
        )

        return monthly_stats


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system.

    Calculates investment-grade performance metrics including risk-adjusted
    returns, drawdown analysis, and capital-based statistics.
    """

    def calculate_metrics(
        self,
        daily_returns: List[float],
        portfolio_values: List[float],
        trades: List,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            daily_returns: List of daily portfolio returns
            portfolio_values: List of daily portfolio values
            trades: List of completed trades
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            Dictionary with comprehensive performance metrics
        """
        if not daily_returns or not portfolio_values:
            return self._empty_metrics()

        # Convert to numpy arrays for calculations
        returns = np.array(daily_returns)
        values = np.array(portfolio_values)

        # Basic return metrics
        total_return = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1

        # Risk metrics
        return_std = np.std(returns) if len(returns) > 1 else 0
        annualized_volatility = return_std * np.sqrt(252)

        # Sharpe ratio
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = (
            excess_return / annualized_volatility if annualized_volatility > 0 else 0
        )

        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = (
            np.std(downside_returns) if len(downside_returns) > 1 else return_std
        )
        annualized_downside_vol = downside_std * np.sqrt(252)
        sortino_ratio = (
            excess_return / annualized_downside_vol
            if annualized_downside_vol > 0
            else 0
        )

        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(values)

        # Trade-based metrics
        trade_metrics = self._calculate_trade_based_metrics(trades) if trades else {}

        # Combine all metrics
        return {
            # Return metrics
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            # Risk metrics
            "volatility": return_std,
            "annualized_volatility": annualized_volatility,
            "annualized_volatility_pct": annualized_volatility * 100,
            # Risk-adjusted metrics
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            # Drawdown metrics
            **drawdown_metrics,
            # Trade metrics
            **trade_metrics,
            # Portfolio metrics
            "initial_value": values[0],
            "final_value": values[-1],
            "peak_value": np.max(values),
            "trading_days": trading_days,
            # Additional statistics
            "best_day": np.max(returns) if len(returns) > 0 else 0,
            "worst_day": np.min(returns) if len(returns) > 0 else 0,
            "positive_days": np.sum(returns > 0),
            "negative_days": np.sum(returns < 0),
            "positive_day_pct": (
                (np.sum(returns > 0) / len(returns)) * 100 if returns.size > 0 else 0
            ),
        }

    def _calculate_drawdown_metrics(
        self, portfolio_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(portfolio_values) == 0:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
                "current_drawdown": 0.0,
                "current_drawdown_pct": 0.0,
            }

        # Calculate running maximum (peak values)
        peaks = np.maximum.accumulate(portfolio_values)

        # Calculate drawdowns
        drawdowns = peaks - portfolio_values
        drawdown_pcts = drawdowns / peaks

        # Maximum drawdown
        max_drawdown = np.max(drawdowns)
        max_drawdown_pct = np.max(drawdown_pcts)

        # Current drawdown
        current_drawdown = drawdowns[-1]
        current_drawdown_pct = drawdown_pcts[-1]

        # Maximum drawdown duration
        max_dd_duration = self._calculate_max_drawdown_duration(portfolio_values, peaks)

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct * 100,
            "max_drawdown_duration": max_dd_duration,
            "current_drawdown": current_drawdown,
            "current_drawdown_pct": current_drawdown_pct * 100,
        }

    def _calculate_max_drawdown_duration(
        self, portfolio_values: np.ndarray, peaks: np.ndarray
    ) -> int:
        """Calculate the maximum drawdown duration in days."""
        if len(portfolio_values) == 0:
            return 0

        max_duration = 0
        current_duration = 0

        for i in range(len(portfolio_values)):
            if portfolio_values[i] < peaks[i]:
                # In drawdown
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                # New peak, reset duration
                current_duration = 0

        return max_duration

    def _calculate_trade_based_metrics(self, trades: List) -> Dict[str, Any]:
        """Calculate metrics based on individual trades."""
        if not trades:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_hold_days": 0.0,
            }

        # Calculate trade statistics
        trade_pnls = [trade.pnl for trade in trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]

        # Basic trade metrics
        num_trades = len(trades)
        win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0

        # P&L metrics
        avg_trade_pnl = np.mean(trade_pnls)
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0

        # Profit factor
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Holding periods
        hold_days = [trade.hold_days for trade in trades]
        avg_hold_days = np.mean(hold_days)

        return {
            "num_trades": num_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_trade_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "avg_hold_days": avg_hold_days,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for failed calculations."""
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
