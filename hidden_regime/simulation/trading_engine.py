"""
Core trading simulation engine with capital-based tracking.

Provides realistic trading simulation using actual capital allocation,
share tracking, and proper buy/sell timing as specified in simulation.md.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .performance_analytics import PerformanceAnalyzer, TradeJournal
from .risk_management import RiskManager


class SignalType(Enum):
    """Trading signal types."""

    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    entry_value: float = None

    def __post_init__(self):
        if self.entry_value is None:
            self.entry_value = abs(self.shares) * self.entry_price


@dataclass
class Trade:
    """Represents a completed trade."""

    symbol: str
    shares: int
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_reason: str  # 'signal', 'stop_loss', 'end_of_data'


class TradingSimulationEngine:
    """
    Core trading simulation engine with capital-based tracking.

    Simulates realistic trading conditions:
    - Sell at current day's Close price
    - Buy at next day's Open price
    - Actual share and capital tracking
    - Risk management with stop-losses
    - Comprehensive trade journaling
    """

    def __init__(
        self,
        initial_capital: float = None,
        default_shares: int = 100,
        transaction_cost: float = 0.0,
        enable_shorting: bool = True,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Initialize trading simulation engine.

        Args:
            initial_capital: Starting capital (if None, calculated from default_shares * first price)
            default_shares: Default number of shares to trade if no capital specified
            transaction_cost: Cost per trade (flat fee or percentage)
            enable_shorting: Whether to allow short selling
            risk_manager: Risk management system for stop-losses and position limits
        """
        self.initial_capital = initial_capital
        self.default_shares = default_shares
        self.transaction_cost = transaction_cost
        self.enable_shorting = enable_shorting
        self.risk_manager = risk_manager or RiskManager()

        # Portfolio state
        self.cash = 0.0
        self.positions: Dict[str, Position] = {}
        self.trade_journal = TradeJournal()
        self.performance_analyzer = PerformanceAnalyzer()

        # Simulation history
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []

        # Simulation metadata
        self.simulation_start_date = None
        self.simulation_end_date = None

    def initialize_simulation(
        self, price_data: pd.DataFrame, price_column: str = "close"
    ) -> bool:
        """
        Initialize simulation with price data.

        Args:
            price_data: DataFrame with OHLC price data
            price_column: Column to use for initial capital calculation

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Validate required columns
            required_columns = ["open", "close"]
            if not all(col in price_data.columns for col in required_columns):
                missing = [
                    col for col in required_columns if col not in price_data.columns
                ]
                raise ValueError(f"Missing required columns: {missing}")

            # Set initial capital if not provided
            if self.initial_capital is None:
                first_price = price_data[price_column].iloc[0]
                self.initial_capital = self.default_shares * first_price

            self.cash = self.initial_capital

            # Set simulation date range
            self.simulation_start_date = price_data.index[0]
            self.simulation_end_date = price_data.index[-1]

            # Initialize portfolio history
            self.portfolio_history = []
            self.daily_returns = []

            return True

        except Exception as e:
            print(f"❌ Failed to initialize simulation: {e}")
            return False

    def run_simulation(
        self, price_data: pd.DataFrame, signals: pd.DataFrame, symbol: str = "ASSET"
    ) -> Dict[str, Any]:
        """
        Run complete trading simulation.

        Args:
            price_data: DataFrame with OHLC price data
            signals: DataFrame with trading signals for each strategy
            symbol: Asset symbol being traded

        Returns:
            Dictionary with simulation results and performance metrics
        """
        if not self.initialize_simulation(price_data):
            return self._create_failed_simulation_result()

        # Process each trading day
        for i, (date, price_row) in enumerate(price_data.iterrows()):
            # Get signals for this date
            if date in signals.index:
                date_signals = signals.loc[date]

                # Process each signal generator
                for signal_name, signal_value in date_signals.items():
                    self._process_signal(
                        date=date,
                        signal_name=signal_name,
                        signal_value=signal_value,
                        price_row=price_row,
                        symbol=symbol,
                        next_open=self._get_next_open_price(price_data, i),
                    )

            # Check stop-losses on existing positions
            self._check_stop_losses(date, price_row, symbol)

            # Record portfolio state
            self._record_portfolio_state(date, price_row, symbol)

        # Close any remaining positions at end of simulation
        self._close_all_positions(
            self.simulation_end_date,
            price_data.iloc[-1],
            symbol,
            exit_reason="end_of_data",
        )

        # Calculate final performance metrics
        return self._calculate_simulation_results(symbol)

    def _process_signal(
        self,
        date: datetime,
        signal_name: str,
        signal_value: float,
        price_row: pd.Series,
        symbol: str,
        next_open: Optional[float],
    ) -> None:
        """Process a trading signal for a specific strategy."""

        # Convert signal to action
        if abs(signal_value) < 0.1:  # Treat small values as hold
            return

        signal_type = SignalType.BUY if signal_value > 0 else SignalType.SELL

        # Check if we have an existing position for this strategy
        position_key = f"{symbol}_{signal_name}"

        if signal_type == SignalType.SELL and position_key in self.positions:
            # Close existing position at today's close
            self._close_position(
                position_key=position_key,
                exit_date=date,
                exit_price=price_row["close"],
                exit_reason="signal",
            )

        elif signal_type == SignalType.BUY and next_open is not None:
            # Open new position at next day's open
            shares_to_buy = self._calculate_position_size(next_open, signal_name)

            if shares_to_buy > 0:
                self._open_position(
                    position_key=position_key,
                    symbol=symbol,
                    shares=shares_to_buy,
                    entry_price=next_open,
                    entry_date=date,
                    signal_name=signal_name,
                )

    def _calculate_position_size(self, price: float, signal_name: str) -> int:
        """Calculate number of shares to buy based on available capital and risk management."""

        # Apply risk management limits
        max_position_value = self.risk_manager.get_max_position_value(
            self.cash + self._get_total_position_value(), signal_name
        )

        # Calculate affordable shares
        max_affordable_shares = int(self.cash / price)
        max_risk_shares = int(max_position_value / price)

        # Take the minimum of affordable and risk-allowed
        shares = min(max_affordable_shares, max_risk_shares)

        # Ensure we have enough cash after transaction costs
        total_cost = shares * price + self.transaction_cost
        if total_cost > self.cash:
            shares = max(0, int((self.cash - self.transaction_cost) / price))

        return shares

    def _open_position(
        self,
        position_key: str,
        symbol: str,
        shares: int,
        entry_price: float,
        entry_date: datetime,
        signal_name: str,
    ) -> None:
        """Open a new trading position."""

        # Calculate position value and costs
        position_value = shares * entry_price
        total_cost = position_value + self.transaction_cost

        # Check if we have sufficient cash
        if total_cost > self.cash:
            return  # Cannot afford this position

        # Calculate stop-loss price if enabled
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, signal_name)

        # Create position
        position = Position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            entry_date=entry_date,
            stop_loss=stop_loss,
        )

        # Update portfolio
        self.positions[position_key] = position
        self.cash -= total_cost

        # Log trade opening
        print(
            f"📈 Opened position: {shares} shares of {symbol} at ${entry_price:.2f} ({signal_name})"
        )

    def _close_position(
        self,
        position_key: str,
        exit_date: datetime,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Close an existing trading position."""

        if position_key not in self.positions:
            return

        position = self.positions[position_key]

        # Calculate trade results
        exit_value = position.shares * exit_price
        pnl = exit_value - position.entry_value
        pnl_pct = (pnl / position.entry_value) * 100
        hold_days = (exit_date - position.entry_date).days

        # Update cash (subtract transaction cost)
        self.cash += exit_value - self.transaction_cost

        # Create trade record
        trade = Trade(
            symbol=position.symbol,
            shares=position.shares,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_date=position.entry_date,
            exit_date=exit_date,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_days=hold_days,
            exit_reason=exit_reason,
        )

        # Record in trade journal
        self.trade_journal.add_trade(trade)

        # Remove position
        del self.positions[position_key]

        # Log trade closing
        print(
            f"📉 Closed position: {position.shares} shares of {position.symbol} at ${exit_price:.2f} "
            f"(P&L: ${pnl:.2f}, {pnl_pct:+.1f}%, {exit_reason})"
        )

    def _check_stop_losses(
        self, date: datetime, price_row: pd.Series, symbol: str
    ) -> None:
        """Check if any positions should be stopped out."""

        positions_to_close = []
        current_price = price_row["close"]

        for position_key, position in self.positions.items():
            if position.stop_loss is not None:
                # Check if stop-loss is triggered
                if (position.shares > 0 and current_price <= position.stop_loss) or (
                    position.shares < 0 and current_price >= position.stop_loss
                ):
                    positions_to_close.append(position_key)

        # Close stopped out positions
        for position_key in positions_to_close:
            self._close_position(
                position_key=position_key,
                exit_date=date,
                exit_price=current_price,
                exit_reason="stop_loss",
            )

    def _close_all_positions(
        self, exit_date: datetime, price_row: pd.Series, symbol: str, exit_reason: str
    ) -> None:
        """Close all remaining positions at end of simulation."""

        exit_price = price_row["close"]
        position_keys = list(self.positions.keys())

        for position_key in position_keys:
            self._close_position(
                position_key=position_key,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason=exit_reason,
            )

    def _get_next_open_price(
        self, price_data: pd.DataFrame, current_index: int
    ) -> Optional[float]:
        """Get next day's opening price for position entry."""

        if current_index + 1 < len(price_data):
            return price_data.iloc[current_index + 1]["open"]
        return None

    def _get_total_position_value(self) -> float:
        """Calculate total value of all current positions."""
        return sum(pos.entry_value for pos in self.positions.values())

    def _record_portfolio_state(
        self, date: datetime, price_row: pd.Series, symbol: str
    ) -> None:
        """Record daily portfolio state for performance tracking."""

        # Calculate current position values at market price
        position_value = 0.0
        for position in self.positions.values():
            current_price = price_row["close"]
            position_value += position.shares * current_price

        total_portfolio_value = self.cash + position_value
        daily_return = 0.0

        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]["total_value"]
            daily_return = (
                (total_portfolio_value - prev_value) / prev_value
                if prev_value > 0
                else 0.0
            )

        # Record state
        portfolio_state = {
            "date": date,
            "cash": self.cash,
            "position_value": position_value,
            "total_value": total_portfolio_value,
            "daily_return": daily_return,
            "num_positions": len(self.positions),
        }

        self.portfolio_history.append(portfolio_state)
        self.daily_returns.append(daily_return)

    def _calculate_simulation_results(self, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive simulation results and performance metrics."""

        if not self.portfolio_history:
            return self._create_failed_simulation_result()

        # Basic portfolio metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_history[-1]["total_value"]
        total_return = (final_value - initial_value) / initial_value

        # Time-based metrics
        start_date = self.simulation_start_date
        end_date = self.simulation_end_date
        trading_days = len(self.portfolio_history)

        # Use performance analyzer for advanced metrics
        performance_metrics = self.performance_analyzer.calculate_metrics(
            daily_returns=self.daily_returns,
            portfolio_values=[state["total_value"] for state in self.portfolio_history],
            trades=self.trade_journal.get_all_trades(),
        )

        # Combine all results
        results = {
            # Basic metrics
            "symbol": symbol,
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            # Timing
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": trading_days,
            # Performance metrics
            **performance_metrics,
            # Trade statistics
            "trade_journal": self.trade_journal,
            "portfolio_history": self.portfolio_history,
            "total_trades": len(self.trade_journal.get_all_trades()),
            # Risk metrics
            "max_drawdown_dollars": self._calculate_max_drawdown_dollars(),
            "final_cash": self.cash,
            "simulation_success": True,
        }

        return results

    def _calculate_max_drawdown_dollars(self) -> float:
        """Calculate maximum drawdown in dollar terms."""
        if not self.portfolio_history:
            return 0.0

        values = [state["total_value"] for state in self.portfolio_history]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _create_failed_simulation_result(self) -> Dict[str, Any]:
        """Create result dictionary for failed simulation."""
        return {
            "simulation_success": False,
            "error": "Simulation failed to initialize or execute properly",
            "total_return": 0.0,
            "total_trades": 0,
            "trade_journal": TradeJournal(),
            "portfolio_history": [],
        }
