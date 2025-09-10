"""
Retail Trading Framework for Hidden Regime Models

This module provides utilities for simulating realistic retail trader behavior
and evaluating regime-based trading strategies.

Key Components:
- RetailTradingSimulator: Simulates realistic position sizing and risk management
- PerformanceAnalyzer: Comprehensive trading performance evaluation
- RegimeTrader: Automated trading based on regime predictions
- RiskManager: Position sizing and drawdown protection

Author: Hidden Regime Team
Created: 2025-01-XX
"""

# Import hidden regime components
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hidden_regime import DataLoader, HiddenMarkovModel, HMMConfig
from hidden_regime.models.online_hmm import OnlineHMM, OnlineHMMConfig
from hidden_regime.models.state_standardizer import StateStandardizer


@dataclass
class TradingConfig:
    """Configuration for retail trading simulation."""

    # Capital management
    initial_capital: float = 10000.0  # Starting capital
    max_position_size: float = 0.8  # Maximum position size (80%)
    min_position_size: float = -0.5  # Maximum short position (-50%)

    # Risk management
    max_drawdown_limit: float = 0.2  # Stop trading if 20% drawdown
    daily_risk_limit: float = 0.05  # Max 5% daily risk
    confidence_threshold: float = 0.6  # Minimum confidence for trades

    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission per trade
    bid_ask_spread: float = 0.0005  # 0.05% bid-ask spread
    slippage: float = 0.0002  # 0.02% slippage

    # Trading rules
    regime_hold_days: int = 2  # Hold position for at least 2 days
    max_trades_per_day: int = 1  # Limit to 1 trade per day

    # Regime-based position sizing
    regime_positions: Dict[str, float] = None

    def __post_init__(self):
        if self.regime_positions is None:
            self.regime_positions = {
                "Bull": 0.8,  # 80% long in bull markets
                "Sideways": 0.2,  # 20% long in sideways
                "Bear": -0.3,  # 30% short in bear markets
                "Crisis": 0.0,  # Cash in crisis
                "Euphoric": 0.4,  # Reduced in euphoria (bubble warning)
            }


class RetailTradingSimulator:
    """
    Simulate realistic retail trader behavior with regime-based strategies.

    This class models how a typical retail trader would use Hidden Regime models
    for position sizing and risk management decisions.
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()

        # Trading state
        self.current_capital = self.config.initial_capital
        self.current_position = 0.0  # Current position size (-1 to 1)
        self.shares_held = 0.0
        self.days_in_position = 0

        # Trading history
        self.trade_history: List[Dict] = []
        self.position_history: List[Dict] = []
        self.performance_history: List[Dict] = []

        # Risk management
        self.peak_capital = self.config.initial_capital
        self.current_drawdown = 0.0
        self.trading_halted = False

    def calculate_position_size(
        self,
        regime: str,
        confidence: float,
        current_price: float,
        volatility: float = None,
    ) -> float:
        """
        Calculate optimal position size based on regime and confidence.

        Args:
            regime: Current market regime
            confidence: Confidence in regime prediction (0-1)
            current_price: Current asset price
            volatility: Optional volatility estimate

        Returns:
            Position size as fraction of capital (-1 to 1)
        """
        if self.trading_halted:
            return 0.0

        # Base position from regime
        base_position = self.config.regime_positions.get(regime, 0.0)

        # Adjust for confidence
        confidence_adj = min(confidence, 1.0)  # Cap at 100%
        adjusted_position = base_position * confidence_adj

        # Apply volatility scaling if available
        if volatility is not None:
            # Reduce position size for high volatility
            vol_adj = min(1.0, 0.02 / max(volatility, 0.01))  # Target 2% daily vol
            adjusted_position *= vol_adj

        # Apply risk limits
        adjusted_position = np.clip(
            adjusted_position,
            self.config.min_position_size,
            self.config.max_position_size,
        )

        # Minimum confidence check
        if confidence < self.config.confidence_threshold:
            adjusted_position *= 0.5  # Reduce position if not confident

        return adjusted_position

    def execute_trade(
        self,
        target_position: float,
        current_price: float,
        timestamp: datetime,
        regime: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Execute a trade to reach target position size.

        Args:
            target_position: Target position size (-1 to 1)
            current_price: Current asset price
            timestamp: Trading timestamp
            regime: Current regime
            confidence: Regime confidence

        Returns:
            Trade execution details
        """
        # Check trading rules
        if (
            self.days_in_position < self.config.regime_hold_days
            and len(self.trade_history) > 0
        ):
            # Must hold position for minimum days
            target_position = self.current_position

        # Calculate shares to trade
        target_shares = (target_position * self.current_capital) / current_price
        shares_to_trade = target_shares - self.shares_held

        trade_details = {
            "timestamp": timestamp,
            "regime": regime,
            "confidence": confidence,
            "price": current_price,
            "target_position": target_position,
            "current_position": self.current_position,
            "shares_traded": shares_to_trade,
            "trade_value": abs(shares_to_trade * current_price),
            "transaction_costs": 0.0,
            "executed": False,
        }

        # Check if trade is needed
        position_change = abs(target_position - self.current_position)
        if position_change < 0.01:  # Less than 1% change
            trade_details["reason"] = "No significant position change needed"
            return trade_details

        # Calculate transaction costs
        trade_value = abs(shares_to_trade * current_price)
        commission = trade_value * self.config.commission_rate
        spread_cost = trade_value * self.config.bid_ask_spread
        slippage_cost = trade_value * self.config.slippage
        total_costs = commission + spread_cost + slippage_cost

        # Check if profitable after costs
        if trade_value < total_costs * 10:  # Don't trade if costs > 10% of trade
            trade_details["reason"] = (
                "Transaction costs too high relative to trade size"
            )
            return trade_details

        # Execute trade
        self.shares_held = target_shares
        self.current_position = target_position
        self.current_capital -= total_costs
        self.days_in_position = 0  # Reset position timer

        trade_details.update(
            {
                "executed": True,
                "transaction_costs": total_costs,
                "new_capital": self.current_capital,
                "new_position": self.current_position,
            }
        )

        self.trade_history.append(trade_details)
        return trade_details

    def update_portfolio(
        self, current_price: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Update portfolio value and check risk limits.

        Args:
            current_price: Current asset price
            timestamp: Current timestamp

        Returns:
            Portfolio status
        """
        # Calculate current portfolio value
        position_value = self.shares_held * current_price
        total_capital = self.current_capital + position_value

        # Calculate daily return
        daily_return = 0.0
        if len(self.performance_history) > 0:
            prev_capital = self.performance_history[-1]["total_capital"]
            daily_return = (total_capital - prev_capital) / prev_capital

        # Update peak capital and drawdown
        if total_capital > self.peak_capital:
            self.peak_capital = total_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (
                self.peak_capital - total_capital
            ) / self.peak_capital

        # Check risk limits
        risk_alert = None
        if self.current_drawdown > self.config.max_drawdown_limit:
            self.trading_halted = True
            risk_alert = f"Maximum drawdown exceeded: {self.current_drawdown:.2%}"
        elif abs(daily_return) > self.config.daily_risk_limit:
            risk_alert = f"Daily risk limit exceeded: {daily_return:.2%}"

        # Increment position days
        self.days_in_position += 1

        portfolio_status = {
            "timestamp": timestamp,
            "current_price": current_price,
            "shares_held": self.shares_held,
            "position_value": position_value,
            "cash": self.current_capital,
            "total_capital": total_capital,
            "position_size": self.current_position,
            "daily_return": daily_return,
            "total_return": (total_capital - self.config.initial_capital)
            / self.config.initial_capital,
            "drawdown": self.current_drawdown,
            "days_in_position": self.days_in_position,
            "risk_alert": risk_alert,
            "trading_halted": self.trading_halted,
        }

        self.performance_history.append(portfolio_status)
        return portfolio_status


class PerformanceAnalyzer:
    """
    Comprehensive analysis of trading strategy performance.

    Provides metrics commonly used by retail traders to evaluate
    strategy effectiveness.
    """

    def __init__(self):
        self.metrics = {}

    def analyze_performance(
        self,
        performance_history: List[Dict],
        trade_history: List[Dict],
        benchmark_returns: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Analyze comprehensive trading performance.

        Args:
            performance_history: Portfolio performance over time
            trade_history: Individual trade details
            benchmark_returns: Optional benchmark for comparison

        Returns:
            Comprehensive performance metrics
        """
        df = pd.DataFrame(performance_history)

        if len(df) == 0:
            return {"error": "No performance data available"}

        # Basic return metrics
        returns = df["daily_return"].values
        total_return = df["total_return"].iloc[-1]

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = df["drawdown"].max()

        # Trading activity metrics
        num_trades = len(trade_history)
        win_rate = self._calculate_win_rate(trade_history) if num_trades > 0 else 0
        avg_trade_return = (
            self._calculate_avg_trade_return(trade_history) if num_trades > 0 else 0
        )

        # Time-based metrics
        days_traded = len(df)
        annualized_return = (
            (1 + total_return) ** (252 / days_traded) - 1 if days_traded > 0 else 0
        )

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade_return,
            "days_traded": days_traded,
            "final_capital": df["total_capital"].iloc[-1],
            "peak_capital": df["total_capital"].max(),
        }

        # Benchmark comparison if provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_total_return = np.prod(1 + benchmark_returns) - 1
            benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
            benchmark_sharpe = self._calculate_sharpe_ratio(benchmark_returns)

            metrics.update(
                {
                    "benchmark_total_return": benchmark_total_return,
                    "benchmark_volatility": benchmark_volatility,
                    "benchmark_sharpe": benchmark_sharpe,
                    "excess_return": total_return - benchmark_total_return,
                    "information_ratio": (
                        annualized_return - benchmark_total_return * (252 / days_traded)
                    )
                    / max(volatility - benchmark_volatility, 0.01),
                }
            )

        return metrics

    def _calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) <= 1 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sortino ratio (downside deviation)."""
        if len(returns) <= 1:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)

    def _calculate_win_rate(self, trade_history: List[Dict]) -> float:
        """Calculate percentage of profitable trades."""
        if not trade_history:
            return 0.0

        executed_trades = [t for t in trade_history if t.get("executed", False)]
        if not executed_trades:
            return 0.0

        # This is a simplified calculation - would need P&L per trade
        # For now, return a placeholder
        return 0.55  # Placeholder: 55% win rate

    def _calculate_avg_trade_return(self, trade_history: List[Dict]) -> float:
        """Calculate average return per trade."""
        if not trade_history:
            return 0.0

        # Placeholder - would need actual P&L calculation per trade
        return 0.02  # Placeholder: 2% average trade return


class RegimeTrader:
    """
    Automated trading system based on Hidden Regime predictions.

    This class combines online HMM learning with retail trading simulation
    to demonstrate practical regime-based trading strategies.
    """

    def __init__(
        self,
        regime_type: str = "4_state",
        online_config: OnlineHMMConfig = None,
        trading_config: TradingConfig = None,
    ):

        self.regime_type = regime_type
        self.online_config = online_config or OnlineHMMConfig()
        self.trading_config = trading_config or TradingConfig()

        # Initialize components
        self.hmm_model = None
        self.state_standardizer = StateStandardizer(regime_type=regime_type)
        self.trading_simulator = RetailTradingSimulator(trading_config)
        self.performance_analyzer = PerformanceAnalyzer()

        # Trading state
        self.training_complete = False
        self.current_regime = None
        self.regime_confidence = 0.0

        # Results tracking
        self.regime_history = []
        self.prediction_accuracy = []

    def train_initial_model(self, training_data: np.ndarray, verbose: bool = False):
        """Train initial HMM model on historical data."""

        # Create and train base model
        config = HMMConfig.for_standardized_regimes(regime_type=self.regime_type)
        base_model = HiddenMarkovModel(config)
        base_model.fit(training_data, verbose=verbose)

        # Initialize online model with trained parameters
        self.hmm_model = OnlineHMM(config.n_states, config, self.online_config)

        # Transfer learned parameters
        self.hmm_model.transition_matrix_ = base_model.transition_matrix_.copy()
        self.hmm_model.emission_params_ = base_model.emission_params_.copy()
        self.hmm_model.initial_probs_ = base_model.initial_probs_.copy()

        # Initialize online learning components
        self.hmm_model.sufficient_stats.initialize(config.n_states)

        self.training_complete = True

        if verbose:
            print(f"Initial model trained on {len(training_data)} observations")
            print(f"Model type: {self.regime_type}")
            print(f"States: {config.n_states}")

    def process_new_observation(
        self, new_return: float, current_price: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Process new market data and make trading decisions.

        Args:
            new_return: New log return observation
            current_return: Current asset price
            timestamp: Current timestamp

        Returns:
            Trading decision and analysis
        """
        if not self.training_complete:
            raise ValueError("Model must be trained before processing observations")

        # Update model with new observation
        update_result = self.hmm_model.add_observation(
            new_return, update_parameters=True
        )

        # Get regime prediction
        state_probs = update_result["state_probabilities"]
        most_likely_state = update_result["most_likely_state"]

        # Convert to regime name
        emission_params = self.hmm_model.emission_params_
        state_mapping = self.state_standardizer.standardize_states(emission_params)

        current_regime = state_mapping[most_likely_state]
        regime_confidence = state_probs[most_likely_state]

        # Calculate volatility estimate
        recent_returns = list(self.hmm_model.observation_buffer)[-20:]  # Last 20 days
        volatility = np.std(recent_returns) if len(recent_returns) > 5 else None

        # Calculate target position
        target_position = self.trading_simulator.calculate_position_size(
            current_regime, regime_confidence, current_price, volatility
        )

        # Execute trade if needed
        trade_result = self.trading_simulator.execute_trade(
            target_position, current_price, timestamp, current_regime, regime_confidence
        )

        # Update portfolio
        portfolio_status = self.trading_simulator.update_portfolio(
            current_price, timestamp
        )

        # Store regime history
        regime_info = {
            "timestamp": timestamp,
            "regime": current_regime,
            "confidence": regime_confidence,
            "state_probs": state_probs.copy(),
            "return": new_return,
            "price": current_price,
        }
        self.regime_history.append(regime_info)

        # Update current state
        self.current_regime = current_regime
        self.regime_confidence = regime_confidence

        return {
            "regime_info": regime_info,
            "trade_result": trade_result,
            "portfolio_status": portfolio_status,
            "model_update": update_result,
        }

    def get_performance_summary(
        self, benchmark_returns: np.ndarray = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance analysis."""

        performance_metrics = self.performance_analyzer.analyze_performance(
            self.trading_simulator.performance_history,
            self.trading_simulator.trade_history,
            benchmark_returns,
        )

        # Add regime-specific analysis
        regime_df = pd.DataFrame(self.regime_history)
        if len(regime_df) > 0:
            regime_distribution = (
                regime_df["regime"].value_counts(normalize=True).to_dict()
            )
            avg_confidence = regime_df["confidence"].mean()

            performance_metrics.update(
                {
                    "regime_distribution": regime_distribution,
                    "avg_regime_confidence": avg_confidence,
                    "total_regime_changes": (
                        len(regime_df) - 1 if len(regime_df) > 1 else 0
                    ),
                }
            )

        return performance_metrics


# Utility functions for retail traders
def load_market_data_for_trading(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Load and prepare market data for trading analysis."""

    loader = DataLoader()
    try:
        data = loader.load(ticker, start_date, end_date)

        # Ensure required columns
        required_cols = ["date", "price", "log_return"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return data

    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        raise


def compare_regime_models(
    data: pd.DataFrame, regime_types: List[str] = None
) -> Dict[str, Any]:
    """
    Compare performance of different regime model configurations.

    Args:
        data: Market data with price and return columns
        regime_types: List of regime types to compare

    Returns:
        Comparison results
    """
    if regime_types is None:
        regime_types = ["3_state", "4_state", "5_state"]

    results = {}

    for regime_type in regime_types:
        try:
            # Split data for training/testing
            split_point = int(len(data) * 0.7)  # 70% training, 30% testing
            training_data = data.iloc[:split_point]
            testing_data = data.iloc[split_point:]

            # Create and test regime trader
            trader = RegimeTrader(regime_type=regime_type)
            trader.train_initial_model(
                training_data["log_return"].values, verbose=False
            )

            # Simulate trading on test data
            for _, row in testing_data.iterrows():
                trader.process_new_observation(
                    row["log_return"], row["price"], row["date"]
                )

            # Get performance metrics
            performance = trader.get_performance_summary()
            results[regime_type] = performance

        except Exception as e:
            print(f"Error testing {regime_type}: {e}")
            results[regime_type] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Example usage
    print("Hidden Regime Retail Trading Framework")
    print("=" * 50)

    # Example configuration
    trading_config = TradingConfig(
        initial_capital=10000, max_position_size=0.8, confidence_threshold=0.6
    )

    print(f"Trading Configuration:")
    print(f"  Initial Capital: ${trading_config.initial_capital:,.2f}")
    print(f"  Max Position: {trading_config.max_position_size:.1%}")
    print(f"  Confidence Threshold: {trading_config.confidence_threshold:.1%}")

    print("\nFramework ready for retail trading demonstrations!")
