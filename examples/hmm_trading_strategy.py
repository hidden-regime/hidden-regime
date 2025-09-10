"""
HMM Trading Strategy Example for hidden-regime package.

Demonstrates how to build trading strategies based on HMM regime detection
including position sizing, risk management, and performance analysis.
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the hidden-regime package
import hidden_regime as hr
from hidden_regime import HiddenMarkovModel, HMMConfig


class RegimeBasedStrategy:
    """
    Trading strategy based on HMM regime detection.

    Implements position sizing, risk management, and performance tracking
    based on detected market regimes.
    """

    def __init__(
        self,
        hmm_config: Optional[HMMConfig] = None,
        position_limits: Dict[int, float] = None,
        risk_params: Dict[str, float] = None,
    ):
        """
        Initialize regime-based trading strategy.

        Args:
            hmm_config: HMM configuration
            position_limits: Maximum position size by regime {regime: max_position}
            risk_params: Risk management parameters
        """
        self.hmm_config = hmm_config or HMMConfig.for_market_data(conservative=True)

        # Default position limits by regime (as fraction of capital)
        self.position_limits = position_limits or {
            0: 0.0,  # Bear regime: no long positions
            1: 0.3,  # Sideways regime: moderate position
            2: 1.0,  # Bull regime: full position
        }

        # Risk management parameters
        self.risk_params = risk_params or {
            "stop_loss_pct": 0.05,  # 5% stop loss
            "confidence_threshold": 0.6,  # Minimum regime confidence
            "max_drawdown_limit": 0.15,  # 15% max drawdown limit
            "rebalance_frequency": 5,  # Rebalance every N days
        }

        self.hmm_model = None
        self.is_trained = False

        # Performance tracking
        self.portfolio_value = 10000.0  # Starting capital
        self.positions = 0.0  # Current position
        self.cash = self.portfolio_value
        self.performance_history = []
        self.trades = []

    def train_model(self, returns: np.ndarray, verbose: bool = False) -> None:
        """Train HMM model on historical returns."""
        self.hmm_model = HiddenMarkovModel(config=self.hmm_config)
        self.hmm_model.fit(returns, verbose=verbose)
        self.is_trained = True

        if verbose:
            print(
                f"Model trained: {self.hmm_model.n_states} states, "
                f"{self.hmm_model.training_history_['iterations']} iterations"
            )

    def get_regime_signal(self, new_return: float) -> Dict[str, any]:
        """
        Get trading signal based on regime detection.

        Args:
            new_return: New return observation

        Returns:
            Dictionary with regime info and trading signal
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")

        # Update regime probabilities
        regime_info = self.hmm_model.update_with_observation(new_return)

        # Determine position based on regime and confidence
        regime = regime_info["most_likely_regime"]
        confidence = regime_info["confidence"]

        # Base position from regime
        base_position = self.position_limits.get(regime, 0.0)

        # Adjust for confidence
        if confidence < self.risk_params["confidence_threshold"]:
            base_position *= 0.5  # Reduce position if low confidence

        signal = {
            "regime": regime,
            "confidence": confidence,
            "regime_interpretation": regime_info["regime_interpretation"],
            "target_position": base_position,
            "expected_return": regime_info["expected_return"],
            "expected_volatility": regime_info["expected_volatility"],
            "regime_probabilities": regime_info["regime_probabilities"],
        }

        return signal

    def execute_trade(
        self, signal: Dict[str, any], current_price: float, date: datetime
    ) -> Optional[Dict[str, any]]:
        """
        Execute trade based on signal.

        Args:
            signal: Trading signal from get_regime_signal
            current_price: Current asset price
            date: Current date

        Returns:
            Trade information if trade executed, None otherwise
        """
        target_position = signal["target_position"]
        current_position_pct = self.positions * current_price / self.portfolio_value

        # Check if rebalancing is needed
        position_diff = abs(target_position - current_position_pct)

        if position_diff < 0.05:  # Don't trade for small differences
            return None

        # Calculate trade size
        target_dollar_position = target_position * self.portfolio_value
        current_dollar_position = self.positions * current_price

        trade_dollars = target_dollar_position - current_dollar_position
        shares_to_trade = trade_dollars / current_price

        # Check if we have enough cash for buys
        if shares_to_trade > 0 and trade_dollars > self.cash:
            shares_to_trade = self.cash / current_price
            trade_dollars = shares_to_trade * current_price

        # Execute trade
        if abs(shares_to_trade) > 0.001:  # Minimum trade size
            self.positions += shares_to_trade
            self.cash -= trade_dollars

            trade_info = {
                "date": date,
                "shares": shares_to_trade,
                "price": current_price,
                "dollar_amount": trade_dollars,
                "regime": signal["regime"],
                "confidence": signal["confidence"],
                "position_after": self.positions,
                "cash_after": self.cash,
            }

            self.trades.append(trade_info)
            return trade_info

        return None

    def update_performance(self, current_price: float, date: datetime) -> None:
        """Update performance tracking."""
        portfolio_value = self.cash + self.positions * current_price

        performance_record = {
            "date": date,
            "price": current_price,
            "positions": self.positions,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "return": (
                (portfolio_value / self.portfolio_value - 1)
                if self.portfolio_value > 0
                else 0.0
            ),
        }

        self.performance_history.append(performance_record)
        self.portfolio_value = portfolio_value

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        if len(self.performance_history) < 2:
            return {}

        df = pd.DataFrame(self.performance_history)
        returns = df["return"].pct_change().dropna()

        if len(returns) == 0:
            return {}

        metrics = {
            "total_return": df["return"].iloc[-1],
            "annualized_return": df["return"].iloc[-1] * (252 / len(df)),
            "volatility": returns.std() * np.sqrt(252),
            "sharpe_ratio": (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            ),
            "max_drawdown": self._calculate_max_drawdown(df["portfolio_value"]),
            "num_trades": len(self.trades),
            "win_rate": self._calculate_win_rate(),
        }

        return metrics

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.min()

    def _calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades."""
        if len(self.trades) < 2:
            return 0.0

        profitable_trades = 0
        for i in range(1, len(self.trades)):
            if (
                self.trades[i]["dollar_amount"]
                * (self.trades[i]["price"] - self.trades[i - 1]["price"])
                > 0
            ):
                profitable_trades += 1

        return profitable_trades / max(1, len(self.trades) - 1)


def generate_market_data_with_regimes(
    n_days: int = 252, random_seed: int = 42
) -> pd.DataFrame:
    """Generate realistic market data with regime switching."""
    np.random.seed(random_seed)

    # Regime definitions
    regimes = {
        0: {"mean": -0.015, "std": 0.035, "duration": 20},  # Bear
        1: {"mean": 0.002, "std": 0.018, "duration": 30},  # Sideways
        2: {"mean": 0.012, "std": 0.025, "duration": 25},  # Bull
    }

    # Generate regime sequence
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    regime_sequence = []
    returns = []
    prices = [100.0]  # Starting price

    current_regime = 1
    days_in_regime = 0

    for i in range(n_days):
        # Check for regime switch
        days_in_regime += 1
        switch_prob = 1.0 / regimes[current_regime]["duration"]

        if days_in_regime > 5 and np.random.random() < switch_prob:
            # Switch to a different regime
            available_regimes = [r for r in regimes.keys() if r != current_regime]
            current_regime = np.random.choice(available_regimes)
            days_in_regime = 0

        regime_sequence.append(current_regime)

        # Generate return for current regime
        regime = regimes[current_regime]
        daily_return = np.random.normal(
            regime["mean"] / 252, regime["std"] / np.sqrt(252)
        )
        returns.append(daily_return)

        # Calculate new price
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "date": dates,
            "price": prices[1:],  # Skip initial price
            "log_return": returns,
            "true_regime": regime_sequence,
        }
    )

    return data


def run_strategy_backtest(
    data: pd.DataFrame, train_pct: float = 0.6, verbose: bool = True
) -> Tuple[RegimeBasedStrategy, pd.DataFrame]:
    """
    Run complete strategy backtest.

    Args:
        data: Market data with price and log_return columns
        train_pct: Percentage of data for training
        verbose: Print progress information

    Returns:
        Tuple of (strategy, performance_df)
    """
    # Split data
    train_size = int(len(data) * train_pct)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    if verbose:
        print(
            f"Training period: {train_data['date'].iloc[0].date()} to {train_data['date'].iloc[-1].date()}"
        )
        print(
            f"Testing period: {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()}"
        )

    # Initialize strategy
    strategy = RegimeBasedStrategy(
        hmm_config=HMMConfig(
            n_states=3,
            max_iterations=100,
            initialization_method="kmeans",
            random_seed=42,
        )
    )

    # Train model
    if verbose:
        print("\nTraining HMM model...")

    strategy.train_model(train_data["log_return"].values, verbose=verbose)

    # Initialize performance tracking
    strategy.update_performance(test_data["price"].iloc[0], test_data["date"].iloc[0])

    # Run backtest
    if verbose:
        print("\nRunning backtest...")

    day_count = 0
    for idx, row in test_data.iterrows():
        day_count += 1

        # Get trading signal
        signal = strategy.get_regime_signal(row["log_return"])

        # Execute trade if needed
        trade = strategy.execute_trade(signal, row["price"], row["date"])

        # Update performance
        strategy.update_performance(row["price"], row["date"])

        # Periodic progress update
        if verbose and day_count % 50 == 0:
            current_return = strategy.performance_history[-1]["return"]
            print(
                f"  Day {day_count}: Return = {current_return:.2%}, "
                f"Current regime = {signal['regime']} ({signal['regime_interpretation']})"
            )

    # Create performance DataFrame
    performance_df = pd.DataFrame(strategy.performance_history)

    return strategy, performance_df


def analyze_strategy_performance(
    strategy: RegimeBasedStrategy,
    performance_df: pd.DataFrame,
    benchmark_data: pd.DataFrame,
) -> None:
    """Analyze and display strategy performance."""
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Calculate metrics
    metrics = strategy.calculate_performance_metrics()

    # Calculate benchmark performance (buy and hold)
    benchmark_return = (
        benchmark_data["price"].iloc[-1] / benchmark_data["price"].iloc[0]
    ) - 1
    benchmark_volatility = benchmark_data["log_return"].std() * np.sqrt(252)
    benchmark_sharpe = (
        benchmark_return * 252 / len(benchmark_data)
    ) / benchmark_volatility

    print("Strategy Performance:")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
    print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Number of Trades: {metrics.get('num_trades', 0)}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")

    print("\nBenchmark (Buy & Hold) Performance:")
    print(f"  Total Return: {benchmark_return:.2%}")
    print(f"  Annualized Return: {benchmark_return * 252 / len(benchmark_data):.2%}")
    print(f"  Volatility: {benchmark_volatility:.2%}")
    print(f"  Sharpe Ratio: {benchmark_sharpe:.3f}")

    # Regime analysis
    print("\nRegime-Based Trade Analysis:")
    if strategy.trades:
        trades_by_regime = {}
        for trade in strategy.trades:
            regime = trade["regime"]
            if regime not in trades_by_regime:
                trades_by_regime[regime] = []
            trades_by_regime[regime].append(trade)

        regime_names = {0: "Bear", 1: "Sideways", 2: "Bull"}
        for regime, trades in trades_by_regime.items():
            total_dollars = sum(abs(trade["dollar_amount"]) for trade in trades)
            print(
                f"  {regime_names.get(regime, f'Regime {regime}')}: {len(trades)} trades, "
                f"${total_dollars:,.0f} total volume"
            )


def plot_strategy_results(
    performance_df: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    strategy: RegimeBasedStrategy,
) -> None:
    """Plot strategy performance results."""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Regime-Based Trading Strategy Results", fontsize=16)

        # Plot 1: Portfolio value vs benchmark
        axes[0, 0].plot(
            performance_df["date"],
            performance_df["portfolio_value"],
            label="Strategy",
            linewidth=2,
        )

        # Normalize benchmark to same starting value
        benchmark_normalized = (
            benchmark_data["price"]
            / benchmark_data["price"].iloc[0]
            * performance_df["portfolio_value"].iloc[0]
        )
        axes[0, 0].plot(
            benchmark_data["date"],
            benchmark_normalized,
            label="Buy & Hold",
            linewidth=2,
            alpha=0.7,
        )

        axes[0, 0].set_title("Portfolio Value Over Time")
        axes[0, 0].set_ylabel("Portfolio Value ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Position sizes over time
        axes[0, 1].plot(performance_df["date"], performance_df["positions"])
        axes[0, 1].set_title("Position Size Over Time")
        axes[0, 1].set_ylabel("Shares Held")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Returns distribution
        strategy_returns = performance_df["return"].pct_change().dropna()
        benchmark_returns = benchmark_data["log_return"]

        axes[1, 0].hist(
            strategy_returns, bins=30, alpha=0.7, label="Strategy", density=True
        )
        axes[1, 0].hist(
            benchmark_returns, bins=30, alpha=0.7, label="Benchmark", density=True
        )
        axes[1, 0].set_title("Returns Distribution")
        axes[1, 0].set_xlabel("Daily Return")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Trade sizes and dates
        if strategy.trades:
            trade_dates = [trade["date"] for trade in strategy.trades]
            trade_amounts = [trade["dollar_amount"] for trade in strategy.trades]
            trade_regimes = [trade["regime"] for trade in strategy.trades]

            colors = ["red", "yellow", "green"]
            for regime in [0, 1, 2]:
                regime_trades = [
                    (date, amount)
                    for date, amount, r in zip(
                        trade_dates, trade_amounts, trade_regimes
                    )
                    if r == regime
                ]
                if regime_trades:
                    dates, amounts = zip(*regime_trades)
                    axes[1, 1].scatter(
                        dates,
                        amounts,
                        c=colors[regime],
                        label=f"Regime {regime}",
                        alpha=0.7,
                    )

            axes[1, 1].set_title("Trade Amounts by Regime")
            axes[1, 1].set_xlabel("Date")
            axes[1, 1].set_ylabel("Trade Amount ($)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        try:
            plt.savefig(
                "hmm_trading_strategy_results.png", dpi=150, bbox_inches="tight"
            )
            print("   ✓ Plot saved as 'hmm_trading_strategy_results.png'")
        except Exception as e:
            print(f"   ⚠ Could not save plot: {e}")

        try:
            plt.show()
            print("   ✓ Interactive plots displayed")
        except Exception as e:
            print(f"   ⚠ Could not display plots: {e}")

    except ImportError:
        print("   ⚠ Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"   ✗ Plotting failed: {e}")


def main():
    """Run the complete trading strategy demonstration."""
    print("Hidden Regime - HMM Trading Strategy Demo")
    print("Demonstrates regime-based trading using Hidden Markov Models")
    print()

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        # Generate market data
        print("1. Generating synthetic market data with regime switching...")
        data = generate_market_data_with_regimes(n_days=400, random_seed=42)

        print(f"   Generated {len(data)} days of market data")
        print(
            f"   Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}"
        )
        print(
            f"   Return statistics: mean={data['log_return'].mean():.4f}, "
            f"std={data['log_return'].std():.4f}"
        )

        # Run backtest
        print("\n2. Running regime-based trading strategy...")
        strategy, performance_df = run_strategy_backtest(
            data, train_pct=0.6, verbose=True
        )

        # Analyze performance
        test_start_idx = int(len(data) * 0.6)
        benchmark_data = data.iloc[test_start_idx:].copy()
        analyze_strategy_performance(strategy, performance_df, benchmark_data)

        # Create visualizations
        plot_strategy_results(performance_df, benchmark_data, strategy)

        print("\n" + "=" * 60)
        print("TRADING STRATEGY DEMO COMPLETED!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("- HMM regime detection can inform position sizing decisions")
        print("- Risk management rules help control drawdowns")
        print("- Strategy performance depends on regime detection accuracy")
        print("- Real-world implementation requires additional considerations:")
        print("  * Transaction costs and slippage")
        print("  * Market impact and liquidity constraints")
        print("  * Model retraining and parameter drift")
        print("  * Multiple asset classes and correlation changes")

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("Check that all required dependencies are installed.")
        raise


if __name__ == "__main__":
    main()
