"""
Simulation orchestrator for integrating trading simulation with case studies.

Provides high-level coordination between regime detection, signal generation,
and trading simulation as specified in simulation.md.
"""

import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..analysis.technical_indicators import TechnicalIndicatorAnalyzer
from ..config.simulation import SimulationConfig, SimulationResult
from .risk_management import RiskLimits, RiskManager
from .signal_generators import (
    BuyHoldSignalGenerator,
    HMMSignalGenerator,
    MultiSignalGenerator,
    TechnicalIndicatorSignalGenerator,
)
from .trading_engine import TradingSimulationEngine


class SimulationOrchestrator:
    """
    High-level orchestrator for trading simulation.

    Coordinates regime detection, signal generation, and simulation execution
    to provide comprehensive trading strategy analysis.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation orchestrator.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.config.validate()

        # Initialize components
        self.risk_manager = self._create_risk_manager()
        self.signal_generator = MultiSignalGenerator()
        self.trading_engine = None  # Created per simulation

        # Setup signal generators
        self._setup_signal_generators()

        # Technical indicator analyzer for TA signals
        if self.config.technical_indicators:
            self.ta_analyzer = TechnicalIndicatorAnalyzer()
        else:
            self.ta_analyzer = None

    def run_simulation(
        self,
        price_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
        symbol: str = "ASSET",
    ) -> SimulationResult:
        """
        Run complete trading simulation.

        Args:
            price_data: DataFrame with OHLC price data
            regime_data: Optional DataFrame with regime predictions
            symbol: Asset symbol being traded

        Returns:
            Comprehensive simulation results
        """
        print(f"🎯 Starting trading simulation for {symbol}")
        start_time = time.time()

        try:
            # Validate input data
            if not self._validate_input_data(price_data, regime_data):
                return self._create_failed_result(symbol, "Invalid input data")

            # Generate all trading signals
            print("📊 Generating trading signals...")
            signals_df = self._generate_all_signals(price_data, regime_data)

            if signals_df.empty:
                return self._create_failed_result(
                    symbol, "No trading signals generated"
                )

            # Create and initialize trading engine
            self.trading_engine = TradingSimulationEngine(
                initial_capital=self.config.initial_capital,
                default_shares=self.config.default_shares,
                transaction_cost=self.config.transaction_cost,
                enable_shorting=self.config.enable_shorting,
                risk_manager=self.risk_manager,
            )

            # Run simulation for each strategy
            print("🔄 Running strategy simulations...")
            strategy_results = {}

            for strategy_name in signals_df.columns:
                print(f"  📈 Simulating {strategy_name}...")

                # Create strategy-specific signals
                strategy_signals = pd.DataFrame(
                    {strategy_name: signals_df[strategy_name]}
                )

                # Run simulation for this strategy
                strategy_result = self.trading_engine.run_simulation(
                    price_data=price_data, signals=strategy_signals, symbol=symbol
                )

                if strategy_result.get("simulation_success", False):
                    strategy_results[strategy_name] = strategy_result
                    print(
                        f"    ✅ {strategy_name}: {strategy_result.get('total_return_pct', 0):.2f}% return"
                    )
                else:
                    print(f"    ❌ {strategy_name}: Simulation failed")

            # Analyze results and create final summary
            print("📋 Analyzing simulation results...")
            simulation_result = self._analyze_simulation_results(
                strategy_results=strategy_results,
                signals_df=signals_df,
                symbol=symbol,
                start_time=start_time,
                price_data=price_data,
            )

            execution_time = time.time() - start_time
            print(f"✅ Simulation completed in {execution_time:.2f} seconds")

            return simulation_result

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Simulation failed after {execution_time:.2f} seconds: {e}")
            return self._create_failed_result(symbol, str(e))

    def _create_risk_manager(self) -> RiskManager:
        """Create risk manager from configuration."""
        risk_limits = RiskLimits(
            max_position_pct=self.config.max_position_pct,
            max_portfolio_risk=self.config.max_portfolio_risk,
            stop_loss_pct=self.config.stop_loss_pct,
            max_total_exposure=self.config.max_total_exposure,
            max_drawdown_pct=self.config.max_drawdown_pct,
        )
        return RiskManager(risk_limits)

    def _setup_signal_generators(self) -> None:
        """Setup signal generators based on configuration."""

        # Add buy-and-hold generator
        if "buy_and_hold" in self.config.signal_generators:
            self.signal_generator.add_buy_hold_generator()

        # Add HMM generators
        for hmm_strategy in self.config.hmm_strategy_types:
            if f"hmm_{hmm_strategy}" in self.config.signal_generators:
                self.signal_generator.add_generator(HMMSignalGenerator(hmm_strategy))

        # Add technical indicator generators
        for indicator in self.config.technical_indicators:
            self.signal_generator.add_technical_indicator_generator(indicator)

    def _generate_all_signals(
        self, price_data: pd.DataFrame, regime_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Generate signals from all configured generators."""

        # Prepare additional data for signal generation
        additional_data = (
            regime_data.copy() if regime_data is not None else pd.DataFrame()
        )

        # Add technical indicators to additional data
        if self.ta_analyzer and self.config.technical_indicators:
            try:
                # Calculate technical indicators
                ta_results = self.ta_analyzer.analyze_comprehensive_indicators(
                    price_data, indicators=self.config.technical_indicators
                )

                # Add indicator values to additional_data
                if "indicators" in ta_results:
                    for indicator_name, indicator_values in ta_results[
                        "indicators"
                    ].items():
                        if isinstance(indicator_values, pd.Series):
                            additional_data[indicator_name] = indicator_values

            except Exception as e:
                warnings.warn(f"Failed to calculate technical indicators: {e}")

        # Generate all signals
        try:
            signals_df = self.signal_generator.generate_all_signals(
                price_data, additional_data
            )
            print(
                f"  📊 Generated signals for {len(signals_df.columns)} strategies over {len(signals_df)} days"
            )
            return signals_df
        except Exception as e:
            print(f"  ❌ Signal generation failed: {e}")
            return pd.DataFrame()

    def _validate_input_data(
        self, price_data: pd.DataFrame, regime_data: Optional[pd.DataFrame]
    ) -> bool:
        """Validate input data for simulation."""

        # Check price data
        required_price_columns = ["open", "high", "low", "close"]
        if not all(col in price_data.columns for col in required_price_columns):
            missing = [
                col for col in required_price_columns if col not in price_data.columns
            ]
            print(f"❌ Missing required price columns: {missing}")
            return False

        if len(price_data) < 2:
            print("❌ Insufficient price data (need at least 2 days)")
            return False

        # Check regime data if HMM strategies are enabled
        hmm_strategies = [
            s for s in self.config.signal_generators if s.startswith("hmm_")
        ]
        if hmm_strategies and regime_data is not None:
            if "predicted_state" not in regime_data.columns:
                print(
                    "❌ HMM strategies enabled but no 'predicted_state' in regime data"
                )
                return False

        return True

    def _analyze_simulation_results(
        self,
        strategy_results: Dict[str, Dict[str, Any]],
        signals_df: pd.DataFrame,
        symbol: str,
        start_time: float,
        price_data: pd.DataFrame,
    ) -> SimulationResult:
        """Analyze simulation results and create summary."""

        if not strategy_results:
            return self._create_failed_result(
                symbol, "No successful strategy simulations"
            )

        # Find best strategy by Sharpe ratio
        best_strategy = None
        best_sharpe = float("-inf")

        for strategy_name, result in strategy_results.items():
            sharpe = result.get("sharpe_ratio", float("-inf"))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = strategy_name

        # Get benchmark comparison (vs buy-and-hold if available)
        benchmark_comparison = None
        if self.config.benchmark_strategy in strategy_results:
            benchmark_result = strategy_results[self.config.benchmark_strategy]
            if best_strategy and best_strategy != self.config.benchmark_strategy:
                best_result = strategy_results[best_strategy]
                benchmark_comparison = {
                    "benchmark_strategy": self.config.benchmark_strategy,
                    "benchmark_return": benchmark_result.get("total_return_pct", 0),
                    "best_strategy": best_strategy,
                    "best_return": best_result.get("total_return_pct", 0),
                    "excess_return": best_result.get("total_return_pct", 0)
                    - benchmark_result.get("total_return_pct", 0),
                }

        # Calculate overall portfolio metrics (using best strategy)
        if best_strategy and best_strategy in strategy_results:
            best_result = strategy_results[best_strategy]
        else:
            # Use first available result
            best_result = list(strategy_results.values())[0]

        # Create simulation result
        simulation_result = SimulationResult(
            simulation_success=True,
            symbol=symbol,
            start_date=(
                str(price_data.index[0].date())
                if hasattr(price_data.index[0], "date")
                else str(price_data.index[0])
            ),
            end_date=(
                str(price_data.index[-1].date())
                if hasattr(price_data.index[-1], "date")
                else str(price_data.index[-1])
            ),
            initial_capital=best_result.get("initial_capital", 0),
            final_value=best_result.get("final_value", 0),
            total_return=best_result.get("total_return", 0),
            total_return_pct=best_result.get("total_return_pct", 0),
            annualized_return=best_result.get("annualized_return", 0),
            volatility=best_result.get("annualized_volatility", 0),
            sharpe_ratio=best_result.get("sharpe_ratio", 0),
            sortino_ratio=best_result.get("sortino_ratio", 0),
            max_drawdown=best_result.get("max_drawdown", 0),
            max_drawdown_pct=best_result.get("max_drawdown_pct", 0),
            total_trades=best_result.get("total_trades", 0),
            win_rate=best_result.get("win_rate", 0),
            profit_factor=best_result.get("profit_factor", 0),
            avg_trade_pnl=best_result.get("avg_trade_pnl", 0),
            strategy_results=strategy_results,
            best_strategy=best_strategy,
            benchmark_comparison=benchmark_comparison,
            trade_journal=best_result.get("trade_journal"),
            portfolio_history=best_result.get("portfolio_history"),
            daily_signals=signals_df if self.config.save_daily_signals else None,
            simulation_config=self.config,
            execution_time=time.time() - start_time,
        )

        return simulation_result

    def _create_failed_result(
        self, symbol: str, error_message: str
    ) -> SimulationResult:
        """Create simulation result for failed simulation."""
        return SimulationResult(
            simulation_success=False,
            symbol=symbol,
            start_date="",
            end_date="",
            initial_capital=0,
            final_value=0,
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            total_trades=0,
            win_rate=0,
            profit_factor=0,
            avg_trade_pnl=0,
            strategy_results={},
            best_strategy=None,
            benchmark_comparison=None,
            execution_time=0,
        )

    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of generated signals."""
        if signals_df.empty:
            return {}

        summary = {}
        for strategy in signals_df.columns:
            strategy_signals = signals_df[strategy]
            summary[strategy] = {
                "total_signals": len(strategy_signals),
                "buy_signals": (strategy_signals == 1).sum(),
                "sell_signals": (strategy_signals == -1).sum(),
                "hold_signals": (strategy_signals == 0).sum(),
                "buy_signal_pct": (strategy_signals == 1).mean() * 100,
                "sell_signal_pct": (strategy_signals == -1).mean() * 100,
                "hold_signal_pct": (strategy_signals == 0).mean() * 100,
            }

        return summary
