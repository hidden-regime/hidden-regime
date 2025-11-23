"""
Walk-Forward Backtest for Regime Deterioration Short Strategy

This script implements PROPER temporal isolation:
1. Train HMM on initial lookback period
2. Walk forward day-by-day, retraining as new data becomes available
3. Never use future data that wasn't available at decision time

This eliminates lookahead bias and provides realistic trading performance.

Usage:
    python examples/quickstart/test_deterioration_walkforward.py
    python examples/quickstart/test_deterioration_walkforward.py --start 2020-01-01 --end 2024-01-01

Requirements:
    - hidden-regime package installed
    - Internet connection (downloads SPY data)

Output:
    - Temporally isolated backtest results
    - Realistic Sharpe ratio estimate
    - Trade log with no lookahead bias
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from tqdm import tqdm

# Add package to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import hidden_regime as hr
from hidden_regime.pipeline.temporal import TemporalController
from hidden_regime.analysis.regime_quality import RegimeQualityAnalyzer


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    deterioration_score: float
    entry_regime: str
    exit_regime: str
    exit_reason: str
    pnl: float
    pnl_pct: float


class WalkForwardDeteriorationBacktest:
    """
    Walk-forward backtest with proper temporal isolation.

    Key differences from simple backtest:
    - HMM retrained as new data arrives
    - Regime predictions only use past data
    - No lookahead bias
    """

    def __init__(
        self,
        ticker: str = 'SPY',
        n_states: int = 3,
        initial_lookback: int = 500,  # Initial training period (2 years recommended for 3+ states)
        retrain_freq: int = 20,       # Retrain every N days
        initial_capital: float = 100000.0,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.05,
        max_hold_days: int = 60,
        target_profit_pct: float = 0.15,
        # Deterioration detection parameters
        confidence_lookback: int = 10,
        min_regime_age: int = 20,
        min_confidence_decline: float = -0.05,
        min_deterioration_score: float = 0.3,
    ):
        """Initialize walk-forward backtest."""
        self.ticker = ticker
        self.n_states = n_states
        self.initial_lookback = initial_lookback
        self.retrain_freq = retrain_freq

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self.target_profit_pct = target_profit_pct

        # Deterioration detection
        self.confidence_lookback = confidence_lookback
        self.min_regime_age = min_regime_age
        self.min_confidence_decline = min_confidence_decline
        self.min_deterioration_score = min_deterioration_score

        self.trades: List[Trade] = []
        self.position = None
        self.regime_history = []
        self.bull_regimes = None

        self.days_since_retrain = 0
        self.retrain_count = 0

        # Quality analyzer for data-driven regime identification
        self.quality_analyzer = RegimeQualityAnalyzer(
            min_sharpe=0.3,
            min_return_annual=0.05,
            min_days=20,
            min_persistence=0.5,
        )

    def run(
        self,
        start_date: str = '2020-01-01',
        end_date: str = '2024-01-01'
    ) -> Dict:
        """
        Run walk-forward backtest with proper temporal isolation.

        Args:
            start_date: Backtest start (YYYY-MM-DD)
            end_date: Backtest end (YYYY-MM-DD)

        Returns:
            Dict with backtest results
        """
        print("=" * 70)
        print("WALK-FORWARD BACKTEST - REGIME DETERIORATION SHORT")
        print("=" * 70)
        print()
        print(f"Parameters:")
        print(f"  Ticker:              {self.ticker}")
        print(f"  States:              {self.n_states}")
        print(f"  Period:              {start_date} to {end_date}")
        print(f"  Initial Lookback:    {self.initial_lookback} days")
        print(f"  Retrain Frequency:   Every {self.retrain_freq} days")
        print(f"  Capital:             ${self.initial_capital:,.2f}")
        print()

        # Step 1: Download all data (we'll only use it temporally)
        print("Step 1: Downloading full dataset...")
        print("(This data will be accessed with temporal isolation)")
        print()

        from hidden_regime.config.data import FinancialDataConfig
        from hidden_regime.factories import component_factory

        data_config = FinancialDataConfig(
            ticker=self.ticker,
            start_date=start_date,
            end_date=end_date
        )
        data_loader = component_factory.create_data_component(data_config)
        full_data = data_loader.update()

        print(f"✅ Downloaded {len(full_data)} days of data")
        print(f"   Data range: {full_data.index[0].date()} to {full_data.index[-1].date()}")
        print()

        # Step 2: Create pipeline for regime detection
        print("Step 2: Creating regime detection pipeline...")
        print()

        pipeline = hr.create_financial_pipeline(
            ticker=self.ticker,
            n_states=self.n_states,
            include_report=False  # We don't need reports in backtest
        )

        # Step 3: Create temporal controller
        print("Step 3: Setting up temporal controller...")
        print("(This ensures NO future data leakage)")
        print()

        controller = TemporalController(
            pipeline=pipeline,
            full_dataset=full_data,
            enable_data_collection=True
        )

        # Step 4: Walk forward through time
        print("Step 4: Walking forward through time...")
        print()
        print("-" * 70)
        print()

        # Get dates to walk through
        walk_dates = full_data.index[self.initial_lookback:]  # Start after initial training

        for current_date in tqdm(walk_dates):
            date_str = current_date.strftime('%Y-%m-%d')

            # Retrain if needed
            if self.days_since_retrain >= self.retrain_freq or self.retrain_count == 0:
                self._retrain_model(controller, date_str)
                self.days_since_retrain = 0
                self.retrain_count += 1

            # Update pipeline to current date (temporal isolation!)
            try:
                controller.update_as_of(date_str)
            except Exception as e:
                print(f"⚠️  Error updating to {date_str}: {e}")
                continue

            # Get regime prediction (only using data up to current_date!)
            regime_data = pipeline.component_outputs.get('interpreter')
            if regime_data is None:
                regime_data = pipeline.component_outputs.get('analysis')

            if regime_data is None or len(regime_data) == 0:
                continue

            # Get current regime (last row only - this is TODAY's prediction)
            current_row = regime_data.iloc[-1]
            regime = current_row.get('regime_name', current_row.get('predicted_state', 'Unknown'))
            confidence = current_row.get('confidence', 0.5)

            # Get current price
            price = full_data.loc[current_date, 'price']

            # Update regime history
            self._update_regime_history(current_date, regime, confidence)

            # Identify bull regimes on first day with enough data
            if self.bull_regimes is None and len(regime_data) > 100:
                self.bull_regimes = self._identify_bull_regimes(regime_data)
                print(f"Identified tradeable regimes: {self.bull_regimes}")
                print()

            # Check exit conditions for existing position
            if self.position is not None:
                self._check_exit_conditions(current_date, regime, confidence, price)

            # Check for new deterioration signals (only if no position)
            if self.position is None and self.bull_regimes is not None:
                deterioration_score = self._detect_deterioration(regime, confidence)

                if deterioration_score >= self.min_deterioration_score:
                    self._enter_short_position(
                        current_date, regime, confidence, price, deterioration_score
                    )

            self.days_since_retrain += 1

        # Close any remaining position at end
        if self.position is not None:
            final_date = walk_dates[-1]
            final_price = full_data.loc[final_date, 'price']
            self._exit_position(final_date, final_price, 'End', "Backtest end")

        print("-" * 70)
        print()

        # Calculate statistics
        return self._calculate_statistics(start_date, end_date)

    def _retrain_model(self, controller: TemporalController, current_date: str):
        """Retrain HMM model on data up to current date."""
        # Retraining happens automatically when we call update_as_of
        # The model sees only data up to current_date
        pass  # Controller handles this

    def _update_regime_history(self, date: datetime, regime: str, confidence: float):
        """Update regime history."""
        self.regime_history.append({
            'date': date,
            'regime': regime,
            'confidence': confidence
        })

        # Keep only recent history
        max_history = max(self.confidence_lookback, 30)
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]

    def _identify_bull_regimes(self, regime_data: pd.DataFrame) -> List[str]:
        """
        Identify which regimes to trade using data-driven quality metrics.

        Replaces heuristic string matching with actual performance analysis.
        """
        # Need price data from the original dataset for quality analysis
        # Get it from the controller's dataset
        try:
            price_data = regime_data[['price']].copy() if 'price' in regime_data.columns else None

            if price_data is None:
                # Fallback to old heuristic method
                print("⚠️  No price data available for quality analysis, using heuristic fallback")
                return self._identify_bull_regimes_heuristic(regime_data)

            # Use quality analyzer to identify tradeable regimes
            tradeable = self.quality_analyzer.identify_tradeable_regimes(
                regime_data=regime_data,
                price_data=price_data,
                strategy_type='bull',  # We want positive-return regimes for deterioration detection
            )

            if tradeable:
                # Print quality metrics
                print("✅ Identified tradeable regimes (data-driven):")
                for name, metrics in sorted(tradeable.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True):
                    print(f"   {name}:")
                    print(f"     - Sharpe: {metrics['sharpe_ratio']:.2f}")
                    print(f"     - Annual Return: {metrics['annual_return']:.1%}")
                    print(f"     - Win Rate: {metrics['win_rate']:.1%}")
                    print(f"     - Persistence: {metrics['persistence_score']:.2f}")
                    print(f"     - Days: {metrics['days']}")

                return list(tradeable.keys())
            else:
                print("⚠️  No regimes meet quality criteria for trading")
                return []

        except Exception as e:
            print(f"⚠️  Error in quality analysis: {e}, using heuristic fallback")
            return self._identify_bull_regimes_heuristic(regime_data)

    def _identify_bull_regimes_heuristic(self, regime_data: pd.DataFrame) -> List[str]:
        """Fallback heuristic method (string matching)."""
        regime_col = 'regime_name' if 'regime_name' in regime_data.columns else 'predicted_state'
        unique_regimes = regime_data[regime_col].unique()

        # Check for explicit bull keywords
        bull_keywords = ["bull", "uptrend", "growth", "euphoric", "expansion"]
        bull_regimes = [r for r in unique_regimes
                       if any(keyword in r.lower() for keyword in bull_keywords)]

        if bull_regimes:
            return bull_regimes

        # Check for "Strong" regimes (but NOT "Strong Sideways")
        strong_regimes = [r for r in unique_regimes
                         if "strong" in r.lower() and "sideways" not in r.lower()]

        if strong_regimes:
            return strong_regimes

        # Default: return empty (don't trade)
        return []

    def _detect_deterioration(self, current_regime: str, confidence: float) -> float:
        """
        Detect regime deterioration.

        Returns deterioration score (0-1).
        """
        # Only trade bull/strong regimes
        if current_regime not in self.bull_regimes:
            return 0.0

        # Need sufficient history
        if len(self.regime_history) < self.confidence_lookback:
            return 0.0

        # Check 1: Regime age
        days_in_regime = self._count_consecutive_days(current_regime)
        if days_in_regime < self.min_regime_age:
            return 0.0

        # Check 2: Confidence declining
        confidence_trend = self._calculate_confidence_trend()
        if confidence_trend > self.min_confidence_decline:
            return 0.0

        # Calculate deterioration score
        maturity_score = min(1.0, (days_in_regime - 20) / 100.0)
        decline_score = min(1.0, abs(confidence_trend) / 0.10)

        deterioration_score = (maturity_score * 0.3) + (decline_score * 0.7)

        return deterioration_score

    def _count_consecutive_days(self, regime: str) -> int:
        """Count consecutive days in current regime."""
        if not self.regime_history:
            return 0

        count = 0
        for entry in reversed(self.regime_history):
            if entry['regime'] == regime:
                count += 1
            else:
                break

        return count

    def _calculate_confidence_trend(self) -> float:
        """Calculate confidence trend (slope)."""
        if len(self.regime_history) < self.confidence_lookback:
            return 0.0

        recent = self.regime_history[-self.confidence_lookback:]
        confidences = [entry['confidence'] for entry in recent]

        x = np.arange(len(confidences))
        slope = np.polyfit(x, confidences, 1)[0]

        return slope

    def _enter_short_position(
        self,
        date: datetime,
        regime: str,
        confidence: float,
        price: float,
        deterioration_score: float
    ):
        """Enter short position."""
        position_size = deterioration_score * self.max_position_size

        self.position = {
            'entry_date': date,
            'entry_price': price,
            'entry_regime': regime,
            'entry_confidence': confidence,
            'position_size': position_size,
            'deterioration_score': deterioration_score
        }

        print(f"SHORT ENTRY: {date.date()} @ ${price:.2f}")
        print(f"  Regime: {regime}, Confidence: {confidence:.3f}")
        print(f"  Deterioration: {deterioration_score:.2f}, Size: {abs(position_size):.1%}")
        print()

    def _check_exit_conditions(
        self,
        date: datetime,
        regime: str,
        confidence: float,
        price: float
    ):
        """Check if we should exit."""
        if self.position is None:
            return

        entry_price = self.position['entry_price']
        entry_date = self.position['entry_date']
        entry_confidence = self.position['entry_confidence']

        pnl_pct = (entry_price - price) / entry_price

        # Exit 1: Regime changed (thesis confirmed or broken)
        if regime != self.position['entry_regime']:
            self._exit_position(date, price, regime, f"Regime changed to {regime}")
            return

        # Exit 2: Confidence recovering
        if confidence > entry_confidence + 0.05:
            self._exit_position(date, price, regime, "Confidence recovering")
            return

        # Exit 3: Stop loss
        if pnl_pct < -self.stop_loss_pct:
            self._exit_position(date, price, regime, f"Stop loss ({-self.stop_loss_pct:.1%})")
            return

        # Exit 4: Max hold
        days_held = (date - entry_date).days
        if days_held > self.max_hold_days:
            self._exit_position(date, price, regime, f"Max hold ({self.max_hold_days} days)")
            return

        # Exit 5: Target profit
        if pnl_pct > self.target_profit_pct:
            self._exit_position(date, price, regime, f"Target profit ({self.target_profit_pct:.1%})")
            return

    def _exit_position(
        self,
        date: datetime,
        price: float,
        regime: str,
        reason: str
    ):
        """Exit position."""
        if self.position is None:
            return

        entry_price = self.position['entry_price']
        entry_date = self.position['entry_date']
        position_size = self.position['position_size']

        pnl_pct = (entry_price - price) / entry_price
        pnl = self.capital * abs(position_size) * pnl_pct

        self.capital += pnl

        trade = Trade(
            entry_date=entry_date,
            exit_date=date,
            entry_price=entry_price,
            exit_price=price,
            position_size=position_size,
            deterioration_score=self.position['deterioration_score'],
            entry_regime=self.position['entry_regime'],
            exit_regime=regime,
            exit_reason=reason,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trades.append(trade)

        print(f"EXIT: {date.date()} @ ${price:.2f}")
        print(f"  Reason: {reason}, Days: {(date - entry_date).days}")
        print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2%}), Capital: ${self.capital:,.2f}")
        print()

        self.position = None

    def _calculate_statistics(self, start_date: str, end_date: str) -> Dict:
        """Calculate backtest statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'error': 'No trades executed'
            }

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        years = (end - start).days / 365.25
        trades_per_year = len(self.trades) / years if years > 0 else 0

        returns = [t.pnl_pct for t in self.trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_estimate = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'trades_per_year': trades_per_year,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'final_capital': self.capital,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'avg_win_pct': np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0,
            'avg_loss_pct': np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t.pnl for t in self.trades]),
            'largest_loss': min([t.pnl for t in self.trades]),
            'avg_hold_days': np.mean([(t.exit_date - t.entry_date).days for t in self.trades]),
            'sharpe_estimate': sharpe_estimate,
            'years': years,
            'retrain_count': self.retrain_count
        }


def test_walkforward_deterioration(
    ticker: str = 'SPY',
    n_states: int = 3,
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01',
    initial_capital: float = 100000.0
):
    """
    Test deterioration strategy with proper walk-forward validation.

    Args:
        ticker: Stock ticker
        n_states: Number of HMM states
        start_date: Backtest start
        end_date: Backtest end
        initial_capital: Starting capital

    Returns:
        Dict with backtest results
    """
    backtest = WalkForwardDeteriorationBacktest(
        ticker=ticker,
        n_states=n_states,
        initial_capital=initial_capital
    )

    results = backtest.run(start_date, end_date)

    # Print results
    print("=" * 70)
    print("WALK-FORWARD BACKTEST RESULTS (NO LOOKAHEAD BIAS)")
    print("=" * 70)
    print()

    if results.get('total_trades', 0) == 0:
        print("❌ No trades executed")
        print()
        return results

    print(f"Total Trades:        {results['total_trades']}")
    print(f"Winning Trades:      {results['winning_trades']}")
    print(f"Losing Trades:       {results['losing_trades']}")
    print(f"Win Rate:            {results['win_rate']:.1%}")
    print()
    print(f"Trades per Year:     {results['trades_per_year']:.1f}")
    print(f"Avg Hold Days:       {results['avg_hold_days']:.1f}")
    print(f"Model Retrains:      {results['retrain_count']}")
    print()
    print(f"Total P&L:           ${results['total_pnl']:+,.2f}")
    print(f"Total Return:        {results['total_return']:+.1%}")
    print(f"Final Capital:       ${results['final_capital']:,.2f}")
    print()
    print(f"Average Win:         ${results['avg_win']:,.2f} ({results['avg_win_pct']:+.2%})")
    print(f"Average Loss:        ${results['avg_loss']:,.2f} ({results['avg_loss_pct']:+.2%})")
    print(f"Largest Win:         ${results['largest_win']:,.2f}")
    print(f"Largest Loss:        ${results['largest_loss']:,.2f}")
    print()
    print(f"Sharpe Estimate:     {results['sharpe_estimate']:.2f}")
    print()
    print("=" * 70)
    print("VALIDATION (Sharpe 10+ Target)")
    print("=" * 70)
    print()

    checks = []

    # Trade frequency
    if 30 <= results['trades_per_year'] <= 50:
        checks.append(("✅", f"Trade frequency in range ({results['trades_per_year']:.1f}/year)"))
    else:
        checks.append(("⚠️", f"Trade frequency: {results['trades_per_year']:.1f}/year (target: 30-50)"))

    # Win rate
    if results['win_rate'] >= 0.60:
        checks.append(("✅", f"Win rate meets target ({results['win_rate']:.1%})"))
    else:
        checks.append(("❌", f"Win rate below target ({results['win_rate']:.1%} < 60%)"))

    # Sharpe
    if results['sharpe_estimate'] >= 10.0:
        checks.append(("✅", f"Sharpe meets target ({results['sharpe_estimate']:.2f})"))
    else:
        checks.append(("❌", f"Sharpe below target ({results['sharpe_estimate']:.2f} < 10.0)"))

    # Returns
    if results['total_return'] > 0:
        checks.append(("✅", f"Positive returns ({results['total_return']:+.1%})"))
    else:
        checks.append(("❌", f"Negative returns ({results['total_return']:+.1%})"))

    for symbol, message in checks:
        print(f"{symbol} {message}")

    print()
    print("=" * 70)
    print("IMPORTANT: These results have NO lookahead bias!")
    print("HMM was retrained {} times using only past data.".format(results['retrain_count']))
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Walk-forward backtest of deterioration strategy (no lookahead bias)'
    )
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker')
    parser.add_argument('--n-states', type=int, default=3, help='Number of HMM states')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')

    args = parser.parse_args()

    test_walkforward_deterioration(
        ticker=args.ticker,
        n_states=args.n_states,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )
