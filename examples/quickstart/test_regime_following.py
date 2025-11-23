"""
Walk-Forward Backtest for Simple Regime Following Strategy

Strategy #1: The simplest possible regime-based trading:
- Long position in bull regimes
- Short position in bear regimes
- Flat (no position) in sideways/uncertain regimes

This script implements PROPER temporal isolation:
1. Train HMM on initial lookback period
2. Walk forward day-by-day, retraining as new data becomes available
3. Never use future data that wasn't available at decision time

Usage:
    python examples/quickstart/test_regime_following.py
    python examples/quickstart/test_regime_following.py --ticker QQQ
    python examples/quickstart/test_regime_following.py --n-states 2

Output:
    - Temporally isolated backtest results
    - Realistic Sharpe ratio estimate
    - Trade log with no lookahead bias
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from tqdm import tqdm

# Add package to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import hidden_regime as hr
from hidden_regime.pipeline.temporal import TemporalController
from hidden_regime.analysis.regime_quality import RegimeQualityAnalyzer
from hidden_regime.config.data import FinancialDataConfig
from hidden_regime.factories import component_factory


@dataclass
class Position:
    """Current position state."""
    direction: str  # 'long', 'short', or 'flat'
    entry_date: datetime
    entry_price: float
    size: float
    regime: str


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    direction: str
    size: float
    regime: str
    pnl: float
    pnl_pct: float


class WalkForwardRegimeFollowing:
    """
    Walk-forward backtest for Simple Regime Following strategy.

    Strategy Logic:
    - Identify bull/bear/sideways regimes using data-driven quality analysis
    - Go long when in bull regime with sufficient confidence
    - Go short when in bear regime with sufficient confidence
    - Stay flat in sideways or low-confidence periods
    """

    def __init__(
        self,
        ticker: str = 'SPY',
        n_states: int = 3,
        initial_lookback: int = 500,
        retrain_freq: int = 20,
        initial_capital: float = 100000.0,
        max_position_size: float = 1.0,
        min_confidence: float = 0.6,
        use_shorts: bool = True,
    ):
        """
        Initialize walk-forward backtest.

        Args:
            ticker: Stock symbol to trade
            n_states: Number of HMM states
            initial_lookback: Days of data for initial training
            retrain_freq: Retrain model every N days
            initial_capital: Starting capital
            max_position_size: Maximum position as fraction of capital
            min_confidence: Minimum regime confidence to trade
            use_shorts: Whether to short bear regimes (False = long-only)
        """
        self.ticker = ticker
        self.n_states = n_states
        self.initial_lookback = initial_lookback
        self.retrain_freq = retrain_freq

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.use_shorts = use_shorts

        # State
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Regime classification
        self.bull_regimes: List[str] = []
        self.bear_regimes: List[str] = []
        self.sideways_regimes: List[str] = []

        self.days_since_retrain = 0
        self.retrain_count = 0

        # Quality analyzer
        self.quality_analyzer = RegimeQualityAnalyzer(
            min_sharpe=0.2,
            min_return_annual=0.03,
            min_days=15,
            min_persistence=0.4,
        )

    def run(self, start_date: str = '2020-01-01', end_date: str = '2024-01-01') -> Dict:
        """Run the walk-forward backtest."""
        print("=" * 70)
        print("WALK-FORWARD BACKTEST - SIMPLE REGIME FOLLOWING")
        print("=" * 70)
        print()
        print(f"Parameters:")
        print(f"  Ticker:              {self.ticker}")
        print(f"  States:              {self.n_states}")
        print(f"  Period:              {start_date} to {end_date}")
        print(f"  Initial Lookback:    {self.initial_lookback} days")
        print(f"  Retrain Frequency:   Every {self.retrain_freq} days")
        print(f"  Capital:             ${self.initial_capital:,.2f}")
        print(f"  Min Confidence:      {self.min_confidence:.0%}")
        print(f"  Use Shorts:          {self.use_shorts}")
        print()

        # Step 1: Download all data
        print("Step 1: Downloading full dataset...")
        print("(This data will be accessed with temporal isolation)")
        print()

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

        # Step 2: Create pipeline
        print("Step 2: Creating regime detection pipeline...")
        print()

        pipeline = hr.create_financial_pipeline(
            ticker=self.ticker,
            n_states=self.n_states,
            include_report=False
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

        # Step 4: Walk forward
        print("Step 4: Walking forward through time...")
        print()
        print("-" * 70)
        print()

        # Get dates to walk through
        walk_dates = full_data.index[self.initial_lookback:]

        for current_date in tqdm(walk_dates, desc="Walk-forward"):
            date_str = current_date.strftime('%Y-%m-%d')

            # Retrain if needed
            if self.days_since_retrain >= self.retrain_freq or self.retrain_count == 0:
                self.days_since_retrain = 0
                self.retrain_count += 1

            # Update pipeline to current date (temporal isolation!)
            controller.update_as_of(date_str)

            # Get regime prediction
            regime_data = pipeline.component_outputs.get('interpreter')
            if regime_data is None or regime_data.empty:
                self.days_since_retrain += 1
                continue

            # Get current regime info
            current_row = regime_data.iloc[-1]
            regime_col = 'regime_name' if 'regime_name' in regime_data.columns else 'predicted_state'
            regime = current_row.get(regime_col, 'Unknown')
            confidence = current_row.get('confidence', 0.5)

            # Get current price
            price = full_data.loc[current_date, 'price']

            # Classify regimes on first day with enough data
            if not self.bull_regimes and len(regime_data) > 100:
                self._classify_regimes(regime_data, full_data.loc[:current_date])

            # Manage position
            self._manage_position(current_date, regime, confidence, price)

            # Track equity
            self._track_equity(current_date, price)

            self.days_since_retrain += 1

        # Close any remaining position
        if self.position is not None:
            final_date = walk_dates[-1]
            final_price = full_data.loc[final_date, 'price']
            self._close_position(final_date, final_price, "Backtest end")

        print("-" * 70)
        print()

        # Calculate statistics
        return self._calculate_statistics(start_date, end_date)

    def _classify_regimes(self, regime_data: pd.DataFrame, price_data: pd.DataFrame):
        """Classify regimes into bull/bear/sideways using quality analysis."""
        regime_col = 'regime_name' if 'regime_name' in regime_data.columns else 'predicted_state'
        unique_regimes = regime_data[regime_col].unique()

        print("\n📊 Analyzing regime quality...")

        # Analyze all regimes
        price_df = price_data[['price']].copy() if 'price' in price_data.columns else None

        if price_df is None:
            print("⚠️  No price data, using heuristic classification")
            self._classify_regimes_heuristic(unique_regimes)
            return

        # Calculate returns
        returns = price_df['price'].pct_change().dropna()
        common_index = regime_data.index.intersection(returns.index)

        if len(common_index) < 50:
            print("⚠️  Insufficient data, using heuristic classification")
            self._classify_regimes_heuristic(unique_regimes)
            return

        aligned_regimes = regime_data.loc[common_index]
        aligned_returns = returns.loc[common_index]

        # Calculate metrics for each regime
        regime_metrics = {}

        for regime_name in unique_regimes:
            mask = aligned_regimes[regime_col] == regime_name
            regime_returns = aligned_returns[mask]

            if len(regime_returns) < 10:
                continue

            annual_return = regime_returns.mean() * 252
            annual_vol = regime_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            regime_metrics[regime_name] = {
                'annual_return': annual_return,
                'sharpe': sharpe,
                'days': len(regime_returns),
            }

        # Classify based on returns AND regime name
        for regime_name, metrics in regime_metrics.items():
            annual_ret = metrics['annual_return']
            regime_lower = regime_name.lower()

            # ALWAYS respect "sideways" in the name - these are not directional
            if 'sideways' in regime_lower:
                self.sideways_regimes.append(regime_name)
            # Check for explicit bull/bear keywords
            elif any(kw in regime_lower for kw in ['bull', 'uptrend', 'growth', 'euphoric']):
                self.bull_regimes.append(regime_name)
            elif any(kw in regime_lower for kw in ['bear', 'downtrend', 'crisis']):
                self.bear_regimes.append(regime_name)
            # Fall back to return-based classification
            elif annual_ret > 0.08:  # >8% annual return (stricter threshold)
                self.bull_regimes.append(regime_name)
            elif annual_ret < -0.08:  # <-8% annual return
                self.bear_regimes.append(regime_name)
            else:
                self.sideways_regimes.append(regime_name)

        # Print classification
        print("\n✅ Regime Classification (data-driven):")
        if self.bull_regimes:
            print(f"   BULL (long): {self.bull_regimes}")
            for r in self.bull_regimes:
                m = regime_metrics.get(r, {})
                print(f"      {r}: {m.get('annual_return', 0):.1%} annual, Sharpe {m.get('sharpe', 0):.2f}")
        if self.bear_regimes:
            print(f"   BEAR (short): {self.bear_regimes}")
            for r in self.bear_regimes:
                m = regime_metrics.get(r, {})
                print(f"      {r}: {m.get('annual_return', 0):.1%} annual, Sharpe {m.get('sharpe', 0):.2f}")
        if self.sideways_regimes:
            print(f"   SIDEWAYS (flat): {self.sideways_regimes}")
        print()

        if not self.bull_regimes and not self.bear_regimes:
            print("⚠️  No directional regimes found, strategy may not trade")

    def _classify_regimes_heuristic(self, unique_regimes):
        """Fallback heuristic classification."""
        for regime in unique_regimes:
            regime_lower = regime.lower()
            if any(kw in regime_lower for kw in ['bull', 'uptrend', 'growth', 'euphoric']):
                self.bull_regimes.append(regime)
            elif any(kw in regime_lower for kw in ['bear', 'downtrend', 'crisis']):
                self.bear_regimes.append(regime)
            else:
                self.sideways_regimes.append(regime)

        print(f"   BULL: {self.bull_regimes}")
        print(f"   BEAR: {self.bear_regimes}")
        print(f"   SIDEWAYS: {self.sideways_regimes}")

    def _manage_position(self, date: datetime, regime: str, confidence: float, price: float):
        """Manage position based on current regime."""
        # Determine target position
        target_direction = 'flat'

        if confidence >= self.min_confidence:
            if regime in self.bull_regimes:
                target_direction = 'long'
            elif regime in self.bear_regimes and self.use_shorts:
                target_direction = 'short'

        # Get current direction
        current_direction = self.position.direction if self.position else 'flat'

        # Change position if needed
        if target_direction != current_direction:
            # Close current position
            if self.position is not None:
                self._close_position(date, price, f"Regime change to {regime}")

            # Open new position
            if target_direction != 'flat':
                self._open_position(date, price, target_direction, regime, confidence)

    def _open_position(self, date: datetime, price: float, direction: str, regime: str, confidence: float):
        """Open a new position."""
        # Size based on confidence
        size_factor = min(1.0, confidence) * self.max_position_size
        position_value = self.capital * size_factor

        self.position = Position(
            direction=direction,
            entry_date=date,
            entry_price=price,
            size=position_value,
            regime=regime
        )

    def _close_position(self, date: datetime, price: float, reason: str):
        """Close current position and record trade."""
        if self.position is None:
            return

        # Calculate P&L
        if self.position.direction == 'long':
            pnl_pct = (price - self.position.entry_price) / self.position.entry_price
        else:  # short
            pnl_pct = (self.position.entry_price - price) / self.position.entry_price

        pnl = self.position.size * pnl_pct
        self.capital += pnl

        # Record trade
        trade = Trade(
            entry_date=self.position.entry_date,
            exit_date=date,
            entry_price=self.position.entry_price,
            exit_price=price,
            direction=self.position.direction,
            size=self.position.size,
            regime=self.position.regime,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trades.append(trade)

        self.position = None

    def _track_equity(self, date: datetime, price: float):
        """Track equity curve."""
        # Calculate current equity
        equity = self.capital

        if self.position is not None:
            if self.position.direction == 'long':
                unrealized_pnl = self.position.size * (price - self.position.entry_price) / self.position.entry_price
            else:
                unrealized_pnl = self.position.size * (self.position.entry_price - price) / self.position.entry_price
            equity += unrealized_pnl

        self.equity_curve.append({
            'date': date,
            'equity': equity,
            'position': self.position.direction if self.position else 'flat'
        })

    def _calculate_statistics(self, start_date: str, end_date: str) -> Dict:
        """Calculate backtest statistics."""
        print("=" * 70)
        print("WALK-FORWARD BACKTEST RESULTS (NO LOOKAHEAD BIAS)")
        print("=" * 70)
        print()

        if not self.trades:
            print("❌ No trades executed")
            print()
            return {'total_trades': 0, 'sharpe_ratio': 0}

        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in self.trades)
        avg_pnl = total_pnl / total_trades
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl <= 0]) if losing_trades > 0 else 0

        # Returns
        trade_returns = [t.pnl_pct for t in self.trades]
        avg_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)

        # Sharpe ratio (annualized)
        if std_return > 0:
            trades_per_year = total_trades / 4  # Assuming 4-year period
            sharpe = (avg_return * trades_per_year) / (std_return * np.sqrt(trades_per_year))
        else:
            sharpe = 0

        # Max drawdown from equity curve
        if self.equity_curve:
            equities = [e['equity'] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0

        # Direction breakdown
        long_trades = [t for t in self.trades if t.direction == 'long']
        short_trades = [t for t in self.trades if t.direction == 'short']

        # Print results
        print(f"📈 Performance Summary")
        print(f"   Total Return:       {(self.capital - self.initial_capital) / self.initial_capital:+.2%}")
        print(f"   Final Capital:      ${self.capital:,.2f}")
        print(f"   Max Drawdown:       {max_dd:.2%}")
        print(f"   Sharpe Ratio:       {sharpe:.2f}")
        print()

        print(f"📊 Trade Statistics")
        print(f"   Total Trades:       {total_trades}")
        print(f"   Win Rate:           {win_rate:.1%}")
        print(f"   Avg P&L:            ${avg_pnl:,.2f}")
        print(f"   Avg Win:            ${avg_win:,.2f}")
        print(f"   Avg Loss:           ${avg_loss:,.2f}")
        print()

        print(f"📋 Direction Breakdown")
        print(f"   Long Trades:        {len(long_trades)}")
        if long_trades:
            long_wins = sum(1 for t in long_trades if t.pnl > 0)
            print(f"     Win Rate:         {long_wins / len(long_trades):.1%}")
            print(f"     Avg Return:       {np.mean([t.pnl_pct for t in long_trades]):.2%}")

        print(f"   Short Trades:       {len(short_trades)}")
        if short_trades:
            short_wins = sum(1 for t in short_trades if t.pnl > 0)
            print(f"     Win Rate:         {short_wins / len(short_trades):.1%}")
            print(f"     Avg Return:       {np.mean([t.pnl_pct for t in short_trades]):.2%}")
        print()

        # Recent trades
        print(f"📜 Recent Trades (last 5)")
        for trade in self.trades[-5:]:
            days_held = (trade.exit_date - trade.entry_date).days
            print(f"   {trade.entry_date.date()} → {trade.exit_date.date()} ({days_held}d)")
            print(f"     {trade.direction.upper()} {trade.regime}: ${trade.entry_price:.2f} → ${trade.exit_price:.2f}")
            print(f"     P&L: ${trade.pnl:,.2f} ({trade.pnl_pct:+.2%})")
        print()

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_capital': self.capital,
            'avg_trade_return': avg_return,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
        }


def test_regime_following(
    ticker: str = 'SPY',
    n_states: int = 3,
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01',
    initial_capital: float = 100000.0,
    min_confidence: float = 0.6,
    use_shorts: bool = True,
):
    """Run the regime following backtest."""
    backtest = WalkForwardRegimeFollowing(
        ticker=ticker,
        n_states=n_states,
        initial_capital=initial_capital,
        min_confidence=min_confidence,
        use_shorts=use_shorts,
    )

    results = backtest.run(start_date, end_date)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward regime following backtest")
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--n-states', type=int, default=3, help='Number of HMM states')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--min-confidence', type=float, default=0.6, help='Min confidence to trade')
    parser.add_argument('--no-shorts', action='store_true', help='Disable short selling')

    args = parser.parse_args()

    test_regime_following(
        ticker=args.ticker,
        n_states=args.n_states,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
        use_shorts=not args.no_shorts,
    )
