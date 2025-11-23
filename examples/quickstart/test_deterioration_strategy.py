"""
Local Testing for Regime Deterioration Short Strategy (Strategy #4)

⚠️  WARNING: This script has lookahead bias by default!
⚠️  For realistic results, use test_deterioration_walkforward.py instead!

This script validates the deterioration detection logic locally before
deploying to QuantConnect. Tests the strategy on historical SPY data to ensure:
- Deterioration signals trigger correctly
- Position sizing works as expected
- Exit conditions behave properly
- Trade frequency matches expectations (30-50 trades/year)

LOOKAHEAD BIAS ISSUE:
By default, this script trains the HMM on ALL data at once, then backtests
on those regime predictions. This gives unrealistically good results because
the model "knows" the future when making predictions.

For proper temporal isolation (no lookahead bias), use:
    python examples/quickstart/test_deterioration_walkforward.py

Or enable temporal isolation mode:
    python examples/quickstart/test_deterioration_strategy.py --temporal-isolation

Usage:
    python examples/quickstart/test_deterioration_strategy.py
    python examples/quickstart/test_deterioration_strategy.py --start 2020-01-01 --end 2024-01-01
    python examples/quickstart/test_deterioration_strategy.py --temporal-isolation

Requirements:
    - hidden-regime package installed
    - Internet connection (downloads SPY data from Yahoo Finance)

Output:
    - Trade log with entry/exit signals
    - Performance metrics
    - Strategy validation report
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add package to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import hidden_regime as hr
from hidden_regime.analysis.stability import RegimeStabilityMetrics


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


class DeteriorationDetector:
    """
    Implements the deterioration detection logic from Strategy #4.

    This mirrors the QuantConnect implementation for local testing.
    """

    def __init__(
        self,
        confidence_lookback: int = 10,
        min_transition_prob: float = 0.3,
        min_regime_age: int = 20,
        min_confidence_decline: float = -0.05,
        min_deterioration_score: float = 0.3,
        bull_regimes: List[str] = None
    ):
        """Initialize deterioration detector."""
        self.confidence_lookback = confidence_lookback
        self.min_transition_prob = min_transition_prob
        self.min_regime_age = min_regime_age
        self.min_confidence_decline = min_confidence_decline
        self.min_deterioration_score = min_deterioration_score

        self.regime_history = []
        self.bull_regimes = bull_regimes  # Will be set automatically if None

    def update_history(self, date: datetime, regime: str, confidence: float):
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

    def is_bull_regime(self, regime: str) -> bool:
        """Check if regime is bullish."""
        if self.bull_regimes is None:
            # If not specified, check for common bull regime names
            bull_keywords = ["bull", "uptrend", "euphoric", "growth", "expansion"]
            return any(keyword in regime.lower() for keyword in bull_keywords)
        return regime in self.bull_regimes

    def detect_deterioration(self, current_regime: str, confidence: float) -> float:
        """
        Detect if current regime is showing signs of deterioration.

        Returns deterioration score (0-1), where 0 = no deterioration.
        """
        # Only short bull/uptrend regimes
        if not self.is_bull_regime(current_regime):
            return 0.0

        # Need sufficient history
        if len(self.regime_history) < self.confidence_lookback:
            return 0.0

        # Check 1: How long have we been in this regime?
        days_in_regime = self._count_consecutive_days(current_regime)
        if days_in_regime < self.min_regime_age:
            return 0.0

        # Check 2: Is confidence declining?
        confidence_trend = self._calculate_confidence_trend()
        if confidence_trend > self.min_confidence_decline:
            return 0.0

        # Check 3: Estimate transition probability to bear regime
        transition_prob = self._estimate_transition_probability(confidence_trend)
        if transition_prob < self.min_transition_prob:
            return 0.0

        # Check 4: Is regime becoming more volatile (uncertainty increasing)?
        confidence_volatility = self._calculate_confidence_volatility()

        # All checks passed - calculate deterioration score
        deterioration_score = self._calculate_deterioration_score(
            days_in_regime=days_in_regime,
            confidence_trend=confidence_trend,
            transition_prob=transition_prob,
            confidence_volatility=confidence_volatility
        )

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
        """Calculate confidence trend over lookback period."""
        if len(self.regime_history) < self.confidence_lookback:
            return 0.0

        recent = self.regime_history[-self.confidence_lookback:]
        confidences = [entry['confidence'] for entry in recent]

        # Simple linear regression slope
        x = np.arange(len(confidences))
        y = np.array(confidences)

        slope = np.polyfit(x, y, 1)[0]
        return slope

    def _estimate_transition_probability(self, confidence_trend: float) -> float:
        """Estimate probability of transitioning to bear regime."""
        if confidence_trend >= 0:
            return 0.0

        # Scale negative trend to probability
        # More negative = higher probability
        prob = min(0.9, abs(confidence_trend) * 8.0)
        return prob

    def _calculate_confidence_volatility(self) -> float:
        """Calculate volatility of confidence scores."""
        if len(self.regime_history) < 5:
            return 0.0

        recent = self.regime_history[-self.confidence_lookback:]
        confidences = [entry['confidence'] for entry in recent]

        return np.std(confidences)

    def _calculate_deterioration_score(
        self,
        days_in_regime: int,
        confidence_trend: float,
        transition_prob: float,
        confidence_volatility: float
    ) -> float:
        """Calculate overall deterioration score from multiple signals."""
        scores = []

        # Factor 1: Regime maturity (older = higher score)
        # 20 days = 0.0, 60 days = 0.5, 120 days = 1.0
        maturity_score = min(1.0, (days_in_regime - 20) / 100.0)
        scores.append(maturity_score * 0.2)  # 20% weight

        # Factor 2: Confidence decline (steeper = higher score)
        # -0.05 = 0.5, -0.10 = 1.0
        decline_score = min(1.0, abs(confidence_trend) / 0.10)
        scores.append(decline_score * 0.4)  # 40% weight

        # Factor 3: Transition probability (higher = higher score)
        scores.append(transition_prob * 0.3)  # 30% weight

        # Factor 4: Confidence volatility (higher = higher score)
        # 0.05 = 0.5, 0.10 = 1.0
        volatility_score = min(1.0, confidence_volatility / 0.10)
        scores.append(volatility_score * 0.1)  # 10% weight

        total_score = sum(scores)
        return total_score


class DeteriorationStrategyBacktest:
    """
    Backtest the deterioration short strategy on historical data.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.05,
        max_hold_days: int = 60,
        target_profit_pct: float = 0.15
    ):
        """Initialize backtest."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days
        self.target_profit_pct = target_profit_pct

        self.detector = DeteriorationDetector()
        self.trades: List[Trade] = []
        self.position = None  # Current position if any

    def run(
        self,
        regime_data: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            regime_data: DataFrame with regime predictions and confidence
            price_data: DataFrame with price history (Close column)

        Returns:
            Dict with backtest results
        """
        print("Running deterioration strategy backtest...")
        print()

        # Align data
        regime_data = regime_data.copy()
        price_data = price_data.copy()

        # Identify price column - handle different naming conventions
        price_col = None
        for col in ['Close', 'close', 'price', 'Price']:
            if col in price_data.columns:
                price_col = col
                break

        if price_col is None:
            # Try to find any price-like column
            price_cols = [c for c in price_data.columns if 'close' in c.lower() or 'price' in c.lower()]
            if price_cols:
                price_col = price_cols[0]
            else:
                raise ValueError(f"No price column found in data. Available columns: {list(price_data.columns)}")

        # Merge regime and price data
        data = regime_data.join(price_data[[price_col]], how='inner')
        # Rename to 'price' for consistency
        data = data.rename(columns={price_col: 'price'})

        # Show available regimes for debugging
        regime_col = 'regime_name' if 'regime_name' in data.columns else 'predicted_state'
        unique_regimes = data[regime_col].unique()
        print(f"Regimes in data: {list(unique_regimes)}")

        # Count observations per regime
        regime_counts = data[regime_col].value_counts()
        print(f"Regime distribution:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} days ({count/len(data)*100:.1f}%)")
        print()

        # Auto-identify bull/strong regimes to short when deteriorating
        # If no explicit bull regimes, use "Strong" or best-performing regime
        if self.detector.bull_regimes is None:
            bull_regimes = self._identify_tradeable_regimes(data, unique_regimes)
            self.detector.bull_regimes = bull_regimes
            print(f"Identified tradeable regimes to monitor for deterioration: {bull_regimes}")
            print()

        for idx, row in data.iterrows():
            date = idx
            regime = row.get('regime_name', row.get('predicted_state', 'Unknown'))
            confidence = row['confidence']
            price = row['price']

            # Update detector history
            self.detector.update_history(date, regime, confidence)

            # Check if we need to exit existing position
            if self.position is not None:
                self._check_exit_conditions(date, regime, confidence, price)

            # Check for new deterioration signals (only if no position)
            if self.position is None:
                deterioration_score = self.detector.detect_deterioration(regime, confidence)

                if deterioration_score >= self.detector.min_deterioration_score:
                    self._enter_short_position(
                        date=date,
                        regime=regime,
                        confidence=confidence,
                        price=price,
                        deterioration_score=deterioration_score
                    )

        # Close any remaining position at end
        if self.position is not None:
            final_date = data.index[-1]
            final_price = data.iloc[-1]['price']
            final_regime = data.iloc[-1].get('regime_name', 'Unknown')
            self._exit_position(final_date, final_price, final_regime, "Backtest end")

        # Calculate statistics
        return self._calculate_statistics(data)

    def _enter_short_position(
        self,
        date: datetime,
        regime: str,
        confidence: float,
        price: float,
        deterioration_score: float
    ):
        """Enter short position."""
        # Scale position size by deterioration score
        # Weak signal (0.3) -> 30% short
        # Strong signal (0.8+) -> 100% short
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
        print(f"  Deterioration Score: {deterioration_score:.2f}")
        print(f"  Position Size: {abs(position_size):.1%}")
        print()

    def _check_exit_conditions(
        self,
        date: datetime,
        regime: str,
        confidence: float,
        price: float
    ):
        """Check if we should exit current position."""
        if self.position is None:
            return

        entry_price = self.position['entry_price']
        entry_date = self.position['entry_date']
        entry_regime = self.position['entry_regime']
        entry_confidence = self.position['entry_confidence']

        # Calculate unrealized P&L (short position, so profit when price falls)
        pnl_pct = (entry_price - price) / entry_price

        # Exit 1: Regime changed to bear (thesis confirmed, take profit)
        if regime in ["bearish", "Bear", "Downtrend", "Crash Bear", "Crisis"]:
            self._exit_position(date, price, regime, "Regime confirmed bear (PROFIT)")
            return

        # Exit 2: Regime changed but NOT to bear (thesis broken)
        if regime != entry_regime and regime not in ["bearish", "Bear", "Downtrend"]:
            self._exit_position(date, price, regime, f"Regime changed to {regime} (STOP)")
            return

        # Exit 3: Confidence recovering (deterioration thesis broken)
        if confidence > entry_confidence + 0.05:
            self._exit_position(date, price, regime, "Confidence recovering")
            return

        # Exit 4: Stop loss (5% adverse move)
        if pnl_pct < -self.stop_loss_pct:
            self._exit_position(date, price, regime, f"Stop loss ({-self.stop_loss_pct:.1%})")
            return

        # Exit 5: Maximum hold period
        days_held = (date - entry_date).days
        if days_held > self.max_hold_days:
            self._exit_position(date, price, regime, f"Max hold ({self.max_hold_days} days)")
            return

        # Exit 6: Target profit reached
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
        """Exit current position."""
        if self.position is None:
            return

        entry_price = self.position['entry_price']
        entry_date = self.position['entry_date']
        position_size = self.position['position_size']

        # Calculate P&L (short position)
        pnl_pct = (entry_price - price) / entry_price
        pnl = self.capital * abs(position_size) * pnl_pct

        # Update capital
        self.capital += pnl

        # Record trade
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
        print(f"  Reason: {reason}")
        print(f"  Days held: {(date - entry_date).days}")
        print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2%})")
        print(f"  Capital: ${self.capital:,.2f}")
        print()

        # Clear position
        self.position = None

    def _identify_tradeable_regimes(self, data: pd.DataFrame, unique_regimes) -> List[str]:
        """
        Identify which regimes should be monitored for deterioration.

        Strategy:
        1. Look for explicit bull keywords (Bull, Uptrend, Growth, Euphoric)
        2. Look for "Strong" regimes that can deteriorate
        3. Calculate average returns per regime and pick best-performing

        Returns list of regime names to trade.
        """
        regime_col = 'regime_name' if 'regime_name' in data.columns else 'predicted_state'

        # Strategy 1: Check for explicit bull keywords
        bull_keywords = ["bull", "uptrend", "growth", "euphoric", "expansion"]
        bull_regimes = [r for r in unique_regimes
                       if any(keyword in r.lower() for keyword in bull_keywords)]

        if bull_regimes:
            return bull_regimes

        # Strategy 2: Check for "Strong" regimes (Strong Sideways, etc.)
        strong_regimes = [r for r in unique_regimes if "strong" in r.lower()]

        if strong_regimes:
            return strong_regimes

        # Strategy 3: Calculate average returns per regime
        data = data.copy()
        data['returns'] = data['price'].pct_change()

        regime_returns = {}
        for regime in unique_regimes:
            regime_mask = data[regime_col] == regime
            avg_return = data.loc[regime_mask, 'returns'].mean()
            regime_returns[regime] = avg_return

        # Pick regime(s) with highest average returns
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1], reverse=True)

        # Return top regime if it has positive returns
        if sorted_regimes and sorted_regimes[0][1] > 0:
            return [sorted_regimes[0][0]]

        # No clear tradeable regime found
        return []

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
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

        years = (data.index[-1] - data.index[0]).days / 365.25
        trades_per_year = len(self.trades) / years if years > 0 else 0

        # Calculate Sharpe-like metric
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
            'years': years
        }


def test_deterioration_strategy(
    ticker: str = 'SPY',
    n_states: int = 3,
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01',
    initial_capital: float = 100000.0,
    use_temporal_isolation: bool = False
):
    """
    Test deterioration strategy on historical data.

    Args:
        ticker: Stock ticker to test on
        n_states: Number of HMM states
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        use_temporal_isolation: Use proper temporal isolation (recommended!)

    Returns:
        Dict with test results
    """
    print("=" * 70)
    print("REGIME DETERIORATION SHORT STRATEGY - LOCAL TEST")
    print("=" * 70)
    print()

    if not use_temporal_isolation:
        print("⚠️  WARNING: LOOKAHEAD BIAS MODE!")
        print("   This test trains HMM on ALL data, giving unrealistic results.")
        print("   Use --temporal-isolation flag for realistic performance.")
        print()

    print(f"Parameters:")
    print(f"  Ticker:        {ticker}")
    print(f"  States:        {n_states}")
    print(f"  Period:        {start_date} to {end_date}")
    print(f"  Capital:       ${initial_capital:,.2f}")
    print(f"  Temporal Isolation: {'YES (realistic)' if use_temporal_isolation else 'NO (lookahead bias!)'}")
    print()

    # Step 1: Run regime detection
    print("Step 1: Running regime detection...")
    print()

    if use_temporal_isolation:
        # CORRECT: Use temporal isolation
        from hidden_regime.config.data import FinancialDataConfig
        from hidden_regime.factories import component_factory

        data_config = FinancialDataConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        data_loader = component_factory.create_data_component(data_config)
        full_data = data_loader.update()

        pipeline = hr.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            include_report=False
        )

        from hidden_regime.pipeline.temporal import TemporalController
        controller = TemporalController(
            pipeline=pipeline,
            full_dataset=full_data,
            enable_data_collection=False
        )

        # Update to end date with temporal isolation
        controller.update_as_of(end_date)

        # Get regime results
        regime_data = pipeline.component_outputs.get('interpreter')
        if regime_data is None:
            regime_data = pipeline.component_outputs.get('analysis')

        data_component_result = full_data

    else:
        # WRONG: Lookahead bias (original behavior)
        pipeline = hr.create_financial_pipeline(
            ticker=ticker,
            n_states=n_states,
            start_date=start_date,
            end_date=end_date
        )

        pipeline.update()

        # Get regime results
        regime_data = pipeline.component_outputs.get('interpreter')
        if regime_data is None:
            regime_data = pipeline.component_outputs.get('analysis')

        # Get price data
        data_component_result = pipeline.component_outputs.get('data')

    if regime_data is None:
        print("❌ Error: No regime detection results")
        return None

    print("✅ Regime detection complete")
    print()

    # Step 2: Validate regime quality
    print("Step 2: Validating regime quality...")
    print()

    metrics = RegimeStabilityMetrics(regime_data)
    quality = metrics.get_metrics()

    print(f"  Mean Duration:   {quality['mean_duration']:.1f} days")
    print(f"  Persistence:     {quality['persistence']:.1%}")
    print(f"  Stability Score: {quality['stability_score']:.2f}")
    print(f"  Quality Rating:  {quality['quality_rating']}")
    print(f"  Trading Ready:   {'YES' if quality['is_tradeable'] else 'NO'}")
    print()

    if not quality['is_tradeable']:
        print("⚠️  Warning: Regime quality may not be suitable for trading")
        print("   Consider adjusting n_states or lookback_days")
        print()

    # Step 3: Get price data
    print("Step 3: Loading price data...")
    print()

    if data_component_result is None:
        print("❌ Error: No price data available")
        return None

    print("✅ Price data loaded")
    print()

    # Step 4: Run backtest
    print("Step 4: Running deterioration strategy backtest...")
    print()
    print("-" * 70)
    print()

    backtest = DeteriorationStrategyBacktest(initial_capital=initial_capital)
    results = backtest.run(regime_data, data_component_result)

    print("-" * 70)
    print()

    # Step 5: Report results
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print()

    if results.get('total_trades', 0) == 0:
        print("❌ No trades executed")
        print("   Strategy did not find any deterioration signals")
        print("   Consider:")
        print("   - Adjusting deterioration thresholds")
        print("   - Using longer time period")
        print("   - Checking regime detection quality")
        return results

    print(f"Total Trades:        {results['total_trades']}")
    print(f"Winning Trades:      {results['winning_trades']}")
    print(f"Losing Trades:       {results['losing_trades']}")
    print(f"Win Rate:            {results['win_rate']:.1%}")
    print()
    print(f"Trades per Year:     {results['trades_per_year']:.1f}")
    print(f"Avg Hold Days:       {results['avg_hold_days']:.1f}")
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

    # Validation checks
    print("=" * 70)
    print("STRATEGY VALIDATION (Target: Sharpe 10+)")
    print("=" * 70)
    print()

    checks = []

    # Check 1: Trade frequency
    target_min_trades = 30
    target_max_trades = 50
    if target_min_trades <= results['trades_per_year'] <= target_max_trades:
        checks.append(("✅", f"Trade frequency in target range ({results['trades_per_year']:.1f}/year)"))
    elif results['trades_per_year'] < target_min_trades:
        checks.append(("⚠️", f"Trade frequency low ({results['trades_per_year']:.1f}/year, target: {target_min_trades}-{target_max_trades})"))
    else:
        checks.append(("⚠️", f"Trade frequency high ({results['trades_per_year']:.1f}/year, target: {target_min_trades}-{target_max_trades})"))

    # Check 2: Win rate
    target_win_rate = 0.60
    if results['win_rate'] >= target_win_rate:
        checks.append(("✅", f"Win rate meets target ({results['win_rate']:.1%} >= {target_win_rate:.0%})"))
    else:
        checks.append(("❌", f"Win rate below target ({results['win_rate']:.1%} < {target_win_rate:.0%})"))

    # Check 3: Sharpe estimate
    target_sharpe = 10.0
    if results['sharpe_estimate'] >= target_sharpe:
        checks.append(("✅", f"Sharpe estimate meets target ({results['sharpe_estimate']:.2f} >= {target_sharpe:.1f})"))
    elif results['sharpe_estimate'] >= target_sharpe * 0.7:
        checks.append(("⚠️", f"Sharpe estimate close to target ({results['sharpe_estimate']:.2f}, target: {target_sharpe:.1f})"))
    else:
        checks.append(("❌", f"Sharpe estimate below target ({results['sharpe_estimate']:.2f} < {target_sharpe:.1f})"))

    # Check 4: Positive returns
    if results['total_return'] > 0:
        checks.append(("✅", f"Positive returns ({results['total_return']:+.1%})"))
    else:
        checks.append(("❌", f"Negative returns ({results['total_return']:+.1%})"))

    for symbol, message in checks:
        print(f"{symbol} {message}")

    print()

    # Overall verdict
    passed = sum(1 for symbol, _ in checks if symbol == "✅")
    total = len(checks)

    print("=" * 70)
    if passed == total:
        print("✅ VALIDATION PASSED - Strategy Ready for QuantConnect!")
        print()
        print("Next steps:")
        print("  1. Deploy quantconnect_templates/regime_deterioration_short.py")
        print("  2. Run full backtest on QuantConnect platform")
        print("  3. Optimize parameters if needed")
    elif passed >= total * 0.75:
        print("⚠️  VALIDATION MOSTLY PASSED - Minor Adjustments Recommended")
        print()

        # Provide specific recommendations based on what's missing
        if results['trades_per_year'] < 30:
            print("LOW TRADE FREQUENCY DETECTED:")
            print()
            print("Current parameters are very conservative (high quality, low frequency).")
            print("To increase trade frequency while maintaining quality:")
            print()
            print("Option 1: Relax thresholds in DeteriorationDetector:")
            print("  - min_regime_age: 20 → 15 days (catch deterioration earlier)")
            print("  - min_confidence_decline: -0.05 → -0.03 (less strict decline)")
            print("  - min_deterioration_score: 0.3 → 0.25 (lower entry threshold)")
            print()
            print("Option 2: Test longer time period:")
            print("  - Try 2015-2024 to capture more bull markets")
            print("  - 2020-2024 was mostly sideways/choppy")
            print()
            print("Option 3: Accept low frequency:")
            print(f"  - Current Sharpe {results['sharpe_estimate']:.1f} is excellent")
            print("  - 2 high-quality trades may be all the data provides")
            print("  - Focus on other strategies for frequency")
            print()

        print("Next steps:")
        print("  1. Review warnings above")
        print("  2. Consider parameter tuning (see recommendations)")
        print("  3. Deploy to QuantConnect for full validation")
    else:
        print("❌ VALIDATION FAILED - Strategy Needs Improvement")
        print()
        print("Next steps:")
        print("  1. Review failed checks above")
        print("  2. Adjust deterioration detection parameters")
        print("  3. Consider different regime detection settings")
        print("  4. Re-run local test")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Regime Deterioration Short strategy locally'
    )
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker')
    parser.add_argument('--n-states', type=int, default=3, help='Number of HMM states')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--temporal-isolation', action='store_true',
                       help='Use temporal isolation (RECOMMENDED for realistic results)')

    args = parser.parse_args()

    # Show warning if not using temporal isolation
    if not args.temporal_isolation:
        print()
        print("=" * 70)
        print("⚠️  LOOKAHEAD BIAS WARNING")
        print("=" * 70)
        print()
        print("You are running WITHOUT temporal isolation!")
        print("Results will be unrealistically optimistic due to lookahead bias.")
        print()
        print("For realistic results, use:")
        print("  python examples/quickstart/test_deterioration_strategy.py --temporal-isolation")
        print()
        print("Or use the walk-forward backtest:")
        print("  python examples/quickstart/test_deterioration_walkforward.py")
        print()
        print("=" * 70)
        print()

    test_deterioration_strategy(
        ticker=args.ticker,
        n_states=args.n_states,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        use_temporal_isolation=args.temporal_isolation
    )
