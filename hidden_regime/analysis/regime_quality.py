"""
Regime quality assessment and tradeable regime identification.

This module provides data-driven regime quality metrics to identify
which regimes are suitable for trading, replacing heuristic string matching
with actual performance analysis.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd


class RegimeQualityAnalyzer:
    """
    Analyze regime quality for trading decisions.

    Replaces string-matching heuristics with actual performance metrics
    to identify tradeable regimes.
    """

    def __init__(
        self,
        min_sharpe: float = 0.3,
        min_return_annual: float = 0.05,
        min_days: int = 20,
        min_persistence: float = 0.6,
    ):
        """
        Initialize regime quality analyzer.

        Args:
            min_sharpe: Minimum annualized Sharpe ratio for tradeable regime
            min_return_annual: Minimum annualized return (5% = 0.05)
            min_days: Minimum observations required per regime
            min_persistence: Minimum persistence score (0-1)
        """
        self.min_sharpe = min_sharpe
        self.min_return_annual = min_return_annual
        self.min_days = min_days
        self.min_persistence = min_persistence

    def identify_tradeable_regimes(
        self,
        regime_data: pd.DataFrame,
        price_data: pd.DataFrame,
        strategy_type: str = 'bull',
    ) -> Dict[str, Dict]:
        """
        Identify which regimes are suitable for trading based on actual performance.

        Args:
            regime_data: DataFrame with regime predictions and confidence
            price_data: DataFrame with price data
            strategy_type: 'bull' (positive returns), 'bear' (negative returns), or 'any'

        Returns:
            Dictionary mapping regime names to their quality metrics
        """
        if regime_data.empty or price_data.empty:
            return {}

        # Calculate returns
        returns = price_data['close'].pct_change().dropna() if 'close' in price_data.columns else None
        if returns is None:
            # Try 'price' column
            returns = price_data['price'].pct_change().dropna() if 'price' in price_data.columns else None

        if returns is None:
            warnings.warn("No price column found, cannot identify tradeable regimes")
            return {}

        # Align data
        common_index = regime_data.index.intersection(returns.index)
        if len(common_index) == 0:
            warnings.warn("No common dates between regime and price data")
            return {}

        aligned_regimes = regime_data.loc[common_index]
        aligned_returns = returns.loc[common_index]

        # Get regime column
        regime_col = 'regime_name' if 'regime_name' in aligned_regimes.columns else 'predicted_state'

        # Calculate metrics per regime
        tradeable_regimes = {}

        for regime_name in aligned_regimes[regime_col].unique():
            regime_mask = aligned_regimes[regime_col] == regime_name
            regime_returns = aligned_returns[regime_mask]
            regime_rows = aligned_regimes[regime_mask]

            # Skip if insufficient data
            if len(regime_returns) < self.min_days:
                continue

            # Calculate performance metrics
            metrics = self._calculate_regime_metrics(
                regime_returns,
                regime_rows,
                regime_name
            )

            # Check if tradeable based on strategy type
            if self._is_tradeable(metrics, strategy_type):
                tradeable_regimes[regime_name] = metrics

        return tradeable_regimes

    def _calculate_regime_metrics(
        self,
        returns: pd.Series,
        regime_data: pd.DataFrame,
        regime_name: str,
    ) -> Dict:
        """Calculate comprehensive metrics for a regime."""
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()

        # Annualized metrics
        annual_return = mean_return * 252
        annual_vol = std_return * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Downside risk (for Sortino)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        downside_annual = downside_std * np.sqrt(252)
        sortino = annual_return / downside_annual if downside_annual > 0 else 0

        # Win rate
        win_rate = (returns > 0).mean()

        # Persistence (average confidence)
        avg_confidence = regime_data['confidence'].mean() if 'confidence' in regime_data.columns else 0.5

        # Calculate regime persistence score
        persistence_score = self._calculate_persistence(regime_data, regime_name)

        return {
            'regime_name': regime_name,
            'days': len(returns),
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'persistence_score': persistence_score,
            'mean_daily_return': mean_return,
            'max_drawdown': self._calculate_max_drawdown(returns),
        }

    def _calculate_persistence(self, regime_data: pd.DataFrame, regime_name: str) -> float:
        """
        Calculate regime persistence score (0-1).

        Higher score means regime tends to persist once established.
        Based on average run length vs random switches.
        """
        regime_col = 'regime_name' if 'regime_name' in regime_data.columns else 'predicted_state'

        # Calculate run lengths
        runs = []
        current_run = 0

        for val in regime_data[regime_col]:
            if val == regime_name:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        if not runs:
            return 0.0

        # Average run length
        avg_run = np.mean(runs)

        # Normalize: runs of 1 day = 0, runs of 20+ days = 1.0
        persistence = min(1.0, (avg_run - 1) / 19.0)

        return persistence

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _is_tradeable(self, metrics: Dict, strategy_type: str) -> bool:
        """
        Determine if regime meets quality criteria for trading.

        Args:
            metrics: Regime performance metrics
            strategy_type: 'bull' (long), 'bear' (short), or 'any'
        """
        # Check basic requirements
        if metrics['days'] < self.min_days:
            return False

        if metrics['persistence_score'] < self.min_persistence:
            return False

        # Check strategy-specific requirements
        if strategy_type == 'bull':
            # For long positions: need positive returns and decent Sharpe
            return (
                metrics['annual_return'] >= self.min_return_annual and
                metrics['sharpe_ratio'] >= self.min_sharpe
            )
        elif strategy_type == 'bear':
            # For short positions: need negative returns and decent Sharpe
            return (
                metrics['annual_return'] <= -self.min_return_annual and
                abs(metrics['sharpe_ratio']) >= self.min_sharpe
            )
        elif strategy_type == 'any':
            # Any strong directional regime
            return (
                abs(metrics['annual_return']) >= self.min_return_annual and
                abs(metrics['sharpe_ratio']) >= self.min_sharpe
            )
        else:
            return False

    def rank_regimes_by_quality(
        self,
        regime_metrics: Dict[str, Dict],
        metric: str = 'sharpe_ratio',
    ) -> List[Tuple[str, float]]:
        """
        Rank regimes by a quality metric.

        Args:
            regime_metrics: Output from identify_tradeable_regimes
            metric: Metric to rank by ('sharpe_ratio', 'sortino_ratio', 'annual_return')

        Returns:
            List of (regime_name, metric_value) tuples, sorted best to worst
        """
        rankings = [
            (name, metrics[metric])
            for name, metrics in regime_metrics.items()
            if metric in metrics
        ]

        # Sort descending (higher is better)
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def apply_persistence_filter(
        self,
        regime_data: pd.DataFrame,
        min_consecutive_days: int = 3,
    ) -> pd.DataFrame:
        """
        Apply persistence filtering to reduce regime switching noise.

        Only trust regime changes that persist for min_consecutive_days.

        Args:
            regime_data: DataFrame with regime predictions
            min_consecutive_days: Minimum days to trust a regime switch

        Returns:
            Filtered DataFrame with more persistent regimes
        """
        if regime_data.empty:
            return regime_data

        regime_col = 'regime_name' if 'regime_name' in regime_data.columns else 'predicted_state'

        # Create copy
        filtered = regime_data.copy()

        # Track current regime and its duration
        current_regime = None
        regime_start_idx = 0

        for i, (idx, row) in enumerate(filtered.iterrows()):
            regime = row[regime_col]

            # Regime changed
            if regime != current_regime:
                # Check if previous regime lasted long enough
                if current_regime is not None:
                    regime_duration = i - regime_start_idx

                    if regime_duration < min_consecutive_days:
                        # Revert to previous regime (too short)
                        # Find what came before
                        if regime_start_idx > 0:
                            prev_idx = filtered.index[regime_start_idx - 1]
                            prev_regime = filtered.loc[prev_idx, regime_col]

                            # Revert all short-lived regime observations
                            for j in range(regime_start_idx, i):
                                revert_idx = filtered.index[j]
                                filtered.loc[revert_idx, regime_col] = prev_regime

                # Start tracking new regime
                current_regime = regime
                regime_start_idx = i

        return filtered

    def suggest_optimal_n_states(
        self,
        price_data: pd.DataFrame,
        test_range: Tuple[int, int] = (2, 5),
    ) -> Dict[int, Dict]:
        """
        Test different n_states values and suggest optimal.

        NOTE: This is computationally expensive as it trains multiple HMMs.
        Use sparingly.

        Args:
            price_data: DataFrame with price data
            test_range: (min_states, max_states) to test

        Returns:
            Dictionary with metrics for each n_states value
        """
        # This would require training multiple HMMs
        # Placeholder for now - can be implemented later
        warnings.warn(
            "suggest_optimal_n_states requires training multiple HMMs. "
            "Not implemented yet. Use cross-validation in pipeline instead."
        )
        return {}
