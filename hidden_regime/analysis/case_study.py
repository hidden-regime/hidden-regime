"""
Case study analysis for comprehensive regime detection evaluation.

Provides analysis functions for comparing HMM regime detection against
buy-and-hold strategies and technical indicators with detailed performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .performance import RegimePerformanceAnalyzer
from .technical_indicators import TechnicalIndicatorAnalyzer
from ..utils.exceptions import AnalysisError


class CaseStudyAnalyzer:
    """
    Comprehensive analyzer for case study performance comparison.

    Compares HMM regime detection strategies against buy-and-hold and
    technical indicators with detailed risk and return metrics.
    """

    def __init__(self):
        """Initialize case study analyzer."""
        self.performance_analyzer = RegimePerformanceAnalyzer()
        self.technical_analyzer = TechnicalIndicatorAnalyzer()

    def analyze_hmm_strategy_performance(
        self,
        price_data: pd.DataFrame,
        regime_data: pd.DataFrame,
        price_column: str = 'close',
        regime_column: str = 'predicted_state',
        confidence_column: str = 'confidence',
        strategy_type: str = 'regime_following'
    ) -> Dict[str, Any]:
        """
        Analyze performance of HMM-based trading strategy.

        Args:
            price_data: DataFrame with price data
            regime_data: DataFrame with regime predictions
            price_column: Column name for price data
            regime_column: Column name for regime predictions
            confidence_column: Column name for confidence scores
            strategy_type: Type of strategy ('regime_following', 'regime_contrarian', 'confidence_weighted')

        Returns:
            Dictionary with strategy performance metrics
        """
        # Align data
        aligned_data = price_data.join(regime_data[[regime_column, confidence_column]], how='inner')

        if len(aligned_data) == 0:
            raise AnalysisError("No aligned data between price and regime data")

        # Calculate returns
        aligned_data['returns'] = aligned_data[price_column].pct_change()

        # Generate trading positions based on strategy type
        n_states = int(aligned_data[regime_column].max()) + 1
        positions = self._generate_hmm_positions(
            aligned_data, regime_column, confidence_column, n_states, strategy_type
        )

        # Calculate strategy returns
        strategy_returns = aligned_data['returns'] * positions
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) == 0:
            return self._create_empty_performance_dict()

        # Calculate comprehensive performance metrics with proper trade counting
        performance_metrics = self._calculate_performance_metrics(strategy_returns, positions)

        # Add strategy-specific metrics
        performance_metrics.update({
            'strategy_type': strategy_type,
            'n_states': n_states,
            'avg_position': positions.mean(),
            'position_changes': (positions.diff() != 0).sum(),
            'regime_distribution': aligned_data[regime_column].value_counts().to_dict(),
            'avg_confidence': aligned_data[confidence_column].mean() if confidence_column in aligned_data.columns else None
        })

        return performance_metrics

    def analyze_buy_hold_performance(
        self,
        price_data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Analyze buy-and-hold strategy performance.

        Args:
            price_data: DataFrame with price data
            price_column: Column name for price data

        Returns:
            Dictionary with buy-and-hold performance metrics
        """
        # Calculate returns
        returns = price_data[price_column].pct_change().dropna()

        if len(returns) == 0:
            return self._create_empty_performance_dict()

        # Buy-and-hold: position = 1.0 for entire period (exactly 2 trades: buy at start, sell at end)
        positions = pd.Series(1.0, index=returns.index)

        # Calculate performance metrics with proper trade counting
        performance_metrics = self._calculate_performance_metrics(returns, positions)
        performance_metrics['strategy_type'] = 'buy_and_hold'

        # Override trade count to exactly 2 for buy-and-hold
        performance_metrics['num_trades'] = 2

        return performance_metrics

    def analyze_technical_indicator_performance(
        self,
        price_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        price_column: str = 'close'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance of technical indicator strategies.

        Args:
            price_data: DataFrame with price data
            indicators: Dictionary of indicator_name -> indicator_values
            price_column: Column name for price data

        Returns:
            Dictionary mapping indicator names to performance metrics
        """
        results = {}

        # Calculate base returns
        price_data = price_data.copy()
        price_data['returns'] = price_data[price_column].pct_change()

        for indicator_name, indicator_values in indicators.items():
            try:
                # Generate signals from indicator
                signals = self._generate_indicator_signals(indicator_values, indicator_name)

                # Align with price data
                aligned_signals = signals.reindex(price_data.index).fillna(0)
                aligned_returns = price_data['returns'].reindex(signals.index).dropna()

                if len(aligned_returns) == 0:
                    continue

                # Calculate strategy returns
                strategy_returns = aligned_returns * aligned_signals
                strategy_returns = strategy_returns.dropna()

                if len(strategy_returns) == 0:
                    continue

                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(strategy_returns)
                performance_metrics.update({
                    'strategy_type': f'technical_indicator_{indicator_name}',
                    'indicator_name': indicator_name,
                    'avg_position': aligned_signals.mean(),
                    'position_changes': (aligned_signals.diff() != 0).sum(),
                    'signal_distribution': aligned_signals.value_counts().to_dict()
                })

                results[indicator_name] = performance_metrics

            except Exception as e:
                print(f"Warning: Failed to analyze indicator {indicator_name}: {e}")
                continue

        return results

    def compare_all_strategies(
        self,
        price_data: pd.DataFrame,
        regime_data: pd.DataFrame,
        indicators: Optional[Dict[str, pd.Series]] = None,
        hmm_strategy_types: List[str] = ['regime_following'],
        price_column: str = 'close',
        n_best_indicators: int = 5
    ) -> Dict[str, Any]:
        """
        Compare all strategies (HMM, buy-and-hold, technical indicators).

        Args:
            price_data: DataFrame with price data
            regime_data: DataFrame with regime predictions
            indicators: Dictionary of technical indicators (optional, deprecated)
            hmm_strategy_types: List of HMM strategy types to test
            price_column: Column name for price data
            n_best_indicators: Number of best technical indicators to include

        Returns:
            Comprehensive comparison results
        """
        all_results = {}

        # Analyze buy-and-hold
        try:
            buy_hold_results = self.analyze_buy_hold_performance(price_data, price_column)
            all_results['buy_and_hold'] = buy_hold_results
        except Exception as e:
            print(f"Warning: Buy-and-hold analysis failed: {e}")

        # Analyze HMM strategies
        for strategy_type in hmm_strategy_types:
            try:
                hmm_results = self.analyze_hmm_strategy_performance(
                    price_data, regime_data, price_column, strategy_type=strategy_type
                )
                all_results[f'hmm_{strategy_type}'] = hmm_results
            except Exception as e:
                print(f"Warning: HMM {strategy_type} analysis failed: {e}")

        # Analyze comprehensive technical indicators
        try:
            print("   📊 Analyzing comprehensive technical indicators...")

            # Calculate all technical indicator strategies
            all_indicator_results = self.technical_analyzer.analyze_all_indicator_strategies(
                price_data, price_column
            )

            # Select best N indicators based on Sharpe ratio
            best_indicators = self.technical_analyzer.select_best_indicators(
                all_indicator_results, n_best=n_best_indicators, ranking_metric='sharpe_ratio'
            )

            print(f"   📈 Selected top {len(best_indicators)} technical indicators:")
            for i, (indicator_name, sharpe) in enumerate(best_indicators):
                print(f"      {i+1}. {indicator_name}: Sharpe {sharpe:.3f}")

            # Add best indicators to results
            for indicator_name, _ in best_indicators:
                if indicator_name in all_indicator_results:
                    all_results[f'ta_{indicator_name}'] = all_indicator_results[indicator_name]

            # Store all indicator results for reference
            all_results['_all_technical_indicators'] = all_indicator_results
            all_results['_best_indicators_ranking'] = best_indicators

        except Exception as e:
            print(f"Warning: Comprehensive technical indicator analysis failed: {e}")

            # Fallback to legacy indicator analysis if provided
            if indicators:
                try:
                    indicator_results = self.analyze_technical_indicator_performance(
                        price_data, indicators, price_column
                    )
                    all_results.update(indicator_results)
                except Exception as e:
                    print(f"Warning: Legacy technical indicator analysis also failed: {e}")

        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(all_results)

        return {
            'individual_results': all_results,
            'comparison_summary': comparison_summary,
            'analysis_period': {
                'start': price_data.index.min().strftime('%Y-%m-%d'),
                'end': price_data.index.max().strftime('%Y-%m-%d'),
                'total_days': len(price_data)
            }
        }

    def _generate_hmm_positions(
        self,
        data: pd.DataFrame,
        regime_column: str,
        confidence_column: str,
        n_states: int,
        strategy_type: str
    ) -> pd.Series:
        """Generate trading positions based on HMM regime predictions."""
        positions = pd.Series(0.0, index=data.index)

        if strategy_type == 'regime_following':
            # Long in bull regimes, short in bear, neutral in sideways
            for i, regime in enumerate(data[regime_column]):
                if regime == n_states - 1:  # Highest regime = bull
                    positions.iloc[i] = 1.0
                elif regime == 0:  # Lowest regime = bear
                    positions.iloc[i] = -1.0
                else:  # Middle regimes = neutral
                    positions.iloc[i] = 0.0

        elif strategy_type == 'regime_contrarian':
            # Opposite of regime following
            for i, regime in enumerate(data[regime_column]):
                if regime == n_states - 1:  # Bull -> short
                    positions.iloc[i] = -1.0
                elif regime == 0:  # Bear -> long
                    positions.iloc[i] = 1.0
                else:  # Sideways -> neutral
                    positions.iloc[i] = 0.0

        elif strategy_type == 'confidence_weighted':
            # Scale positions by confidence
            base_positions = self._generate_hmm_positions(
                data, regime_column, confidence_column, n_states, 'regime_following'
            )
            if confidence_column in data.columns:
                confidence_weights = data[confidence_column].fillna(0.5)
                positions = base_positions * confidence_weights
            else:
                positions = base_positions

        return positions

    def _generate_indicator_signals(
        self,
        indicator_values: pd.Series,
        indicator_name: str
    ) -> pd.Series:
        """Generate trading signals from technical indicator values."""
        signals = pd.Series(0.0, index=indicator_values.index)

        if 'rsi' in indicator_name.lower():
            # RSI strategy: sell when > 70, buy when < 30
            signals[indicator_values > 70] = -1.0
            signals[indicator_values < 30] = 1.0

        elif 'macd' in indicator_name.lower():
            # MACD strategy: buy when positive, sell when negative
            if isinstance(indicator_values.iloc[0], (list, tuple, np.ndarray)):
                # MACD line - signal line
                macd_diff = pd.Series([
                    x[0] - x[1] if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1 else 0
                    for x in indicator_values.values
                ], index=indicator_values.index)
                signals[macd_diff > 0] = 1.0
                signals[macd_diff < 0] = -1.0
            else:
                signals[indicator_values > 0] = 1.0
                signals[indicator_values < 0] = -1.0

        elif 'bollinger' in indicator_name.lower():
            # Bollinger Bands: buy near lower band, sell near upper band
            if isinstance(indicator_values.iloc[0], (list, tuple, np.ndarray)):
                # Assume [upper, middle, lower] format
                for i, bands in enumerate(indicator_values.values):
                    if isinstance(bands, (list, tuple, np.ndarray)) and len(bands) >= 3:
                        # Mean reversion strategy
                        signals.iloc[i] = 0.0  # Default neutral
                # Simplified bollinger strategy
                signals = signals.fillna(0.0)
            else:
                # Single value bollinger indicator
                mean_val = indicator_values.mean()
                std_val = indicator_values.std()
                signals[indicator_values < mean_val - std_val] = 1.0  # Buy on low values
                signals[indicator_values > mean_val + std_val] = -1.0  # Sell on high values

        elif 'moving_average' in indicator_name.lower() or 'ma' in indicator_name.lower():
            # Moving average crossover strategy (simplified)
            ma_change = indicator_values.pct_change()
            signals[ma_change > 0.01] = 1.0  # Buy on strong uptrend
            signals[ma_change < -0.01] = -1.0  # Sell on strong downtrend

        else:
            # Generic momentum strategy
            indicator_change = indicator_values.pct_change()
            signals[indicator_change > indicator_change.quantile(0.7)] = 1.0
            signals[indicator_change < indicator_change.quantile(0.3)] = -1.0

        return signals

    def _calculate_performance_metrics(self, returns: pd.Series, positions: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for a returns series."""
        if len(returns) == 0:
            return self._create_empty_performance_dict()

        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)

            # Risk metrics
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)

            # Additional metrics
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
            avg_loss = returns[returns < 0].mean() if (returns < 0).sum() > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            # Skewness and kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else float('inf')

            # Calculate proper trade count from position changes
            num_trades = self._calculate_trade_count(positions) if positions is not None else len(returns)

            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'num_trades': num_trades,
                'start_date': returns.index[0].strftime('%Y-%m-%d'),
                'end_date': returns.index[-1].strftime('%Y-%m-%d')
            }

        except Exception as e:
            print(f"Warning: Error calculating performance metrics: {e}")
            return self._create_empty_performance_dict()

    def _calculate_trade_count(self, positions: pd.Series) -> int:
        """Calculate number of trades from position changes."""
        if positions is None or len(positions) == 0:
            return 0

        # Count position changes (including initial position)
        position_changes = (positions != positions.shift(1)).sum()

        # Adjust for initial position (if not zero, it's a trade)
        if len(positions) > 0 and positions.iloc[0] != 0:
            trades = position_changes
        else:
            trades = position_changes

        # Add final trade if ending with non-zero position
        if len(positions) > 0 and positions.iloc[-1] != 0:
            trades += 1

        return int(trades)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _create_empty_performance_dict(self) -> Dict[str, Any]:
        """Create empty performance dictionary for failed calculations."""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'num_trades': 0,
            'start_date': 'N/A',
            'end_date': 'N/A'
        }

    def _generate_comparison_summary(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary comparison across all strategies."""
        if not all_results:
            return {'error': 'No results to compare'}

        # Extract key metrics for comparison
        comparison_metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']

        summary = {}

        # Find best and worst performers for each metric
        for metric in comparison_metrics:
            metric_values = {}
            for strategy_name, results in all_results.items():
                if metric in results:
                    metric_values[strategy_name] = results[metric]

            if metric_values:
                if metric == 'max_drawdown':  # Lower is better
                    best_strategy = min(metric_values.keys(), key=lambda x: abs(metric_values[x]))
                    worst_strategy = max(metric_values.keys(), key=lambda x: abs(metric_values[x]))
                else:  # Higher is better
                    best_strategy = max(metric_values.keys(), key=lambda x: metric_values[x])
                    worst_strategy = min(metric_values.keys(), key=lambda x: metric_values[x])

                summary[f'best_{metric}'] = {
                    'strategy': best_strategy,
                    'value': metric_values[best_strategy]
                }
                summary[f'worst_{metric}'] = {
                    'strategy': worst_strategy,
                    'value': metric_values[worst_strategy]
                }

        # Overall ranking based on Sharpe ratio
        sharpe_rankings = {}
        for strategy_name, results in all_results.items():
            if 'sharpe_ratio' in results:
                sharpe_rankings[strategy_name] = results['sharpe_ratio']

        if sharpe_rankings:
            sorted_strategies = sorted(sharpe_rankings.items(), key=lambda x: x[1], reverse=True)
            summary['strategy_ranking'] = sorted_strategies

        # Calculate strategy statistics
        summary['num_strategies'] = len(all_results)
        summary['hmm_strategies'] = [name for name in all_results.keys() if 'hmm' in name.lower()]
        summary['indicator_strategies'] = [name for name in all_results.keys()
                                         if name not in ['buy_and_hold'] and 'hmm' not in name.lower()]

        return summary

    def calculate_evolving_performance(
        self,
        price_data: pd.DataFrame,
        regime_data_sequence: List[pd.DataFrame],
        evaluation_dates: List[str],
        strategy_type: str = 'regime_following'
    ) -> List[Dict[str, Any]]:
        """
        Calculate performance metrics as they evolve over time during case study.

        Args:
            price_data: Complete price data
            regime_data_sequence: List of regime predictions for each evaluation date
            evaluation_dates: List of evaluation dates
            strategy_type: HMM strategy type

        Returns:
            List of performance dictionaries for each evaluation date
        """
        evolving_performance = []

        for i, (date, regime_data) in enumerate(zip(evaluation_dates, regime_data_sequence)):
            try:
                # Get data up to current date
                current_date_dt = pd.to_datetime(date)
                historical_price_data = price_data[price_data.index <= current_date_dt]
                historical_regime_data = regime_data[regime_data.index <= current_date_dt]

                if len(historical_price_data) < 30:  # Need minimum data
                    evolving_performance.append(self._create_empty_performance_dict())
                    continue

                # Calculate strategy performance up to this point
                strategy_performance = self.analyze_hmm_strategy_performance(
                    historical_price_data,
                    historical_regime_data,
                    strategy_type=strategy_type
                )

                # Add time-specific information
                strategy_performance.update({
                    'evaluation_date': date,
                    'days_analyzed': len(historical_price_data),
                    'cumulative_return': strategy_performance.get('total_return', 0)
                })

                evolving_performance.append(strategy_performance)

            except Exception as e:
                print(f"Warning: Performance calculation failed for {date}: {e}")
                evolving_performance.append(self._create_empty_performance_dict())

        return evolving_performance

    def generate_performance_report(
        self,
        comparison_results: Dict[str, Any],
        title: str = "Case Study Performance Report"
    ) -> str:
        """
        Generate markdown report summarizing performance comparison.

        Args:
            comparison_results: Results from compare_all_strategies
            title: Report title

        Returns:
            Markdown formatted report string
        """
        report_lines = [
            f"# {title}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Analysis period
        if 'analysis_period' in comparison_results:
            period = comparison_results['analysis_period']
            report_lines.extend([
                "## Analysis Period",
                "",
                f"- **Start Date**: {period.get('start', 'N/A')}",
                f"- **End Date**: {period.get('end', 'N/A')}",
                f"- **Total Days**: {period.get('total_days', 'N/A')}",
                ""
            ])

        # Strategy comparison summary
        if 'comparison_summary' in comparison_results:
            summary = comparison_results['comparison_summary']
            report_lines.extend([
                "## Strategy Performance Summary",
                "",
            ])

            # Strategy rankings
            if 'strategy_ranking' in summary:
                report_lines.extend([
                    "### Strategy Rankings (by Sharpe Ratio)",
                    "",
                ])
                for i, (strategy, sharpe) in enumerate(summary['strategy_ranking']):
                    report_lines.append(f"{i+1}. **{strategy}**: {sharpe:.3f}")
                report_lines.append("")

            # Best performers
            metrics_to_show = ['total_return', 'sharpe_ratio', 'max_drawdown']
            for metric in metrics_to_show:
                if f'best_{metric}' in summary:
                    best = summary[f'best_{metric}']
                    worst = summary.get(f'worst_{metric}', {})

                    report_lines.extend([
                        f"### {metric.replace('_', ' ').title()}",
                        "",
                        f"- **Best**: {best.get('strategy', 'N/A')} ({best.get('value', 0):.3f})",
                        f"- **Worst**: {worst.get('strategy', 'N/A')} ({worst.get('value', 0):.3f})",
                        ""
                    ])

        # Detailed results
        if 'individual_results' in comparison_results:
            report_lines.extend([
                "## Detailed Strategy Results",
                ""
            ])

            results = comparison_results['individual_results']

            # Filter out internal metadata from results
            filtered_results = {
                name: metrics for name, metrics in results.items()
                if not name.startswith('_') and isinstance(metrics, dict) and 'total_return' in metrics
            }

            for strategy_name, metrics in filtered_results.items():
                # Format strategy name nicely
                display_name = strategy_name.replace('_', ' ').replace('ta ', '').title()
                if strategy_name.startswith('ta_'):
                    display_name = f"Technical Indicator: {display_name}"
                elif strategy_name.startswith('hmm_'):
                    display_name = f"HMM {display_name}"

                report_lines.extend([
                    f"### {display_name}",
                    "",
                    f"- **Total Return**: {metrics.get('total_return', 0):.2%}",
                    f"- **Annual Return**: {metrics.get('annualized_return', metrics.get('annual_return', 0)):.2%}",
                    f"- **Annual Volatility**: {metrics.get('annualized_volatility', metrics.get('annual_volatility', 0)):.2%}",
                    f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.3f}",
                    f"- **Sortino Ratio**: {metrics.get('sortino_ratio', 0):.3f}",
                    f"- **Maximum Drawdown**: {metrics.get('maximum_drawdown', metrics.get('max_drawdown', 0)):.2%}",
                    f"- **Win Rate**: {metrics.get('win_rate', 0):.2%}",
                    f"- **Number of Trades**: {metrics.get('number_of_trades', metrics.get('num_trades', 0))}",
                    ""
                ])

            # Add technical indicator summary if available
            if '_best_indicators_ranking' in results:
                report_lines.extend([
                    "### Technical Indicator Rankings",
                    "",
                    "Best performing technical indicators (by Sharpe ratio):",
                    ""
                ])
                for i, (indicator_name, sharpe_ratio) in enumerate(results['_best_indicators_ranking'][:10]):
                    report_lines.append(f"{i+1}. **{indicator_name.replace('_', ' ').title()}**: {sharpe_ratio:.3f}")
                report_lines.append("")

        # Methodology
        report_lines.extend([
            "## Methodology",
            "",
            "This comprehensive analysis compares trading strategies across multiple categories:",
            "",
            "- **Buy and Hold**: Simple buy-and-hold strategy as baseline",
            "- **HMM Regime Following**: Long in bull regimes, short in bear, neutral in sideways",
            "- **Technical Indicators**: 15+ systematic technical analysis strategies including:",
            "  - Trend indicators (SMA, EMA, MACD, ADX, Aroon)",
            "  - Momentum indicators (RSI, Stochastic, Williams %R, CCI, ROC)",
            "  - Volatility indicators (Bollinger Bands, Keltner Channels, ATR)",
            "  - Volume indicators (VWAP, Volume SMA/EMA)",
            "",
            "Each technical indicator uses standard parameters and generates systematic buy/sell signals.",
            "Only the top 5 technical indicators (by Sharpe ratio) are included in final comparison.",
            "",
            "Performance metrics include return, volatility, Sharpe ratio, and risk measures.",
            "All calculations assume no transaction costs or slippage.",
            ""
        ])

        return "\n".join(report_lines)