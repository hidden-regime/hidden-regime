"""
Advanced example: Strategy Composition and A/B Testing

Demonstrates how to:
1. Compose multiple strategies using the decorator pattern
2. Run A/B tests comparing different strategy variants
3. Use RegimeLabel objects as single source of truth
4. Evaluate performance across different market regimes

Key concepts:
- RegimeLabel: Immutable regime semantics from interpreter
- Strategy composition: Decorator pattern for flexible combinations
- Factory pattern: Create strategies from configuration
- Type-safe signals: List[TradingSignal] for audit trail

This example shows why this architecture solves the core problems:
A. Single source of truth: RegimeLabel flows through entire pipeline
B. Strategy flexibility: Compose strategies without code duplication
C. Testability: Easy to compare different strategy variants
"""

import pandas as pd
from typing import List, Dict

from hidden_regime import create_financial_pipeline
from hidden_regime.config.strategy import StrategyConfiguration
from hidden_regime.factories.components import ComponentFactory
from hidden_regime.interpreter.regime_label_builder import RegimeLabelBuilder
from hidden_regime.interpreter.regime_types import (
    RegimeType,
    RegimeCharacteristics,
    RegimeLabel,
)
from hidden_regime.signals.types import TradingSignal
from hidden_regime.strategies import (
    RegimeFollowingStrategy,
    ContrarianStrategy,
    ConfidenceWeightedStrategy,
    VolatilityAdjustedStrategy,
)


class StrategyABTest:
    """
    Framework for A/B testing trading strategies.

    Allows comparing multiple strategy variants on the same regime detection
    output without redundant interpretation.
    """

    def __init__(self, ticker: str, n_states: int = 3):
        """Initialize A/B test framework.

        Args:
            ticker: Stock ticker to analyze
            n_states: Number of HMM states
        """
        self.ticker = ticker
        self.n_states = n_states
        self.pipeline = create_financial_pipeline(ticker, n_states=n_states)
        self.factory = ComponentFactory()

        # Strategy results storage
        self.results: Dict[str, List[TradingSignal]] = {}
        self.performance_metrics: Dict[str, Dict] = {}

    def run_pipeline(self) -> pd.DataFrame:
        """
        Run the regime detection pipeline.

        Returns:
            DataFrame with regime interpretation (regime_label, regime_strength, etc.)
        """
        print(f"\n{'='*70}")
        print(f"Running regime detection for {self.ticker}")
        print(f"{'='*70}")

        result = self.pipeline.update()

        # Extract interpreter output with RegimeLabel objects
        interpreter_output = self.pipeline.component_outputs.get("interpreter")

        if interpreter_output is None or interpreter_output.empty:
            raise ValueError("Pipeline produced no interpreter output")

        print(f"Pipeline complete. Detected {len(interpreter_output)} time periods")
        print(f"Current regime: {interpreter_output.iloc[-1].get('regime_label', 'Unknown')}")

        return interpreter_output

    def add_strategy_variant(self, name: str, config: StrategyConfiguration) -> None:
        """
        Add a strategy variant to test.

        Args:
            name: Human-readable name for this variant (e.g., "Regime Following",
                "Confidence Weighted", etc.)
            config: StrategyConfiguration specifying the strategy
        """
        if not hasattr(self, "_strategy_variants"):
            self._strategy_variants = {}

        print(f"\nAdding strategy variant: {name}")
        print(f"  Type: {config.strategy_type}")
        if config.base_strategy:
            print(f"  Base: {config.base_strategy}")
        print(f"  Thresholds: long={config.long_confidence_threshold}, "
              f"short={config.short_confidence_threshold}")

        # Create strategy from config
        strategy = self.factory.create_strategy_component(config)
        self._strategy_variants[name] = strategy

    def evaluate_strategies(self, interpreter_output: pd.DataFrame) -> None:
        """
        Evaluate all strategy variants on the same regime detection output.

        This demonstrates the core architecture benefit: multiple strategies
        can analyze the same RegimeLabel objects without re-interpretation.

        Args:
            interpreter_output: DataFrame with regime_label column
        """
        if not hasattr(self, "_strategy_variants"):
            self._strategy_variants = {}

        print(f"\n{'='*70}")
        print(f"Evaluating {len(self._strategy_variants)} strategy variants")
        print(f"{'='*70}")

        for variant_name, strategy in self._strategy_variants.items():
            print(f"\nEvaluating: {variant_name}")
            print(f"  Strategy: {strategy.__class__.__name__}")

            signals: List[TradingSignal] = []

            for idx, row in interpreter_output.iterrows():
                regime = row.get("regime_label")

                if regime is None or not isinstance(regime, RegimeLabel):
                    continue

                # Get signal from strategy
                signal = strategy.get_signal_for_regime(regime)

                # Apply risk management
                if strategy.risk_management.prevent_shorts and signal.direction == "short":
                    signal = signal.with_direction("neutral").with_adjusted_position(0.0)

                signals.append(signal)

            # Store results
            self.results[variant_name] = signals

            # Calculate metrics
            metrics = self._calculate_metrics(signals, variant_name)
            self.performance_metrics[variant_name] = metrics

            print(f"  Signals generated: {len(signals)}")
            print(f"  Long signals: {sum(1 for s in signals if s.direction == 'long')}")
            print(f"  Short signals: {sum(1 for s in signals if s.direction == 'short')}")
            print(f"  Neutral signals: {sum(1 for s in signals if s.direction == 'neutral')}")
            print(f"  Avg position size: {metrics['avg_position_size']:.2f}")
            print(f"  Avg confidence: {metrics['avg_confidence']:.2%}")

    def _calculate_metrics(self, signals: List[TradingSignal], variant_name: str) -> Dict:
        """Calculate performance metrics for a strategy variant."""
        if not signals:
            return {}

        long_signals = [s for s in signals if s.direction == "long"]
        short_signals = [s for s in signals if s.direction == "short"]
        neutral_signals = [s for s in signals if s.direction == "neutral"]

        return {
            "total_signals": len(signals),
            "long_count": len(long_signals),
            "short_count": len(short_signals),
            "neutral_count": len(neutral_signals),
            "long_pct": len(long_signals) / len(signals) if signals else 0,
            "short_pct": len(short_signals) / len(signals) if signals else 0,
            "neutral_pct": len(neutral_signals) / len(signals) if signals else 0,
            "avg_position_size": sum(s.position_size for s in signals) / len(signals),
            "max_position_size": max(s.position_size for s in signals),
            "min_position_size": min(s.position_size for s in signals),
            "avg_confidence": sum(s.confidence for s in signals) / len(signals),
            "max_confidence": max(s.confidence for s in signals),
            "min_confidence": min(s.confidence for s in signals),
        }

    def print_comparison(self) -> None:
        """Print comparison of all strategy variants."""
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        # Print metrics table
        metrics_df = pd.DataFrame(self.performance_metrics).T

        print("Signal Distribution:")
        print(metrics_df[["total_signals", "long_pct", "short_pct", "neutral_pct"]])

        print("\n\nPosition Sizing:")
        print(metrics_df[["avg_position_size", "max_position_size"]])

        print("\n\nConfidence Levels:")
        print(metrics_df[["avg_confidence", "max_confidence", "min_confidence"]])

    def compare_signals(self, variant1: str, variant2: str) -> pd.DataFrame:
        """
        Compare signals from two strategy variants.

        Args:
            variant1: Name of first variant
            variant2: Name of second variant

        Returns:
            DataFrame showing signal differences
        """
        signals1 = self.results.get(variant1, [])
        signals2 = self.results.get(variant2, [])

        if not signals1 or not signals2:
            return pd.DataFrame()

        # Align signals by index
        min_len = min(len(signals1), len(signals2))

        comparison = pd.DataFrame({
            f"{variant1}_direction": [s.direction for s in signals1[:min_len]],
            f"{variant1}_position": [s.position_size for s in signals1[:min_len]],
            f"{variant2}_direction": [s.direction for s in signals2[:min_len]],
            f"{variant2}_position": [s.position_size for s in signals2[:min_len]],
        })

        # Calculate agreement
        direction_agreement = sum(
            1 for d1, d2 in zip(
                comparison[f"{variant1}_direction"],
                comparison[f"{variant2}_direction"],
            )
            if d1 == d2
        )

        print(f"\nSignal Agreement: {direction_agreement}/{min_len} "
              f"({direction_agreement/min_len:.1%})")

        return comparison


def main():
    """Run strategy composition and A/B testing example."""
    ticker = "SPY"

    # Create A/B test framework
    test = StrategyABTest(ticker, n_states=3)

    # Define strategy variants to test
    print("\nDefining strategy variants...")

    # Variant 1: Pure Regime Following
    test.add_strategy_variant(
        "Regime Following (Base)",
        StrategyConfiguration(
            strategy_type="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65,
        ),
    )

    # Variant 2: Regime Following with Confidence Weighting
    test.add_strategy_variant(
        "Regime Following + Confidence Weighted",
        StrategyConfiguration(
            strategy_type="confidence_weighted",
            base_strategy="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65,
        ),
    )

    # Variant 3: Regime Following with Volatility Adjustment
    test.add_strategy_variant(
        "Regime Following + Volatility Adjusted",
        StrategyConfiguration(
            strategy_type="volatility_adjusted",
            base_strategy="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65,
        ),
    )

    # Variant 4: Contrarian Strategy
    test.add_strategy_variant(
        "Contrarian (Fade Regime)",
        StrategyConfiguration(
            strategy_type="contrarian",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65,
        ),
    )

    # Run regime detection pipeline
    print("\nRunning regime detection...")
    interpreter_output = test.run_pipeline()

    # Evaluate all strategies on the same regime detection
    test.evaluate_strategies(interpreter_output)

    # Print comparison
    test.print_comparison()

    # Show detailed comparison between variants
    print("\n" + "="*70)
    print("DETAILED VARIANT COMPARISON")
    print("="*70)

    test.compare_signals("Regime Following (Base)", "Regime Following + Confidence Weighted")

    print("\n\nKey Insights:")
    print("1. Single source of truth: All strategies analyzed the same RegimeLabel objects")
    print("2. No re-interpretation: RegimeLabel semantics don't change per strategy")
    print("3. Easy composition: Strategies can be mixed and matched without code duplication")
    print("4. Type-safe: TradingSignal objects provide full audit trail")
    print("5. A/B testable: Compare variants instantly on identical data")


if __name__ == "__main__":
    main()
