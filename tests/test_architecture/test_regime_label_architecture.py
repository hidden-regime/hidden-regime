"""
Integration tests for the new RegimeLabel architecture.

Tests that RegimeLabel, Strategy, and TradingSignal work together correctly.
Verifies the complete pipeline: Interpreter → RegimeLabel → Strategy → TradingSignal
"""

import numpy as np
import pandas as pd
import pytest

from hidden_regime.interpreter.regime_label_builder import RegimeLabelBuilder
from hidden_regime.interpreter.regime_types import (
    RegimeCharacteristics,
    RegimeLabel,
    RegimeType,
    TradingSemantics,
)
from hidden_regime.signal_generation.signal_generator import SignalGenerator
from hidden_regime.signals.types import TradingSignal
from hidden_regime.strategies import (
    ConfidenceWeightedStrategy,
    ContrarianStrategy,
    RegimeFollowingStrategy,
    RiskManagementConfig,
)


class TestRegimeLabelArchitecture:
    """Test suite for regime label architecture integration."""

    @pytest.fixture
    def sample_bullish_characteristics(self):
        """Create sample bullish regime characteristics."""
        return RegimeCharacteristics(
            mean_daily_return=0.0008,
            annualized_return=0.20,
            daily_volatility=0.012,
            annualized_volatility=0.19,
            win_rate=0.55,
            max_drawdown=-0.05,
            return_skewness=0.2,
            return_kurtosis=3.0,
            sharpe_ratio=1.05,
            persistence_days=25.0,
            regime_strength=0.75,
            transition_volatility=0.015,
            transition_probs={"BEARISH": 0.15, "SIDEWAYS": 0.10, "BULLISH": 0.75},
            state_id=0,
        )

    @pytest.fixture
    def sample_bearish_characteristics(self):
        """Create sample bearish regime characteristics."""
        return RegimeCharacteristics(
            mean_daily_return=-0.0005,
            annualized_return=-0.13,
            daily_volatility=0.018,
            annualized_volatility=0.285,
            win_rate=0.45,
            max_drawdown=-0.12,
            return_skewness=-0.3,
            return_kurtosis=4.5,
            sharpe_ratio=-0.45,
            persistence_days=18.0,
            regime_strength=0.70,
            transition_volatility=0.020,
            transition_probs={"BULLISH": 0.20, "SIDEWAYS": 0.15, "BEARISH": 0.65},
            state_id=1,
        )

    def test_regime_label_creation(self, sample_bullish_characteristics):
        """Test creating RegimeLabel from characteristics."""
        trading_semantics = RegimeLabelBuilder._infer_trading_semantics(
            "BULLISH", sample_bullish_characteristics
        )

        label = RegimeLabel(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            color="#4575b4",
            characteristics=sample_bullish_characteristics,
            trading_semantics=trading_semantics,
            regime_strength=0.75,
        )

        assert label.name == "BULLISH"
        assert label.regime_type == RegimeType.BULLISH
        assert label.characteristics.mean_daily_return == 0.0008
        assert label.trading_semantics.bias == "positive"
        assert label.trading_semantics.typical_position_sign == 1

    def test_regime_following_strategy(self, sample_bullish_characteristics):
        """Test RegimeFollowingStrategy with bullish regime."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        strategy = RegimeFollowingStrategy(
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.65,
        )

        signal = strategy.get_signal_for_regime(label)

        assert signal.direction == "long"
        assert signal.position_size == 1.0
        assert signal.confidence > 0.7
        assert signal.regime_label == label
        assert signal.strategy_name == "RegimeFollowing"
        assert signal.valid

    def test_regime_following_strategy_with_risk_management(self, sample_bearish_characteristics):
        """Test RegimeFollowingStrategy with prevent_shorts."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BEARISH",
            regime_type=RegimeType.BEARISH,
            characteristics=sample_bearish_characteristics,
            regime_strength=0.70,
        )

        # Strategy that prevents shorts
        strategy = RegimeFollowingStrategy(
            short_confidence_threshold=0.65,
            risk_management=RiskManagementConfig(prevent_shorts=True),
        )

        signal = strategy.get_signal_for_regime(label)

        # SignalGenerator will convert short to neutral due to risk management
        assert signal.direction == "short"  # Raw signal is short
        assert signal.regime_label == label

    def test_confidence_weighted_strategy(self, sample_bullish_characteristics):
        """Test ConfidenceWeightedStrategy composition."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.70,  # Good confidence
        )

        base_strategy = RegimeFollowingStrategy()
        weighted_strategy = ConfidenceWeightedStrategy(base_strategy)

        signal = weighted_strategy.get_signal_for_regime(label)

        # Position should be scaled by confidence
        assert signal.position_size < 1.0  # Less than base position
        assert signal.position_size > 0.5  # But still significant
        assert signal.direction == "long"

    def test_contrarian_strategy(self, sample_bullish_characteristics):
        """Test ContrarianStrategy fades regime."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        strategy = ContrarianStrategy()
        signal = strategy.get_signal_for_regime(label)

        # Contrarian fades bullish → short
        assert signal.direction == "short"
        assert signal.valid

    def test_signal_generator_with_strategy(self, sample_bullish_characteristics):
        """Test SignalGenerator with Strategy."""
        # Create a simple DataFrame with RegimeLabel objects
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1),
                "state": [0],
                "regime_label": [label],
                "regime_strength": [0.75],
            }
        )

        strategy = RegimeFollowingStrategy()
        generator = SignalGenerator(strategy)

        signals = generator.update(df)

        assert len(signals) == 1
        assert isinstance(signals[0], TradingSignal)
        assert signals[0].direction == "long"
        assert signals[0].regime_label == label
        assert signals[0].strategy_name == "RegimeFollowing"

    def test_signal_generator_with_risk_constraints(self, sample_bullish_characteristics):
        """Test SignalGenerator applies risk management constraints."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        df = pd.DataFrame(
            {
                "regime_label": [label],
            }
        )

        # Strategy with max position size constraint
        strategy = RegimeFollowingStrategy(
            risk_management=RiskManagementConfig(max_position_size=0.5)
        )

        generator = SignalGenerator(strategy)
        signals = generator.update(df)

        # Position should be capped at 0.5
        assert signals[0].position_size == 0.5
        assert "max_position_capped" in signals[0].risk_management_applied

    def test_signal_generator_prevents_shorts(self, sample_bearish_characteristics):
        """Test SignalGenerator prevents shorts when configured."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BEARISH",
            regime_type=RegimeType.BEARISH,
            characteristics=sample_bearish_characteristics,
            regime_strength=0.70,
        )

        df = pd.DataFrame(
            {
                "regime_label": [label],
            }
        )

        strategy = RegimeFollowingStrategy(
            risk_management=RiskManagementConfig(prevent_shorts=True)
        )

        generator = SignalGenerator(strategy)
        signals = generator.update(df)

        # Short should be converted to neutral
        assert signals[0].direction == "neutral"
        assert signals[0].position_size == 0.0
        assert "short_prevented" in signals[0].risk_management_applied

    def test_strategy_composition_with_generator(self, sample_bullish_characteristics):
        """Test strategy composition works with SignalGenerator."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,  # Good confidence
        )

        df = pd.DataFrame(
            {
                "regime_label": [label],
            }
        )

        # Composed strategy: confidence-weighted regime following
        base_strategy = RegimeFollowingStrategy()
        composed_strategy = ConfidenceWeightedStrategy(base_strategy)

        generator = SignalGenerator(composed_strategy)
        signals = generator.update(df)

        # Position scaled by confidence (1.0 * 0.75 = 0.75)
        assert signals[0].position_size == 0.75
        assert signals[0].direction == "long"
        # The base strategy name is preserved through decoration
        assert "RegimeFollowing" in signals[0].strategy_name

    def test_signal_to_dataframe_conversion(self, sample_bullish_characteristics):
        """Test converting TradingSignal objects to DataFrame."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        df = pd.DataFrame(
            {
                "regime_label": [label],
            }
        )

        strategy = RegimeFollowingStrategy()
        generator = SignalGenerator(strategy)
        signals = generator.update(df)

        # Convert to DataFrame for backward compatibility
        signals_df = generator.to_dataframe(signals)

        assert len(signals_df) == 1
        assert "direction" in signals_df.columns
        assert "position_size" in signals_df.columns
        assert "regime_name" in signals_df.columns
        assert signals_df["regime_name"].iloc[0] == "BULLISH"

    def test_regime_label_immutability(self, sample_bullish_characteristics):
        """Test RegimeLabel is immutable."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        # Attempting to modify should raise error
        with pytest.raises(AttributeError):
            label.name = "BEARISH"

    def test_trading_signal_immutability_with_builders(self, sample_bullish_characteristics):
        """Test TradingSignal builder methods return new instances."""
        label = RegimeLabelBuilder.create_regime_label(
            name="BULLISH",
            regime_type=RegimeType.BULLISH,
            characteristics=sample_bullish_characteristics,
            regime_strength=0.75,
        )

        signal = TradingSignal(
            direction="long",
            position_size=1.0,
            confidence=0.75,
            regime_label=label,
            strategy_name="Test",
        )

        # Builder methods should return new instances
        signal2 = signal.with_adjusted_position(0.5)

        assert signal.position_size == 1.0  # Original unchanged
        assert signal2.position_size == 0.5  # New instance
        assert signal is not signal2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
