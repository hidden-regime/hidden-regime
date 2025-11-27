"""Tests for Strategy component factory creation."""

import pytest

from hidden_regime.config.strategy import ComposedStrategyConfiguration, StrategyConfiguration
from hidden_regime.factories.components import ComponentFactory
from hidden_regime.strategies import (
    ConfidenceWeightedStrategy,
    ContrarianStrategy,
    MultiTimeframeAlignmentStrategy,
    RegimeFollowingStrategy,
    VolatilityAdjustedStrategy,
)


class TestStrategyFactory:
    """Test suite for strategy factory creation."""

    @pytest.fixture
    def factory(self):
        """Create a component factory."""
        return ComponentFactory()

    def test_create_regime_following_strategy(self, factory):
        """Test creating regime following strategy from config."""
        config = StrategyConfiguration(
            strategy_type="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.70,
        )

        strategy = factory.create_strategy_component(config)

        assert isinstance(strategy, RegimeFollowingStrategy)
        assert strategy.long_confidence_threshold == 0.65
        assert strategy.short_confidence_threshold == 0.70

    def test_create_contrarian_strategy(self, factory):
        """Test creating contrarian strategy from config."""
        config = StrategyConfiguration(
            strategy_type="contrarian",
            long_confidence_threshold=0.60,
            short_confidence_threshold=0.65,
        )

        strategy = factory.create_strategy_component(config)

        assert isinstance(strategy, ContrarianStrategy)
        assert strategy.confidence_threshold == 0.60

    def test_create_confidence_weighted_strategy(self, factory):
        """Test creating confidence weighted decorator strategy."""
        config = StrategyConfiguration(
            strategy_type="confidence_weighted",
            base_strategy="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.70,
        )

        strategy = factory.create_strategy_component(config)

        assert isinstance(strategy, ConfidenceWeightedStrategy)
        assert isinstance(strategy.wrapped_strategy, RegimeFollowingStrategy)

    def test_create_volatility_adjusted_strategy(self, factory):
        """Test creating volatility adjusted decorator strategy."""
        config = StrategyConfiguration(
            strategy_type="volatility_adjusted",
            base_strategy="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.70,
        )

        strategy = factory.create_strategy_component(config)

        assert isinstance(strategy, VolatilityAdjustedStrategy)
        assert isinstance(strategy.wrapped_strategy, RegimeFollowingStrategy)

    def test_create_multi_timeframe_alignment_strategy(self, factory):
        """Test creating multi-timeframe alignment decorator strategy."""
        config = StrategyConfiguration(
            strategy_type="multi_timeframe_alignment",
            base_strategy="contrarian",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.70,
        )

        strategy = factory.create_strategy_component(config)

        assert isinstance(strategy, MultiTimeframeAlignmentStrategy)
        assert isinstance(strategy.wrapped_strategy, ContrarianStrategy)

    def test_nested_decorator_strategies_via_composition(self, factory):
        """Test creating nested decorator strategies by manually composing.

        Note: Configuration validation prevents decorators from wrapping decorators.
        For complex compositions, instantiate strategies directly.
        """
        base_strategy = factory.create_strategy_component(
            StrategyConfiguration(strategy_type="regime_following")
        )

        confidence_strategy = ConfidenceWeightedStrategy(wrapped_strategy=base_strategy)
        volatility_strategy = VolatilityAdjustedStrategy(
            wrapped_strategy=confidence_strategy
        )

        assert isinstance(volatility_strategy, VolatilityAdjustedStrategy)
        assert isinstance(volatility_strategy.wrapped_strategy, ConfidenceWeightedStrategy)
        assert isinstance(
            volatility_strategy.wrapped_strategy.wrapped_strategy, RegimeFollowingStrategy
        )

    def test_invalid_strategy_type(self, factory):
        """Test that invalid strategy type raises error."""
        with pytest.raises(ValueError):
            StrategyConfiguration(strategy_type="invalid_type")

    def test_decorator_requires_base_strategy(self, factory):
        """Test that decorator strategy without base_strategy raises error."""
        with pytest.raises(ValueError):
            StrategyConfiguration(strategy_type="confidence_weighted")

    def test_composed_strategy_configuration(self, factory):
        """Test using ComposedStrategyConfiguration for complex setups."""
        base_config = StrategyConfiguration(strategy_type="regime_following")

        decorator_configs = [
            StrategyConfiguration(
                strategy_type="confidence_weighted",
                base_strategy="regime_following",
            ),
            StrategyConfiguration(
                strategy_type="volatility_adjusted",
                base_strategy="confidence_weighted",
            ),
        ]

        composed_config = ComposedStrategyConfiguration(
            base_strategy_config=base_config,
            decorator_layers=decorator_configs,
        )

        assert composed_config.base_strategy_config.strategy_type == "regime_following"
        assert len(composed_config.decorator_layers) == 2
        assert composed_config.decorator_layers[0].strategy_type == "confidence_weighted"
        assert composed_config.decorator_layers[1].strategy_type == "volatility_adjusted"

    def test_strategy_config_serialization(self):
        """Test strategy configuration serialization and deserialization."""
        original_config = StrategyConfiguration(
            strategy_type="regime_following",
            long_confidence_threshold=0.65,
            short_confidence_threshold=0.70,
        )

        config_dict = original_config.to_dict()
        restored_config = StrategyConfiguration.from_dict(config_dict)

        assert restored_config.strategy_type == original_config.strategy_type
        assert (
            restored_config.long_confidence_threshold
            == original_config.long_confidence_threshold
        )
        assert (
            restored_config.short_confidence_threshold
            == original_config.short_confidence_threshold
        )

    def test_composed_strategy_serialization(self):
        """Test composed strategy configuration serialization."""
        base_config = StrategyConfiguration(strategy_type="regime_following")
        decorator_configs = [
            StrategyConfiguration(
                strategy_type="confidence_weighted",
                base_strategy="regime_following",
            )
        ]

        original = ComposedStrategyConfiguration(
            base_strategy_config=base_config,
            decorator_layers=decorator_configs,
        )

        config_dict = original.to_dict()
        restored = ComposedStrategyConfiguration.from_dict(config_dict)

        assert restored.base_strategy_config.strategy_type == "regime_following"
        assert len(restored.decorator_layers) == 1
        assert restored.decorator_layers[0].strategy_type == "confidence_weighted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
