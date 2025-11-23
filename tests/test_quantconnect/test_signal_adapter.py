"""
Unit tests for QuantConnect signal adapters.

Tests:
- TradingSignal dataclass
- RegimeSignalAdapter
- MultiAssetSignalAdapter
"""
import pytest
import numpy as np
from dataclasses import asdict

from hidden_regime.quantconnect.signal_adapter import (
    TradingSignal,
    RegimeSignalAdapter,
    MultiAssetSignalAdapter
)


class TestTradingSignal:
    """Test TradingSignal dataclass."""

    def test_initialization(self):
        """Test signal creation."""
        signal = TradingSignal(
            direction='long',
            strength=0.8,
            allocation=1.0,
            confidence=0.9
        )

        assert signal.direction == 'long'
        assert signal.strength == 0.8
        assert signal.allocation == 1.0
        assert signal.confidence == 0.9

    def test_default_values(self):
        """Test default values."""
        signal = TradingSignal(direction='long')

        assert signal.strength == 1.0
        assert signal.allocation == 1.0
        assert signal.confidence == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = TradingSignal(
            direction='short',
            strength=0.5,
            allocation=0.3,
            confidence=0.7
        )

        d = asdict(signal)

        assert d['direction'] == 'short'
        assert d['strength'] == 0.5
        assert d['allocation'] == 0.3
        assert d['confidence'] == 0.7

    def test_valid_directions(self):
        """Test various valid directions."""
        for direction in ['long', 'short', 'neutral', 'cash']:
            signal = TradingSignal(direction=direction)
            assert signal.direction == direction


class TestRegimeSignalAdapter:
    """Test RegimeSignalAdapter class."""

    def test_initialization(self, sample_regime_allocations):
        """Test adapter initialization."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        assert adapter.regime_allocations == sample_regime_allocations

    def test_regime_to_signal_bull(self, sample_regime_allocations):
        """Test Bull regime conversion."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        signal = adapter.regime_to_signal('Bull', confidence=0.8)

        assert isinstance(signal, TradingSignal)
        assert signal.direction == 'long'
        assert signal.allocation == 1.0
        assert signal.confidence == 0.8

    def test_regime_to_signal_bear(self, sample_regime_allocations):
        """Test Bear regime conversion."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        signal = adapter.regime_to_signal('Bear', confidence=0.9)

        assert signal.direction == 'cash'
        assert signal.allocation == 0.0
        assert signal.confidence == 0.9

    def test_regime_to_signal_sideways(self, sample_regime_allocations):
        """Test Sideways regime conversion."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        signal = adapter.regime_to_signal('Sideways', confidence=0.7)

        assert signal.direction == 'long'
        assert signal.allocation == 0.5
        assert signal.confidence == 0.7

    def test_regime_to_signal_crisis(self, sample_regime_allocations):
        """Test Crisis regime conversion (negative allocation)."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        signal = adapter.regime_to_signal('Crisis', confidence=0.85)

        assert signal.direction == 'short'
        assert signal.allocation == 0.3  # abs(-0.3)
        assert signal.confidence == 0.85

    def test_unknown_regime(self):
        """Test handling of unknown regime."""
        adapter = RegimeSignalAdapter(regime_allocations={'Bull': 1.0, 'Bear': 0.0})

        signal = adapter.regime_to_signal('Unknown', confidence=0.5)

        # Should return neutral/default signal
        assert signal.direction in ['neutral', 'cash']
        assert signal.allocation >= 0

    def test_confidence_threshold(self, sample_regime_allocations):
        """Test confidence-based signal strength adjustment."""
        adapter = RegimeSignalAdapter(
            regime_allocations=sample_regime_allocations,
            confidence_threshold=0.7
        )

        # High confidence - full signal
        signal_high = adapter.regime_to_signal('Bull', confidence=0.9)
        assert signal_high.allocation == 1.0

        # Low confidence - reduced signal
        signal_low = adapter.regime_to_signal('Bull', confidence=0.5)
        assert signal_low.allocation < 1.0

    def test_custom_allocations(self):
        """Test with custom allocation mapping."""
        custom_allocations = {
            'Aggressive': 1.5,  # Leveraged
            'Conservative': 0.3,
            'Defensive': -0.5
        }

        adapter = RegimeSignalAdapter(regime_allocations=custom_allocations)

        signal = adapter.regime_to_signal('Aggressive', confidence=0.8)
        assert signal.allocation == 1.5  # Should allow > 1.0


class TestMultiAssetSignalAdapter:
    """Test MultiAssetSignalAdapter class."""

    def test_initialization(self):
        """Test adapter initialization."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'QQQ', 'TLT', 'GLD']
        )

        assert len(adapter.assets) == 4
        assert 'SPY' in adapter.assets

    def test_calculate_allocations_simple(self):
        """Test simple allocation calculation."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'QQQ', 'TLT'],
            allocation_method='equal_weight'
        )

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.8),
            'QQQ': TradingSignal('long', allocation=1.0, confidence=0.7),
            'TLT': TradingSignal('cash', allocation=0.0, confidence=0.9)
        }

        allocations = adapter.calculate_allocations(regime_signals)

        assert isinstance(allocations, dict)
        assert len(allocations) == 3
        assert abs(sum(allocations.values()) - 1.0) < 0.01  # Should sum to 1.0

    def test_confidence_weighted_allocations(self):
        """Test confidence-weighted allocation."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'QQQ'],
            allocation_method='confidence_weighted'
        )

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.9),
            'QQQ': TradingSignal('long', allocation=1.0, confidence=0.3)
        }

        allocations = adapter.calculate_allocations(regime_signals)

        # SPY should have higher allocation due to higher confidence
        assert allocations['SPY'] > allocations['QQQ']

    def test_regime_score_allocations(self):
        """Test regime score-based allocation."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'QQQ', 'TLT'],
            allocation_method='regime_score'
        )

        regime_signals = {
            'SPY': TradingSignal('long', strength=1.0, allocation=1.0, confidence=0.9),
            'QQQ': TradingSignal('long', strength=0.5, allocation=0.5, confidence=0.7),
            'TLT': TradingSignal('cash', strength=0.0, allocation=0.0, confidence=0.6)
        }

        allocations = adapter.calculate_allocations(regime_signals)

        # SPY should have highest allocation (highest strength * confidence)
        assert allocations['SPY'] >= allocations['QQQ']
        assert allocations['TLT'] == 0.0

    def test_risk_parity_allocations(self):
        """Test risk parity allocation method."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'TLT'],
            allocation_method='risk_parity'
        )

        # Provide volatility data
        volatilities = {
            'SPY': 0.20,  # Higher vol
            'TLT': 0.10   # Lower vol
        }

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.8),
            'TLT': TradingSignal('long', allocation=1.0, confidence=0.8)
        }

        allocations = adapter.calculate_allocations(
            regime_signals,
            volatilities=volatilities
        )

        # TLT should have higher allocation due to lower volatility
        assert allocations['TLT'] > allocations['SPY']

    def test_rebalancing_logic(self):
        """Test rebalancing threshold logic."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'QQQ'],
            rebalance_threshold=0.1  # 10% threshold
        )

        current_allocations = {'SPY': 0.6, 'QQQ': 0.4}

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.8),
            'QQQ': TradingSignal('long', allocation=1.0, confidence=0.8)
        }

        # Small change - should not trigger rebalance
        new_allocations_small = {'SPY': 0.62, 'QQQ': 0.38}
        should_rebalance = adapter.should_rebalance(
            current_allocations,
            new_allocations_small
        )
        assert not should_rebalance

        # Large change - should trigger rebalance
        new_allocations_large = {'SPY': 0.8, 'QQQ': 0.2}
        should_rebalance = adapter.should_rebalance(
            current_allocations,
            new_allocations_large
        )
        assert should_rebalance

    def test_defensive_allocation(self):
        """Test defensive allocation in crisis."""
        adapter = MultiAssetSignalAdapter(
            assets=['SPY', 'TLT', 'GLD', 'SHY'],
            defensive_assets=['TLT', 'GLD', 'SHY']
        )

        # Crisis regime signals
        regime_signals = {
            'SPY': TradingSignal('short', allocation=0.0, confidence=0.9),
            'TLT': TradingSignal('long', allocation=1.0, confidence=0.8),
            'GLD': TradingSignal('long', allocation=1.0, confidence=0.8),
            'SHY': TradingSignal('long', allocation=1.0, confidence=0.7)
        }

        allocations = adapter.calculate_allocations(regime_signals)

        # Should allocate to defensive assets only
        assert allocations['SPY'] == 0.0
        assert allocations['TLT'] > 0
        assert allocations['GLD'] > 0

    def test_empty_signals(self):
        """Test handling of empty signal dict."""
        adapter = MultiAssetSignalAdapter(assets=['SPY', 'QQQ'])

        allocations = adapter.calculate_allocations({})

        # Should return default/equal allocation
        assert isinstance(allocations, dict)

    def test_partial_signals(self):
        """Test with signals for subset of assets."""
        adapter = MultiAssetSignalAdapter(assets=['SPY', 'QQQ', 'TLT'])

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.8),
            # Missing QQQ and TLT
        }

        allocations = adapter.calculate_allocations(regime_signals)

        # Should handle missing signals gracefully
        assert isinstance(allocations, dict)
        assert 'SPY' in allocations


class TestSignalAdapterIntegration:
    """Integration tests for signal adapters."""

    def test_regime_to_trading_pipeline(self, sample_regime_allocations):
        """Test complete regime to trading signal pipeline."""
        # Step 1: Create adapter
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        # Step 2: Convert multiple regimes
        regimes = [
            ('Bull', 0.9),
            ('Bear', 0.8),
            ('Sideways', 0.7),
            ('Crisis', 0.85)
        ]

        signals = []
        for regime, confidence in regimes:
            signal = adapter.regime_to_signal(regime, confidence)
            signals.append(signal)

        # Step 3: Verify all signals created
        assert len(signals) == 4
        assert all(isinstance(s, TradingSignal) for s in signals)

    def test_multi_asset_regime_rotation(self):
        """Test multi-asset regime rotation scenario."""
        assets = ['SPY', 'QQQ', 'TLT', 'GLD']

        # Create regime allocations for each asset type
        equity_allocations = {'Bull': 1.0, 'Bear': 0.0, 'Sideways': 0.5}
        bond_allocations = {'Bull': 0.3, 'Bear': 0.7, 'Sideways': 0.5}
        gold_allocations = {'Bull': 0.2, 'Bear': 0.5, 'Crisis': 1.0}

        # Create signals for each asset based on its regime
        regime_signals = {
            'SPY': RegimeSignalAdapter(equity_allocations).regime_to_signal('Bull', 0.8),
            'QQQ': RegimeSignalAdapter(equity_allocations).regime_to_signal('Bull', 0.7),
            'TLT': RegimeSignalAdapter(bond_allocations).regime_to_signal('Bear', 0.9),
            'GLD': RegimeSignalAdapter(gold_allocations).regime_to_signal('Crisis', 0.6)
        }

        # Calculate portfolio allocation
        multi_adapter = MultiAssetSignalAdapter(
            assets=assets,
            allocation_method='confidence_weighted'
        )

        allocations = multi_adapter.calculate_allocations(regime_signals)

        # Verify allocations
        assert isinstance(allocations, dict)
        assert len(allocations) == 4
        assert all(0 <= v <= 1.5 for v in allocations.values())  # Reasonable range

    def test_dynamic_position_sizing(self):
        """Test dynamic position sizing based on regime confidence."""
        adapter = RegimeSignalAdapter(
            regime_allocations={'Bull': 1.0, 'Bear': 0.0},
            use_dynamic_sizing=True
        )

        # Test different confidence levels
        confidences = [0.5, 0.7, 0.9]
        signals = [adapter.regime_to_signal('Bull', conf) for conf in confidences]

        # Higher confidence should lead to larger allocations
        allocations = [s.allocation for s in signals]
        assert allocations[2] >= allocations[1] >= allocations[0]


class TestSignalAdapterEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_allocation(self):
        """Test zero allocation signal."""
        adapter = RegimeSignalAdapter(regime_allocations={'Regime': 0.0})

        signal = adapter.regime_to_signal('Regime', confidence=1.0)

        assert signal.allocation == 0.0
        assert signal.direction in ['cash', 'neutral']

    def test_negative_confidence(self, sample_regime_allocations):
        """Test handling of invalid negative confidence."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        # Should handle gracefully (clip to 0 or raise error)
        try:
            signal = adapter.regime_to_signal('Bull', confidence=-0.5)
            assert signal.confidence >= 0.0
        except ValueError:
            # Expected - invalid confidence
            assert True

    def test_confidence_above_one(self, sample_regime_allocations):
        """Test handling of confidence > 1.0."""
        adapter = RegimeSignalAdapter(regime_allocations=sample_regime_allocations)

        # Should handle gracefully (clip to 1.0 or raise error)
        try:
            signal = adapter.regime_to_signal('Bull', confidence=1.5)
            assert signal.confidence <= 1.0
        except ValueError:
            # Expected - invalid confidence
            assert True

    def test_extreme_allocations(self):
        """Test extreme allocation values."""
        adapter = RegimeSignalAdapter(
            regime_allocations={'Extreme': 5.0}  # 5x leverage
        )

        signal = adapter.regime_to_signal('Extreme', confidence=1.0)

        assert signal.allocation == 5.0  # Should allow extreme values

    def test_single_asset_multi_adapter(self):
        """Test multi-asset adapter with single asset."""
        adapter = MultiAssetSignalAdapter(assets=['SPY'])

        regime_signals = {
            'SPY': TradingSignal('long', allocation=1.0, confidence=0.8)
        }

        allocations = adapter.calculate_allocations(regime_signals)

        assert allocations['SPY'] == 1.0  # Single asset gets full allocation
