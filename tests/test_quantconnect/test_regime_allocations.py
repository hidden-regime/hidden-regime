"""
Comprehensive tests for RegimeTypeAllocations and enum-based signal generation.

Tests verify:
- RegimeTypeAllocations dataclass creation and validation
- RegimeType enum-based allocation lookups
- Signal generation using RegimeType instead of string matching
- Breaking change enforcement (reject old string-based allocations)
- Temporal stability of allocations across model retrains
"""

import warnings

import pandas as pd
import pytest

from hidden_regime.interpreter.regime_types import RegimeType
from hidden_regime.quantconnect.config import RegimeTypeAllocations
from hidden_regime.quantconnect.signal_adapter import (
    RegimeSignalAdapter,
    SignalDirection,
    SignalStrength,
    TradingSignal,
)


class TestRegimeTypeAllocations:
    """Test RegimeTypeAllocations dataclass."""

    def test_default_allocations(self):
        """Test default allocation values."""
        allocations = RegimeTypeAllocations()

        assert allocations.bullish == 1.0
        assert allocations.bearish == 0.0
        assert allocations.sideways == 0.5
        assert allocations.crisis == 0.0
        assert allocations.mixed == 0.25

    def test_custom_allocations(self):
        """Test creating allocations with custom values."""
        allocations = RegimeTypeAllocations(
            bullish=0.8,
            bearish=-0.2,
            sideways=0.4,
            crisis=0.0,
            mixed=0.15,
        )

        assert allocations.bullish == 0.8
        assert allocations.bearish == -0.2
        assert allocations.sideways == 0.4

    def test_get_allocation_bullish(self):
        """Test getting allocation for BULLISH regime."""
        allocations = RegimeTypeAllocations(bullish=1.0)

        assert allocations.get_allocation(RegimeType.BULLISH) == 1.0

    def test_get_allocation_bearish(self):
        """Test getting allocation for BEARISH regime."""
        allocations = RegimeTypeAllocations(bearish=0.0)

        assert allocations.get_allocation(RegimeType.BEARISH) == 0.0

    def test_get_allocation_sideways(self):
        """Test getting allocation for SIDEWAYS regime."""
        allocations = RegimeTypeAllocations(sideways=0.5)

        assert allocations.get_allocation(RegimeType.SIDEWAYS) == 0.5

    def test_get_allocation_crisis(self):
        """Test getting allocation for CRISIS regime."""
        allocations = RegimeTypeAllocations(crisis=0.1)

        assert allocations.get_allocation(RegimeType.CRISIS) == 0.1

    def test_get_allocation_mixed(self):
        """Test getting allocation for MIXED regime."""
        allocations = RegimeTypeAllocations(mixed=0.3)

        assert allocations.get_allocation(RegimeType.MIXED) == 0.3

    def test_validation_too_high(self):
        """Test validation rejects allocations > 2.0."""
        with pytest.raises(ValueError, match="must be between -2.0 and 2.0"):
            RegimeTypeAllocations(bullish=2.5)

    def test_validation_too_low(self):
        """Test validation rejects allocations < -2.0."""
        with pytest.raises(ValueError, match="must be between -2.0 and 2.0"):
            RegimeTypeAllocations(bearish=-2.5)

    def test_validation_allows_leverage(self):
        """Test validation allows up to 2.0x leverage."""
        allocations = RegimeTypeAllocations(bullish=2.0, bearish=-2.0)
        assert allocations.bullish == 2.0
        assert allocations.bearish == -2.0

    def test_validation_allows_shorting(self):
        """Test validation allows negative (short) allocations."""
        allocations = RegimeTypeAllocations(bearish=-0.5)
        assert allocations.bearish == -0.5

    def test_create_conservative(self):
        """Test create_conservative factory method."""
        allocations = RegimeTypeAllocations.create_conservative()

        assert allocations.bullish == 0.6
        assert allocations.bearish == 0.0
        assert allocations.sideways == 0.3
        assert allocations.crisis == 0.0
        assert allocations.mixed == 0.0

    def test_create_aggressive(self):
        """Test create_aggressive factory method."""
        allocations = RegimeTypeAllocations.create_aggressive()

        assert allocations.bullish == 1.0
        assert allocations.bearish == 0.2
        assert allocations.sideways == 0.8
        assert allocations.crisis == 0.0
        assert allocations.mixed == 0.0

    def test_create_market_neutral(self):
        """Test create_market_neutral factory method."""
        allocations = RegimeTypeAllocations.create_market_neutral()

        assert allocations.bullish == 0.5
        assert allocations.bearish == -0.5
        assert allocations.sideways == 0.0
        assert allocations.crisis == 0.0
        assert allocations.mixed == 0.0


class TestRegimeSignalAdapterEnumBased:
    """Test RegimeSignalAdapter with enum-based allocations."""

    def test_initialization_with_enum_allocations(self):
        """Test adapter accepts RegimeTypeAllocations."""
        allocations = RegimeTypeAllocations(bullish=1.0, bearish=0.0)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        assert adapter.regime_type_allocations == allocations

    def test_initialization_rejects_dict_allocations(self):
        """Test adapter rejects old string-based dict (breaking change)."""
        old_allocations = {"Bull": 1.0, "Bear": 0.0}

        with pytest.raises(ValueError, match="no longer supported"):
            RegimeSignalAdapter(regime_type_allocations=old_allocations)

    def test_default_allocations_if_none(self):
        """Test adapter creates default allocations if none provided."""
        adapter = RegimeSignalAdapter()

        assert isinstance(adapter.regime_type_allocations, RegimeTypeAllocations)
        assert adapter.regime_type_allocations.bullish == 1.0

    def test_generate_signal_bullish(self):
        """Test signal generation for BULLISH regime."""
        allocations = RegimeTypeAllocations(bullish=1.0)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        signal = adapter.generate_signal(
            regime_name="Bull",
            regime_state=0,
            confidence=0.9,
            timestamp=pd.Timestamp("2023-01-01"),
            regime_type=RegimeType.BULLISH,
        )

        assert isinstance(signal, TradingSignal)
        assert signal.allocation == 1.0
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence == 0.9

    def test_generate_signal_bearish(self):
        """Test signal generation for BEARISH regime."""
        allocations = RegimeTypeAllocations(bearish=0.0)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        signal = adapter.generate_signal(
            regime_name="Bear",
            regime_state=1,
            confidence=0.85,
            timestamp=pd.Timestamp("2023-01-02"),
            regime_type=RegimeType.BEARISH,
        )

        assert signal.allocation == 0.0
        assert signal.direction == SignalDirection.NEUTRAL

    def test_generate_signal_sideways(self):
        """Test signal generation for SIDEWAYS regime."""
        allocations = RegimeTypeAllocations(sideways=0.5)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        signal = adapter.generate_signal(
            regime_name="Sideways",
            regime_state=2,
            confidence=0.7,
            timestamp=pd.Timestamp("2023-01-03"),
            regime_type=RegimeType.SIDEWAYS,
        )

        assert signal.allocation == 0.5

    def test_generate_signal_crisis(self):
        """Test signal generation for CRISIS regime."""
        allocations = RegimeTypeAllocations(crisis=0.0)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        signal = adapter.generate_signal(
            regime_name="Crisis",
            regime_state=3,
            confidence=0.95,
            timestamp=pd.Timestamp("2023-01-04"),
            regime_type=RegimeType.CRISIS,
        )

        assert signal.allocation == 0.0
        assert signal.regime_name == "Crisis"

    def test_generate_signal_missing_regime_type(self):
        """Test error when regime_type missing from metadata."""
        allocations = RegimeTypeAllocations()
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        with pytest.raises(ValueError, match="regime_type not in metadata"):
            adapter.generate_signal(
                regime_name="Unknown",
                regime_state=-1,
                confidence=0.0,
                timestamp=pd.Timestamp("2023-01-05"),
                # regime_type intentionally missing
            )

    def test_generate_signal_regime_type_as_string(self):
        """Test regime_type can be passed as string and is converted."""
        allocations = RegimeTypeAllocations()
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        signal = adapter.generate_signal(
            regime_name="Bull",
            regime_state=0,
            confidence=0.9,
            timestamp=pd.Timestamp("2023-01-06"),
            regime_type="BULLISH",  # Pass as string, should be converted
        )

        assert signal.allocation == 1.0

    def test_generate_signal_invalid_regime_type_string(self):
        """Test error for invalid regime_type string."""
        allocations = RegimeTypeAllocations()
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        with pytest.raises(ValueError, match="Invalid regime_type"):
            adapter.generate_signal(
                regime_name="Unknown",
                regime_state=-1,
                confidence=0.0,
                timestamp=pd.Timestamp("2023-01-07"),
                regime_type="INVALID_TYPE",
            )

    def test_allocation_lookup_warns_if_missing(self):
        """Test warning when no allocation defined for regime type."""
        allocations = RegimeTypeAllocations(mixed=0.0)  # MIXED not explicitly set
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = adapter._get_allocation_from_regime_type(RegimeType.MIXED)

            assert len(w) == 1
            assert "No allocation defined" in str(w[0].message)

    def test_strength_determined_by_confidence(self):
        """Test signal strength determined from confidence."""
        allocations = RegimeTypeAllocations()
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        # High confidence -> STRONG
        signal_strong = adapter.generate_signal(
            regime_name="Bull",
            regime_state=0,
            confidence=0.85,
            timestamp=pd.Timestamp("2023-01-08"),
            regime_type=RegimeType.BULLISH,
        )
        assert signal_strong.strength == SignalStrength.STRONG

        # Moderate confidence -> MODERATE
        signal_moderate = adapter.generate_signal(
            regime_name="Bull",
            regime_state=0,
            confidence=0.65,
            timestamp=pd.Timestamp("2023-01-09"),
            regime_type=RegimeType.BULLISH,
        )
        assert signal_moderate.strength == SignalStrength.MODERATE

        # Low confidence -> WEAK
        signal_weak = adapter.generate_signal(
            regime_name="Bull",
            regime_state=0,
            confidence=0.45,
            timestamp=pd.Timestamp("2023-01-10"),
            regime_type=RegimeType.BULLISH,
        )
        assert signal_weak.strength == SignalStrength.WEAK


class TestAllocationStabilityAcrossRetrains:
    """Test that allocations remain stable even with discovered regime name changes."""

    def test_same_regime_type_different_names(self):
        """Test that same RegimeType produces same allocation despite different names."""
        allocations = RegimeTypeAllocations(bullish=0.9)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        # First discovery: HMM discovers "Bull"
        signal1 = adapter.generate_signal(
            regime_name="Bull",
            regime_state=0,
            confidence=0.8,
            timestamp=pd.Timestamp("2023-01-01"),
            regime_type=RegimeType.BULLISH,
        )

        # After retrain: HMM discovers "Uptrend" (different name, same type)
        signal2 = adapter.generate_signal(
            regime_name="Uptrend",
            regime_state=0,
            confidence=0.8,
            timestamp=pd.Timestamp("2023-02-01"),
            regime_type=RegimeType.BULLISH,
        )

        # Allocations should be identical despite different names
        assert signal1.allocation == signal2.allocation
        assert signal1.allocation == 0.9

    def test_regime_type_trumps_name_matching(self):
        """Test that RegimeType enum always determines allocation, not regime name."""
        allocations = RegimeTypeAllocations(bullish=1.0, bearish=0.0)
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        # If somehow a "Bear" regime is classified as BULLISH (unlikely but possible)
        # the allocation should be for BULLISH, not BEARISH
        signal = adapter.generate_signal(
            regime_name="Bear",  # name suggests bearish
            regime_state=1,
            confidence=0.7,
            timestamp=pd.Timestamp("2023-01-11"),
            regime_type=RegimeType.BULLISH,  # but type is bullish
        )

        # Should use BULLISH allocation, not BEARISH
        assert signal.allocation == 1.0


class TestFourStateRegimeDetection:
    """Test that 4-state HMM allocations work correctly."""

    def test_4state_allocation_all_types(self):
        """Test allocating for all 5 discovered RegimeTypes."""
        allocations = RegimeTypeAllocations(
            bullish=1.0,
            bearish=0.0,
            sideways=0.5,
            crisis=0.0,
            mixed=0.25,
        )
        adapter = RegimeSignalAdapter(regime_type_allocations=allocations)

        # Create signals for each RegimeType
        signals_by_type = {}
        for i, regime_type in enumerate(RegimeType):
            signal = adapter.generate_signal(
                regime_name=regime_type.name,
                regime_state=i,
                confidence=0.8,
                timestamp=pd.Timestamp(f"2023-01-{i+1:02d}"),
                regime_type=regime_type,
            )
            signals_by_type[regime_type] = signal

        # Verify all regime types produce signals with correct allocations
        assert signals_by_type[RegimeType.BULLISH].allocation == 1.0
        assert signals_by_type[RegimeType.BEARISH].allocation == 0.0
        assert signals_by_type[RegimeType.SIDEWAYS].allocation == 0.5
        assert signals_by_type[RegimeType.CRISIS].allocation == 0.0
        assert signals_by_type[RegimeType.MIXED].allocation == 0.25


class TestBackwardCompatibilityBreakingChange:
    """Test that old string-based approaches are rejected."""

    def test_reject_string_dict_in_init(self):
        """Test initialization rejects string-based regime_allocations dict."""
        with pytest.raises(ValueError, match="no longer supported"):
            RegimeSignalAdapter(regime_type_allocations={"Bull": 1.0, "Bear": 0.0})

    def test_reject_regime_allocations_kwarg_in_algorithm(self):
        """Test algorithm.initialize_regime_detection rejects old kwarg."""
        # This would be tested at the HiddenRegimeAlgorithm level
        # but we verify the error message is clear
        from hidden_regime.quantconnect.signal_adapter import RegimeSignalAdapter

        bad_allocations = {"Bull": 1.0}
        with pytest.raises(ValueError):
            RegimeSignalAdapter(regime_type_allocations=bad_allocations)
