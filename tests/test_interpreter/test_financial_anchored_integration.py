"""Integration tests for FinancialInterpreter with AnchoredInterpreter.

Tests verify:
1. AnchoredInterpreter instantiation with correct configuration
2. Anchored labels applied to regime interpretation
3. Stable label behavior during parameter drift
4. Configuration toggle (enabled/disabled)
5. Anchor status tracking and reset
6. Backward compatibility (disabled mode behaves like before)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.interpreter.anchored import AnchoredInterpreter


class TestAnchoredInterpreterIntegration:
    """Test FinancialInterpreter integration with AnchoredInterpreter."""

    def test_anchored_interpreter_instantiation_when_enabled(self):
        """AnchoredInterpreter should be created when use_anchored_interpretation=True."""
        config = InterpreterConfiguration(
            n_states=3,
            use_anchored_interpretation=True,
            anchor_update_rate=0.01
        )
        interpreter = FinancialInterpreter(config)

        assert interpreter._anchored_interpreter is not None
        assert isinstance(interpreter._anchored_interpreter, AnchoredInterpreter)
        assert interpreter._anchored_interpreter.anchor_update_rate == 0.01
        # AnchoredInterpreter has default regime anchors
        assert 'BULLISH' in interpreter._anchored_interpreter.regime_anchors
        assert 'BEARISH' in interpreter._anchored_interpreter.regime_anchors

    def test_no_anchored_interpreter_when_disabled(self):
        """AnchoredInterpreter should not be created when use_anchored_interpretation=False."""
        config = InterpreterConfiguration(
            n_states=3,
            use_anchored_interpretation=False
        )
        interpreter = FinancialInterpreter(config)

        assert interpreter._anchored_interpreter is None

    def test_anchored_labels_applied_to_interpretation(self):
        """Anchored interpretation should modify regime labels."""
        config = InterpreterConfiguration(
            n_states=2,
            interpretation_method="data_driven",
            use_anchored_interpretation=True,
            anchor_update_rate=0.01
        )
        interpreter = FinancialInterpreter(config)

        # Create sample model output
        model_output = pd.DataFrame({
            'state': [0, 1, 0, 1],
            'confidence': [0.9, 0.85, 0.92, 0.88],
            'emission_means': [
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008])
            ],
            'emission_stds': [
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012])
            ]
        })

        raw_data = pd.DataFrame({
            'log_return': [0.001, -0.0008, 0.0015, -0.0012]
        })

        # Update interpreter
        result = interpreter.update(model_output, raw_data=raw_data)

        # Should have regime labels
        assert 'regime_label' in result.columns
        assert result['regime_label'].notna().all()

        # Check that anchored interpreter was updated (history should have entries)
        assert interpreter._anchored_interpreter is not None
        status = interpreter.get_anchored_interpretation_status()
        total_updates = sum(status['anchor_update_history_counts'].values())
        assert total_updates > 0

    def test_stable_labels_with_parameter_drift(self):
        """Anchored interpretation should provide stable labels despite parameter drift."""
        config = InterpreterConfiguration(
            n_states=2,
            interpretation_method="data_driven",
            use_anchored_interpretation=True,
            anchor_update_rate=0.01  # Slow adaptation
        )
        interpreter = FinancialInterpreter(config)

        # First interpretation: establish anchors
        model_output_1 = pd.DataFrame({
            'state': [0, 1],
            'confidence': [0.9, 0.85],
            'emission_means': [
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008])
            ],
            'emission_stds': [
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012])
            ]
        })

        raw_data_1 = pd.DataFrame({
            'log_return': [0.001, -0.0008]
        })

        result_1 = interpreter.update(model_output_1, raw_data=raw_data_1)
        label_1 = result_1['regime_label'].iloc[0]

        # Second interpretation: parameters drift
        model_output_2 = pd.DataFrame({
            'state': [0, 1],
            'confidence': [0.9, 0.85],
            'emission_means': [
                np.array([0.0012, -0.0007]),  # Slightly different means
                np.array([0.0012, -0.0007])
            ],
            'emission_stds': [
                np.array([0.009, 0.011]),  # Slightly different stds
                np.array([0.009, 0.011])
            ]
        })

        raw_data_2 = pd.DataFrame({
            'log_return': [0.0012, -0.0007]
        })

        result_2 = interpreter.update(model_output_2, raw_data=raw_data_2)
        label_2 = result_2['regime_label'].iloc[0]

        # Labels should remain stable (same as before)
        assert label_1 == label_2, "Anchored interpretation should provide stable labels"

    def test_backward_compatibility_disabled_mode(self):
        """Disabled anchored interpretation should behave identically to before."""
        # Create two interpreters: one with anchoring disabled, one with default config
        config_disabled = InterpreterConfiguration(
            n_states=3,
            interpretation_method="data_driven",
            use_anchored_interpretation=False
        )
        interpreter_disabled = FinancialInterpreter(config_disabled)

        # Create sample model output
        model_output = pd.DataFrame({
            'state': [0, 1, 2],
            'confidence': [0.9, 0.85, 0.92],
            'emission_means': [
                np.array([0.002, 0.0, -0.002]),
                np.array([0.002, 0.0, -0.002]),
                np.array([0.002, 0.0, -0.002])
            ],
            'emission_stds': [
                np.array([0.008, 0.010, 0.015]),
                np.array([0.008, 0.010, 0.015]),
                np.array([0.008, 0.010, 0.015])
            ]
        })

        raw_data = pd.DataFrame({
            'log_return': [0.002, 0.0, -0.002]
        })

        result = interpreter_disabled.update(model_output, raw_data=raw_data)

        # Should have regime labels
        assert 'regime_label' in result.columns
        assert result['regime_label'].notna().all()

        # No anchored interpreter should exist
        assert interpreter_disabled._anchored_interpreter is None

    def test_configuration_validation_anchor_update_rate(self):
        """anchor_update_rate should be validated as float in [0.0, 1.0]."""
        # Valid rate
        config = InterpreterConfiguration(
            n_states=3,
            anchor_update_rate=0.5
        )
        assert config.anchor_update_rate == 0.5

        # Invalid rate (out of range)
        with pytest.raises(ValueError, match="anchor_update_rate must be in"):
            InterpreterConfiguration(
                n_states=3,
                anchor_update_rate=1.5
            )

        with pytest.raises(ValueError, match="anchor_update_rate must be in"):
            InterpreterConfiguration(
                n_states=3,
                anchor_update_rate=-0.1
            )

    def test_configuration_validation_use_anchored_interpretation(self):
        """use_anchored_interpretation should be validated as bool."""
        # Valid bool
        config = InterpreterConfiguration(
            n_states=3,
            use_anchored_interpretation=True
        )
        assert config.use_anchored_interpretation is True

        # Invalid type
        with pytest.raises(ValueError, match="use_anchored_interpretation must be bool"):
            InterpreterConfiguration(
                n_states=3,
                use_anchored_interpretation="yes"
            )

    def test_get_anchored_interpretation_status_enabled(self):
        """get_anchored_interpretation_status should return anchor information when enabled."""
        config = InterpreterConfiguration(
            n_states=2,
            use_anchored_interpretation=True,
            anchor_update_rate=0.02
        )
        interpreter = FinancialInterpreter(config)

        # Update with some data to establish anchors
        model_output = pd.DataFrame({
            'state': [0, 1],
            'confidence': [0.9, 0.85],
            'emission_means': [
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008])
            ],
            'emission_stds': [
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012])
            ]
        })

        raw_data = pd.DataFrame({
            'log_return': [0.001, -0.0008]
        })

        interpreter.update(model_output, raw_data=raw_data)

        # Get status
        status = interpreter.get_anchored_interpretation_status()

        assert status['enabled'] is True
        assert status['anchor_update_rate'] == 0.02
        assert 'regime_anchors' in status
        assert len(status['regime_anchors']) > 0

        # Check anchor structure
        for regime_name, anchor_info in status['regime_anchors'].items():
            assert 'mean' in anchor_info
            assert 'std' in anchor_info
            assert isinstance(anchor_info['mean'], float)
            assert isinstance(anchor_info['std'], float)

        # Check update history counts
        assert 'anchor_update_history_counts' in status
        for regime_name, count in status['anchor_update_history_counts'].items():
            assert isinstance(count, int)
            assert count >= 0

    def test_get_anchored_interpretation_status_disabled(self):
        """get_anchored_interpretation_status should indicate disabled when off."""
        config = InterpreterConfiguration(
            n_states=2,
            use_anchored_interpretation=False
        )
        interpreter = FinancialInterpreter(config)

        status = interpreter.get_anchored_interpretation_status()

        assert status['enabled'] is False
        assert 'status' in status
        assert 'Anchored interpretation disabled' in status['status']

    def test_reset_anchored_interpretation(self):
        """reset_anchored_interpretation should clear all anchor update history."""
        config = InterpreterConfiguration(
            n_states=2,
            use_anchored_interpretation=True
        )
        interpreter = FinancialInterpreter(config)

        # Establish anchors with update
        model_output = pd.DataFrame({
            'state': [0, 1],
            'confidence': [0.9, 0.85],
            'emission_means': [
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008])
            ],
            'emission_stds': [
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012])
            ]
        })

        raw_data = pd.DataFrame({
            'log_return': [0.001, -0.0008]
        })

        interpreter.update(model_output, raw_data=raw_data)

        # Verify history exists
        status_before = interpreter.get_anchored_interpretation_status()
        total_updates_before = sum(status_before['anchor_update_history_counts'].values())
        assert total_updates_before > 0

        # Reset
        interpreter.reset_anchored_interpretation()

        # Verify history cleared
        status_after = interpreter.get_anchored_interpretation_status()
        total_updates_after = sum(status_after['anchor_update_history_counts'].values())
        assert total_updates_after == 0

    def test_configuration_serialization_with_anchored_params(self):
        """Configuration with anchored params should serialize/deserialize correctly."""
        config = InterpreterConfiguration(
            n_states=3,
            use_anchored_interpretation=True,
            anchor_update_rate=0.015
        )

        # Serialize
        config_dict = config.to_dict()

        # Verify serialization
        assert config_dict['use_anchored_interpretation'] is True
        assert config_dict['anchor_update_rate'] == 0.015

        # Deserialize
        config_restored = InterpreterConfiguration.from_dict(config_dict)

        # Verify restoration
        assert config_restored.use_anchored_interpretation is True
        assert config_restored.anchor_update_rate == 0.015

    def test_different_anchor_update_rates(self):
        """Different anchor update rates should affect adaptation speed."""
        # Slow adaptation (rate=0.001)
        config_slow = InterpreterConfiguration(
            n_states=2,
            use_anchored_interpretation=True,
            anchor_update_rate=0.001
        )
        interpreter_slow = FinancialInterpreter(config_slow)

        # Fast adaptation (rate=0.1)
        config_fast = InterpreterConfiguration(
            n_states=2,
            use_anchored_interpretation=True,
            anchor_update_rate=0.1
        )
        interpreter_fast = FinancialInterpreter(config_fast)

        # Both should have anchored interpreters
        assert interpreter_slow._anchored_interpreter is not None
        assert interpreter_fast._anchored_interpreter is not None

        # Rates should differ
        assert (interpreter_slow._anchored_interpreter.anchor_update_rate !=
                interpreter_fast._anchored_interpreter.anchor_update_rate)

    def test_anchored_interpretation_with_manual_labels_disabled(self):
        """Anchored interpretation should be skipped if manual labels are set."""
        config = InterpreterConfiguration(
            n_states=2,
            force_regime_labels=['Bull', 'Bear'],
            acknowledge_override=True,
            use_anchored_interpretation=True  # Should be ignored when manual labels set
        )
        interpreter = FinancialInterpreter(config)

        model_output = pd.DataFrame({
            'state': [0, 1],
            'confidence': [0.9, 0.85],
            'emission_means': [
                np.array([0.001, -0.0008]),
                np.array([0.001, -0.0008])
            ],
            'emission_stds': [
                np.array([0.008, 0.012]),
                np.array([0.008, 0.012])
            ]
        })

        raw_data = pd.DataFrame({
            'log_return': [0.001, -0.0008]
        })

        result = interpreter.update(model_output, raw_data=raw_data)

        # Labels should be manual ones, not anchored
        assert result['regime_label'].iloc[0] == 'Bull'
        assert result['regime_label'].iloc[1] == 'Bear'
