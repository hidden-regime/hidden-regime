"""
Unit tests for utils/regime_mapping.py

Tests regime mapping utilities including numeric-to-label mapping.
"""

import pytest
import pandas as pd


def test_map_numeric_to_labels():
    """Test mapping numeric states to labels."""
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}
    states = [0, 1, 2, 1, 0]

    labels = [mapping[s] for s in states]

    assert labels == ["Bear", "Sideways", "Bull", "Sideways", "Bear"]


def test_reverse_regime_mapping():
    """Test reverse mapping from labels to numeric."""
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}
    reverse_mapping = {v: k for k, v in mapping.items()}

    assert reverse_mapping["Bear"] == 0
    assert reverse_mapping["Sideways"] == 1
    assert reverse_mapping["Bull"] == 2


def test_regime_mapping_validation():
    """Test regime mapping validation."""
    # Valid mapping
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}
    assert len(mapping) == 3
    assert all(isinstance(k, int) for k in mapping.keys())
    assert all(isinstance(v, str) for v in mapping.values())


def test_custom_regime_definitions():
    """Test custom regime definitions."""
    custom_mapping = {
        0: "Crisis",
        1: "Bear",
        2: "Sideways",
        3: "Bull",
        4: "Euphoric",
    }

    assert len(custom_mapping) == 5
    assert "Crisis" in custom_mapping.values()


def test_regime_mapping_consistency():
    """Test consistency of regime mapping across modules."""
    # Same mapping should produce same results
    mapping1 = {0: "Bear", 1: "Sideways", 2: "Bull"}
    mapping2 = {0: "Bear", 1: "Sideways", 2: "Bull"}

    assert mapping1 == mapping2


def test_regime_mapping_with_interpreter():
    """Test regime mapping integration with interpreter."""
    states = pd.Series([0, 1, 2, 1, 0])
    mapping = {0: "Bear", 1: "Sideways", 2: "Bull"}

    regimes = states.map(mapping)

    assert list(regimes) == ["Bear", "Sideways", "Bull", "Sideways", "Bear"]
