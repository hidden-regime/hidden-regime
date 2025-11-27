"""
Extended interpreter that produces RegimeLabel objects.

This module wraps the FinancialInterpreter and adds a new 'regime_label' column
containing RegimeLabel objects (not strings).

This adapter approach allows us to:
1. Preserve existing Interpreter functionality
2. Gradually migrate to the new architecture
3. Maintain backward compatibility while moving forward
"""

from typing import Optional

import numpy as np
import pandas as pd

from hidden_regime.config.interpreter import InterpreterConfiguration
from hidden_regime.interpreter.financial import FinancialInterpreter
from hidden_regime.interpreter.regime_label_builder import RegimeLabelBuilder
from hidden_regime.interpreter.regime_types import RegimeType


class RegimeLabelInterpreter(FinancialInterpreter):
    """
    Extended FinancialInterpreter that produces RegimeLabel objects.

    Extends FinancialInterpreter to add a 'regime_label' column containing
    RegimeLabel objects instead of strings.

    The RegimeLabel objects encapsulate:
    - Financial characteristics (returns, volatility, drawdown, etc.)
    - Trading semantics (bias, position sign, confidence thresholds)
    - Metadata for strategy extensions

    This enables seamless integration with the new Strategy architecture
    while maintaining all existing Interpreter functionality.
    """

    def __init__(self, config: InterpreterConfiguration):
        """Initialize regime label interpreter.

        Args:
            config: InterpreterConfiguration object
        """
        super().__init__(config)
        self._regime_label_cache = {}

    def update(
        self,
        model_output: pd.DataFrame,
        raw_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Interpret model output and add RegimeLabel objects.

        Args:
            model_output: Raw model predictions
            raw_data: Optional raw OHLCV data for performance calculations

        Returns:
            DataFrame with 'regime_label' column containing RegimeLabel objects
        """
        # Call parent to get standard interpretation
        interpreted = super().update(model_output, raw_data)

        # Add RegimeLabel column
        interpreted["regime_label"] = interpreted.apply(
            lambda row: self._create_regime_label_for_row(row),
            axis=1
        )

        return interpreted

    def _create_regime_label_for_row(self, row: pd.Series) -> Optional["RegimeLabel"]:
        """
        Create a RegimeLabel object for a single row.

        Extracts the metrics from the row and builds a RegimeLabel.

        Args:
            row: Single row from interpreted output

        Returns:
            RegimeLabel object or None if metrics unavailable
        """
        state = row.get("state")

        # Check cache first
        cache_key = state
        if cache_key in self._regime_label_cache:
            return self._regime_label_cache[cache_key]

        # Extract regime name and type from existing columns
        regime_name_str = row.get("regime_label")  # String from parent
        if regime_name_str is None:
            return None

        # Normalize to standard enum names
        regime_type = self._normalize_regime_type(regime_name_str)
        regime_name = regime_type.name  # "BULLISH", "BEARISH", etc.

        # Extract financial metrics
        metrics = self._extract_metrics_for_state(state, row)
        if metrics is None:
            return None

        # Build the RegimeLabel
        try:
            regime_label = RegimeLabelBuilder.build_from_metrics(
                regime_name=regime_name,
                regime_type=regime_type,
                regime_strength=row.get("regime_strength", 0.5),
                **metrics
            )

            # Cache for performance
            self._regime_label_cache[cache_key] = regime_label

            return regime_label
        except Exception as e:
            # Fail gracefully if we can't build the label
            import warnings
            warnings.warn(f"Failed to build RegimeLabel for state {state}: {e}")
            return None

    def _normalize_regime_type(self, label_str: str) -> RegimeType:
        """
        Convert string regime label to RegimeType enum.

        Maps complex labels (e.g., "Euphoric Bull") to standard RegimeType.

        Args:
            label_str: Regime label string

        Returns:
            RegimeType enum value
        """
        if label_str is None:
            return RegimeType.MIXED

        label_lower = label_str.lower()

        # Crisis takes priority
        if "crisis" in label_lower or "crash" in label_lower:
            return RegimeType.CRISIS

        # Then check for direction
        if "bull" in label_lower or "uptrend" in label_lower or "euphoric" in label_lower:
            return RegimeType.BULLISH
        elif "bear" in label_lower or "downtrend" in label_lower:
            return RegimeType.BEARISH
        elif "sideways" in label_lower or "flat" in label_lower or "range" in label_lower:
            return RegimeType.SIDEWAYS
        else:
            return RegimeType.MIXED

    def _extract_metrics_for_state(self, state: int, row: pd.Series) -> Optional[dict]:
        """
        Extract financial metrics for a state.

        Pulls metrics from the regime profiles cache if available,
        otherwise estimates from the current row.

        Args:
            state: HMM state index
            row: Current row from interpretation

        Returns:
            Dictionary of metrics for RegimeLabelBuilder, or None
        """
        # Try to get from cached profiles
        if self._regime_profiles and state in self._regime_profiles:
            profile = self._regime_profiles[state]
            return {
                "mean_daily_return": profile.get("mean_return", 0.0) / 252,
                "annualized_return": profile.get("mean_return", 0.0),
                "daily_volatility": profile.get("volatility", 0.1) / np.sqrt(252),
                "annualized_volatility": profile.get("volatility", 0.1),
                "win_rate": profile.get("win_rate", 0.5),
                "max_drawdown": profile.get("max_drawdown", -0.1),
                "return_skewness": profile.get("skewness", 0.0),
                "return_kurtosis": profile.get("kurtosis", 0.0),
                "sharpe_ratio": profile.get("sharpe", 0.0),
                "persistence_days": profile.get("persistence", 20.0),
                "transition_volatility": profile.get("transition_vol", 0.1),
                "state_id": state,
            }

        # Otherwise, use row values or defaults
        return {
            "mean_daily_return": row.get("regime_return", 0.0) / 252 if row.get("regime_return") else 0.0,
            "annualized_return": row.get("regime_return", 0.0) or 0.0,
            "daily_volatility": row.get("regime_volatility", 0.1) / np.sqrt(252) if row.get("regime_volatility") else 0.1,
            "annualized_volatility": row.get("regime_volatility", 0.1) or 0.1,
            "win_rate": 0.5,  # Default if not available
            "max_drawdown": -0.1,  # Default
            "return_skewness": 0.0,  # Default
            "return_kurtosis": 0.0,  # Default
            "sharpe_ratio": 0.0,  # Default
            "persistence_days": 20.0,  # Default
            "transition_volatility": 0.1,  # Default
            "state_id": state,
        }
