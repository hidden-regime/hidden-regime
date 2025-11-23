"""Component output schemas for validation.

Defines the expected output structure for each pipeline component.
Ensures clean data flow and catches integration bugs early.
"""

from typing import Dict, List, Set

import pandas as pd


class OutputSchema:
    """Validates component outputs against expected schema."""

    def __init__(
        self,
        name: str,
        required_columns: List[str],
        optional_columns: List[str] = None,
        description: str = "",
    ):
        """Initialize schema definition.

        Args:
            name: Schema name (e.g., "Model Output")
            required_columns: List of required column names
            optional_columns: List of optional column names
            description: Human-readable description
        """
        self.name = name
        self.required_columns = set(required_columns)
        self.optional_columns = set(optional_columns or [])
        self.description = description

    def validate(self, df: pd.DataFrame, strict: bool = False) -> Dict[str, any]:
        """Validate a DataFrame against this schema.

        Args:
            df: DataFrame to validate
            strict: If True, fail on extra columns. If False, warn only.

        Returns:
            Dictionary with validation result and any errors/warnings

        Raises:
            ValueError: If required columns are missing
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check for missing required columns
        df_columns = set(df.columns)
        missing = self.required_columns - df_columns
        if missing:
            result["valid"] = False
            result["errors"].append(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )

        # Check for extra columns
        allowed = self.required_columns | self.optional_columns
        extra = df_columns - allowed
        if extra:
            if strict:
                result["valid"] = False
                result["errors"].append(
                    f"Unexpected columns in {self.name}: {', '.join(sorted(extra))}"
                )
            else:
                result["warnings"].append(
                    f"Extra columns in {self.name}: {', '.join(sorted(extra))}"
                )

        # Check for empty DataFrame
        if len(df) == 0:
            result["warnings"].append("DataFrame is empty")

        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Schema({self.name}): "
            f"{len(self.required_columns)} required, "
            f"{len(self.optional_columns)} optional"
        )


# Define schemas for each component output

MODEL_OUTPUT_SCHEMA = OutputSchema(
    name="Model Output",
    required_columns=[
        "state",  # int: 0, 1, 2, ... (no semantic meaning)
        "confidence",  # float: max probability for state
        "state_probabilities",  # array: all state probabilities
    ],
    optional_columns=[
        "timestamp",
        "emission_means",  # array: HMM emission parameter means
        "emission_stds",  # array: HMM emission parameter stds
        "transition_probabilities",  # array: transition matrix
    ],
    description="Pure HMM output with state indices and probabilities. NO domain knowledge.",
)

INTERPRETER_OUTPUT_SCHEMA = OutputSchema(
    name="Interpreter Output",
    required_columns=[
        "state",  # From model
        "confidence",  # From model
        "state_probabilities",  # From model
        "regime_label",  # str: "Bear", "Bull", "Sideways" (domain interpretation)
        "regime_type",  # str: "bearish", "bullish", "sideways", "crisis"
        "regime_color",  # str: hex color for visualization
        "regime_strength",  # float: 0-1 confidence in regime
    ],
    optional_columns=[
        "timestamp",
        "regime_return",  # float: expected annual return for regime
        "regime_volatility",  # float: expected annual volatility
        "emission_means",
        "emission_stds",
        "transition_probabilities",
    ],
    description="Model output + domain knowledge. All regime interpretation here.",
)

SIGNAL_OUTPUT_SCHEMA = OutputSchema(
    name="Signal Generator Output",
    required_columns=[
        # All interpreter columns
        "state",
        "confidence",
        "state_probabilities",
        "regime_label",
        "regime_type",
        "regime_color",
        "regime_strength",
        # Signal generation columns
        "base_signal",  # float: -1.0 to 1.0 (short to long)
        "signal_strength",  # float: 0-1 (confidence in signal)
        "position_size",  # float: sized position (-max to +max)
        "signal_valid",  # bool: whether to trade
    ],
    optional_columns=[
        "timestamp",
        "regime_return",
        "regime_volatility",
        "regime_changed",  # bool: regime transition detected
        "strategy_name",  # str: which strategy generated this
        "entry_confidence",  # float: trade conviction
        "emission_means",
        "emission_stds",
        "transition_probabilities",
    ],
    description="Interpreter output + trading signals. All position sizing here.",
)


def validate_component_output(
    component_name: str,
    output: pd.DataFrame,
    strict: bool = False,
) -> Dict[str, any]:
    """Validate output from a specific component.

    Args:
        component_name: "model", "interpreter", "signals"
        output: Component output DataFrame
        strict: If True, fail on extra columns

    Returns:
        Validation result dictionary

    Raises:
        ValueError: If component_name not recognized
    """
    schemas = {
        "model": MODEL_OUTPUT_SCHEMA,
        "interpreter": INTERPRETER_OUTPUT_SCHEMA,
        "signals": SIGNAL_OUTPUT_SCHEMA,
    }

    if component_name not in schemas:
        raise ValueError(
            f"Unknown component: {component_name}. "
            f"Valid options: {list(schemas.keys())}"
        )

    schema = schemas[component_name]
    return schema.validate(output, strict=strict)


def assert_valid_output(
    component_name: str,
    output: pd.DataFrame,
    strict: bool = False,
) -> None:
    """Assert that component output is valid.

    Args:
        component_name: "model", "interpreter", "signals"
        output: Component output DataFrame
        strict: If True, fail on extra columns

    Raises:
        ValueError: If output is invalid
    """
    result = validate_component_output(component_name, output, strict=strict)

    if not result["valid"]:
        error_msg = f"\n{component_name.upper()} OUTPUT VALIDATION FAILED:\n"
        error_msg += "\n".join(f"  - {error}" for error in result["errors"])
        raise ValueError(error_msg)

    if result["warnings"]:
        import warnings

        warnings.warn(
            f"{component_name.upper()} output has warnings:\n"
            + "\n".join(f"  - {w}" for w in result["warnings"])
        )


__all__ = [
    "OutputSchema",
    "MODEL_OUTPUT_SCHEMA",
    "INTERPRETER_OUTPUT_SCHEMA",
    "SIGNAL_OUTPUT_SCHEMA",
    "validate_component_output",
    "assert_valid_output",
]
