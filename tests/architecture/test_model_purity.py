"""
Test Model Purity - Ensure models/ contains no financial domain knowledge.

This test enforces the architectural principle that Model components should be
pure mathematics with no financial terminology. All financial domain knowledge
belongs in the Interpreter component.

This is a CRITICAL architectural constraint for:
- Model reusability across domains
- Clear separation of concerns
- Maintainability and testability
"""

import os
import re
from pathlib import Path

import pytest


# Financial terms that should NOT appear in model layer code
# NOTE: Only include terms that are specifically financial, not generic math/CS terms
# "alpha"/"beta" are excluded - they're standard mathematical notation (HMM forward-backward, Greek letters)
# "signal" is excluded - it's a generic term in signal processing and time series
# "trade" is excluded - it has non-financial meanings (e.g., "trade-off")
FORBIDDEN_FINANCIAL_TERMS = [
    "bull", "bullish",
    "bear", "bearish",
    "crisis",
    "sideways",
    "regime_type",  # This is an interpreter concept
    "market",
    "stock",
    "price",
    "portfolio",
    "trading",  # "trade" alone is too generic (trade-off), but "trading" is financial
    "buy", "sell",
    "long", "short",
    "profit", "loss",
    "sharpe", "sortino",
    "drawdown",
]

# Patterns to exclude from checks (comments, docstrings, test data)
EXCLUDED_PATTERNS = [
    r'^\s*#',  # Comment lines
    r'^\s*"""',  # Docstring start
    r"^\s*'''",  # Docstring start (single quotes)
    r'raise.*Error.*\(',  # Error messages
    r'warnings\.warn',  # Warning messages
    r'print\(',  # Print statements (usually for debugging/logging)
    r'logger\.',  # Log statements
]


def should_check_line(line: str) -> bool:
    """Determine if a line should be checked for forbidden terms."""
    for pattern in EXCLUDED_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return False
    return True


def get_model_files():
    """Get all Python files in the models/ directory."""
    models_dir = Path(__file__).parent.parent.parent / "hidden_regime" / "models"

    if not models_dir.exists():
        pytest.skip(f"Models directory not found: {models_dir}")

    python_files = list(models_dir.glob("*.py"))

    # Exclude __init__.py as it may contain imports/documentation
    python_files = [f for f in python_files if f.name != "__init__.py"]

    if not python_files:
        pytest.skip("No Python files found in models/ directory")

    return python_files


@pytest.mark.architecture
class TestModelPurity:
    """Test that models/ contains no financial domain knowledge."""

    def test_no_financial_terms_in_model_code(self):
        """
        Verify models/ contains no financial terminology in actual code.

        This test scans all Python files in hidden_regime/models/ for forbidden
        financial terms. Comments and docstrings are excluded from checks since
        they may explain financial concepts without implementing them.
        """
        violations = []

        model_files = get_model_files()

        for file_path in model_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                # Skip comments and docstrings
                if not should_check_line(line):
                    continue

                # Check for forbidden terms
                line_lower = line.lower()
                for term in FORBIDDEN_FINANCIAL_TERMS:
                    # Use word boundaries to avoid false positives
                    if re.search(rf'\b{term}\b', line_lower):
                        violations.append({
                            'file': file_path.name,
                            'line': line_num,
                            'term': term,
                            'code': line.strip()
                        })

        # Generate helpful error message
        if violations:
            error_msg = "\n\nMODEL PURITY VIOLATION: Financial terms found in models/\n"
            error_msg += "=" * 80 + "\n\n"
            error_msg += "The Model component should contain ONLY mathematical logic.\n"
            error_msg += "All financial domain knowledge belongs in the Interpreter component.\n\n"
            error_msg += "Violations found:\n\n"

            for v in violations:
                error_msg += f"  {v['file']}:{v['line']} - term '{v['term']}'\n"
                error_msg += f"    Code: {v['code']}\n\n"

            error_msg += "To fix:\n"
            error_msg += "  1. Move financial logic to hidden_regime/interpreter/\n"
            error_msg += "  2. Model should output states (0, 1, 2) and parameters\n"
            error_msg += "  3. Interpreter maps states to financial regimes\n"

            pytest.fail(error_msg)

    def test_model_outputs_are_domain_agnostic(self):
        """
        Verify HMM model outputs contain no financial terminology.

        This test imports the main HMM class and checks that its public API
        (return types, variable names) doesn't expose financial concepts.
        """
        from hidden_regime.models.hmm import HiddenMarkovModel
        from hidden_regime.config.model import HMMConfig

        # Check class attributes don't contain financial terms
        model_attrs = dir(HiddenMarkovModel)

        violations = []
        for attr in model_attrs:
            # Skip private attributes and dunder methods
            if attr.startswith('_'):
                continue

            attr_lower = attr.lower()
            for term in FORBIDDEN_FINANCIAL_TERMS:
                if term in attr_lower:
                    violations.append(f"Attribute '{attr}' contains financial term '{term}'")

        if violations:
            error_msg = "\n\nMODEL API VIOLATION: Financial terms in HMM interface\n"
            error_msg += "=" * 80 + "\n\n"
            for v in violations:
                error_msg += f"  {v}\n"
            pytest.fail(error_msg)

    def test_model_config_is_domain_agnostic(self):
        """Verify HMMConfig contains no financial-specific parameters."""
        from hidden_regime.config.model import HMMConfig
        import inspect

        # Get HMMConfig __init__ signature
        sig = inspect.signature(HMMConfig.__init__)
        params = sig.parameters.keys()

        violations = []
        for param in params:
            if param == 'self':
                continue

            param_lower = param.lower()
            for term in FORBIDDEN_FINANCIAL_TERMS:
                if term in param_lower:
                    violations.append(f"Parameter '{param}' contains financial term '{term}'")

        if violations:
            error_msg = "\n\nCONFIG VIOLATION: Financial terms in HMMConfig\n"
            error_msg += "=" * 80 + "\n\n"
            for v in violations:
                error_msg += f"  {v}\n"
            pytest.fail(error_msg)


@pytest.mark.architecture
def test_model_directory_structure():
    """Verify models/ directory contains only mathematical algorithms."""
    models_dir = Path(__file__).parent.parent.parent / "hidden_regime" / "models"

    if not models_dir.exists():
        pytest.skip(f"Models directory not found: {models_dir}")

    # Check README.md exists and documents the purity principle
    readme_path = models_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        # README should mention model purity / domain-agnostic
        has_purity_docs = (
            'domain-agnostic' in readme_content.lower() or
            'pure math' in readme_content.lower() or
            'no financial' in readme_content.lower()
        )

        assert has_purity_docs, (
            "models/README.md should document the model purity principle "
            "(domain-agnostic, pure mathematics, no financial concepts)"
        )
