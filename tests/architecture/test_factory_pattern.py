"""
Test Factory Pattern Enforcement - Ensure factory pattern is used correctly.

This test enforces that:
1. Examples use factory functions (not direct instantiation)
2. Component factories are available and functional
3. Deprecation warnings exist for direct instantiation

This is critical for:
- Consistent component creation
- Future architectural changes (factories can add validation, caching, etc.)
- User education through examples
"""

import ast
import re
from pathlib import Path

import pytest


# Components that should ONLY be created via factory
FACTORY_REQUIRED_CLASSES = [
    'HiddenMarkovModel',
    'FinancialDataLoader',
    'FinancialObservationGenerator',
]

# Directories where factory pattern is REQUIRED
FACTORY_REQUIRED_DIRS = [
    'examples/',
]


def get_python_files(directory_pattern: str):
    """Get all Python files matching directory pattern."""
    repo_root = Path(__file__).parent.parent.parent
    pattern_path = repo_root / directory_pattern

    if not pattern_path.exists():
        return []

    return list(pattern_path.rglob("*.py"))


def find_direct_instantiations(file_path: Path) -> list:
    """
    Find direct instantiations of components that should use factories.

    Returns list of (line_number, class_name, code_snippet) tuples.
    """
    violations = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST to find instantiations
        tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if this is a direct class instantiation
                if isinstance(node.func, ast.Name):
                    class_name = node.func.id
                    if class_name in FACTORY_REQUIRED_CLASSES:
                        # Get line number and code snippet
                        line_num = node.lineno
                        violations.append((line_num, class_name))

    except SyntaxError:
        # Skip files with syntax errors
        pass

    return violations


@pytest.mark.architecture
class TestFactoryPattern:
    """Test that factory pattern is enforced in examples."""

    def test_examples_use_factory_pattern(self):
        """
        Verify all examples use factory pattern instead of direct instantiation.

        Examples are the primary way users learn the library. They MUST demonstrate
        best practices by using factory functions.
        """
        violations = []

        for dir_pattern in FACTORY_REQUIRED_DIRS:
            python_files = get_python_files(dir_pattern)

            for file_path in python_files:
                # Skip __init__.py files
                if file_path.name == '__init__.py':
                    continue

                file_violations = find_direct_instantiations(file_path)

                for line_num, class_name in file_violations:
                    # Get relative path for cleaner output
                    rel_path = file_path.relative_to(Path(__file__).parent.parent.parent)
                    violations.append({
                        'file': str(rel_path),
                        'line': line_num,
                        'class': class_name,
                    })

        if violations:
            error_msg = "\n\nFACTORY PATTERN VIOLATION: Direct instantiation in examples\n"
            error_msg += "=" * 80 + "\n\n"
            error_msg += "Examples MUST use factory pattern to teach best practices.\n\n"
            error_msg += "Violations found:\n\n"

            for v in violations:
                error_msg += f"  {v['file']}:{v['line']} - Direct {v['class']}() call\n"

            error_msg += "\nTo fix:\n"
            error_msg += "  Replace direct instantiation with factory:\n\n"
            error_msg += "  # BAD\n"
            error_msg += "  model = HiddenMarkovModel(config)\n\n"
            error_msg += "  # GOOD\n"
            error_msg += "  from hidden_regime.factories import component_factory\n"
            error_msg += "  model = component_factory.create_model_component(config)\n"

            pytest.fail(error_msg)

    def test_component_factories_exist(self):
        """Verify all required factory functions exist and are accessible."""
        from hidden_regime.factories import component_factory

        required_factories = [
            'create_model_component',
            'create_data_component',
            'create_observation_component',
            'create_interpreter_component',
            'create_signal_generator_component',
        ]

        missing = []
        for factory_name in required_factories:
            if not hasattr(component_factory, factory_name):
                missing.append(factory_name)

        if missing:
            error_msg = "\n\nMISSING FACTORIES: Required factory functions not found\n"
            error_msg += "=" * 80 + "\n\n"
            for name in missing:
                error_msg += f"  component_factory.{name}\n"
            pytest.fail(error_msg)

    def test_deprecation_warnings_exist(self):
        """
        Verify that direct instantiation triggers deprecation warnings.

        This helps guide users toward factory pattern even if they bypass it.
        """
        import warnings
        from hidden_regime.models.hmm import HiddenMarkovModel
        from hidden_regime.config.model import HMMConfig

        # Create a simple config
        config = HMMConfig(n_states=2, max_iterations=1)

        # Direct instantiation should trigger FutureWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = HiddenMarkovModel(config)

            # Check that a warning was raised
            assert len(w) > 0, (
                "Direct instantiation of HiddenMarkovModel should trigger FutureWarning. "
                "Add deprecation warning to guide users toward factory pattern."
            )

            # Check warning type
            assert issubclass(w[0].category, FutureWarning), (
                f"Expected FutureWarning, got {w[0].category.__name__}"
            )

            # Check warning message mentions factory
            warning_msg = str(w[0].message).lower()
            assert 'factory' in warning_msg or 'create_' in warning_msg, (
                "Deprecation warning should mention factory pattern. "
                f"Got: {w[0].message}"
            )

    def test_pipeline_factories_exist(self):
        """Verify high-level pipeline factories are available."""
        import hidden_regime as hr

        required_pipeline_factories = [
            'create_financial_pipeline',
            'create_trading_pipeline',
            'create_simple_regime_pipeline',
        ]

        missing = []
        for factory_name in required_pipeline_factories:
            if not hasattr(hr, factory_name):
                missing.append(factory_name)

        if missing:
            error_msg = "\n\nMISSING PIPELINE FACTORIES\n"
            error_msg += "=" * 80 + "\n\n"
            for name in missing:
                error_msg += f"  hidden_regime.{name}\n"
            pytest.fail(error_msg)


@pytest.mark.architecture
def test_factory_pattern_in_documentation():
    """Verify factory pattern is documented in CLAUDE.md and README."""
    repo_root = Path(__file__).parent.parent.parent

    # Check CLAUDE.md
    claude_md = repo_root / "CLAUDE.md"
    if claude_md.exists():
        with open(claude_md, 'r', encoding='utf-8') as f:
            content = f.read()

        assert 'factory' in content.lower(), (
            "CLAUDE.md should document factory pattern usage"
        )

        # Should mention create_ functions
        assert 'create_' in content, (
            "CLAUDE.md should show factory function examples (create_*)"
        )

    # Check README
    readme = repo_root / "README.md"
    if readme.exists():
        with open(readme, 'r', encoding='utf-8') as f:
            content = f.read()

        # README should show at least one factory usage
        has_factory_example = (
            'create_financial_pipeline' in content or
            'create_trading_pipeline' in content or
            'component_factory' in content
        )

        assert has_factory_example, (
            "README.md should demonstrate factory pattern in examples"
        )
