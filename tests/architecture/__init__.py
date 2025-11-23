"""
Architectural Compliance Tests

These tests enforce core architectural principles of the Hidden Regime library.
Unlike unit/integration tests that verify functionality, architectural tests
verify structural integrity and design patterns.

Tests in this directory ensure:
1. Model Purity - Models contain only mathematics, no financial concepts
2. Factory Pattern - Components are created via factories, not direct instantiation
3. Pipeline Flow - Component data flow follows documented architecture
4. Interpreter Separation - Financial logic isolated in interpreter layer

Running Architectural Tests:
    pytest tests/architecture/ -v

These tests run in CI to prevent architectural regression.
"""
