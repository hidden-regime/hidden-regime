"""
Example Verification Harness - Ensure all examples run without errors

This script systematically verifies that all example scripts in /examples/advanced
can run successfully with real market data. Used as a pre-commit check and
continuous integration test.

Purpose: Automated example validation

What it does:
1. Discovers all example scripts
2. Executes each with timeout protection
3. Captures output and errors
4. Generates comprehensive report
5. Fails if any example breaks

Usage:
    python test_examples_runnable.py
    python test_examples_runnable.py --timeout 60
    python test_examples_runnable.py --examples 01,02,05
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse
import time


class ExampleRunner:
    """Runs and validates example scripts."""

    def __init__(self, examples_dir=None, timeout=120):
        """Initialize runner.

        Args:
            examples_dir: Path to examples directory
            timeout: Timeout per example in seconds
        """
        if examples_dir is None:
            examples_dir = Path(__file__).parent
        self.examples_dir = Path(examples_dir)
        self.timeout = timeout
        self.results = {}
        self.start_time = None
        self.end_time = None

    def discover_examples(self):
        """Find all example scripts in directory."""
        examples = []
        for file in sorted(self.examples_dir.glob('*.py')):
            # Skip test files, notebooks, and internal files
            if file.name.startswith('test_'):
                continue
            if file.name.startswith('_'):
                continue
            if 'notebook' in file.name:
                continue
            examples.append(file)
        return examples

    def run_example(self, example_path):
        """Run a single example with timeout.

        Args:
            example_path: Path to example script

        Returns:
            dict: Result with status, output, error, runtime
        """
        print(f"\n{'='*70}")
        print(f"Running: {example_path.name}")
        print(f"{'='*70}")

        start = time.time()
        try:
            # Run with timeout
            result = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(example_path.parent)
            )

            runtime = time.time() - start

            # Check result
            if result.returncode == 0:
                print(f"✓ PASSED ({runtime:.1f}s)")
                return {
                    'status': 'passed',
                    'runtime': runtime,
                    'stdout': result.stdout[:500],  # Truncate for report
                    'error': None
                }
            else:
                print(f"✗ FAILED")
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error (first 500 chars):\n{result.stderr[:500]}")
                return {
                    'status': 'failed',
                    'runtime': runtime,
                    'stdout': result.stdout[:500],
                    'error': result.stderr[:500]
                }

        except subprocess.TimeoutExpired:
            runtime = time.time() - start
            print(f"✗ TIMEOUT (exceeded {self.timeout}s)")
            return {
                'status': 'timeout',
                'runtime': runtime,
                'stdout': '',
                'error': f'Exceeded {self.timeout}s timeout'
            }

        except Exception as e:
            runtime = time.time() - start
            print(f"✗ ERROR: {str(e)}")
            return {
                'status': 'error',
                'runtime': runtime,
                'stdout': '',
                'error': str(e)
            }

    def run_all(self, example_numbers=None):
        """Run all examples.

        Args:
            example_numbers: List of example numbers to run (e.g., [1, 2, 5])
        """
        self.start_time = datetime.now()
        examples = self.discover_examples()

        if example_numbers:
            # Filter examples
            filtered = []
            for num in example_numbers:
                pattern = f'{num:02d}_'
                filtered.extend([e for e in examples if pattern in e.name])
            examples = filtered

        print(f"Found {len(examples)} example(s) to run")
        print(f"Timeout: {self.timeout}s per example")

        # Run each example
        for example in examples:
            result = self.run_example(example)
            self.results[example.name] = result

        self.end_time = datetime.now()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results.values() if r['status'] == 'passed')
        failed = sum(1 for r in self.results.values() if r['status'] != 'passed')
        total = len(self.results)
        total_time = (self.end_time - self.start_time).total_seconds()

        print(f"\nResults:")
        print(f"  Passed: {passed}/{total}")
        print(f"  Failed: {failed}/{total}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time: {total_time/total:.1f}s per example")

        # Detailed results
        print(f"\nDetailed Results:")
        for name, result in sorted(self.results.items()):
            status_symbol = '✓' if result['status'] == 'passed' else '✗'
            print(f"  {status_symbol} {name:50s} [{result['status']:8s}] ({result['runtime']:6.1f}s)")

        # Failures
        failures = {k: v for k, v in self.results.items() if v['status'] != 'passed'}
        if failures:
            print(f"\nFailed Examples:")
            for name, result in failures.items():
                print(f"\n  {name}:")
                print(f"    Status: {result['status']}")
                if result['error']:
                    print(f"    Error: {result['error'][:200]}")

        return failed == 0

    def generate_json_report(self, output_file='example_test_results.json'):
        """Generate JSON report for CI/CD integration."""
        report = {
            'timestamp': self.start_time.isoformat(),
            'total_time': (self.end_time - self.start_time).total_seconds(),
            'examples': self.results,
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results.values() if r['status'] == 'passed'),
                'failed': sum(1 for r in self.results.values() if r['status'] != 'passed')
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nJSON report saved to {output_file}")

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify that all examples run successfully'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout per example in seconds (default: 120)'
    )
    parser.add_argument(
        '--examples',
        type=str,
        help='Comma-separated example numbers to run (e.g., "01,02,05")'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Output JSON report file'
    )

    args = parser.parse_args()

    # Parse example numbers
    example_numbers = None
    if args.examples:
        try:
            example_numbers = [int(x.strip()) for x in args.examples.split(',')]
        except ValueError:
            print(f"Error: Invalid example numbers: {args.examples}")
            return 1

    # Run examples
    runner = ExampleRunner(timeout=args.timeout)
    runner.run_all(example_numbers=example_numbers)

    # Print summary
    success = runner.print_summary()

    # Generate report
    if args.report:
        runner.generate_json_report(args.report)

    # Return exit code
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
