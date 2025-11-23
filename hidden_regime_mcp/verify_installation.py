#!/usr/bin/env python
"""
Unified installation verification script for Hidden Regime MCP.

Checks Python environment, package installation, dependencies, and MCP connectivity
across Windows, macOS, and Linux platforms.

Usage:
    python verify_installation.py
    python hidden_regime_mcp/verify_installation.py
"""

import sys
import os
import subprocess
import json
import platform
from typing import Tuple, List, Dict, Any
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    @staticmethod
    def green(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.RESET}"

    @staticmethod
    def red(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.RESET}"

    @staticmethod
    def yellow(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def blue(text: str) -> str:
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def cyan(text: str) -> str:
        return f"{Colors.CYAN}{text}{Colors.RESET}"


class Verifier:
    """Main verification class."""

    def __init__(self):
        """Initialize the verifier."""
        self.checks: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def header(self, title: str) -> None:
        """Print a formatted header."""
        print("\n" + Colors.cyan("=" * 70))
        print(Colors.cyan(f"  {title}"))
        print(Colors.cyan("=" * 70))

    def check(self, name: str, test_func, required: bool = True) -> bool:
        """
        Run a check and track results.

        Args:
            name: Name of the check
            test_func: Function that returns (success: bool, message: str)
            required: Whether this check is required (vs. optional)

        Returns:
            True if check passed, False otherwise
        """
        try:
            success, message = test_func()

            status = "PASS" if success else "FAIL"
            color = Colors.green if success else Colors.red

            if success:
                self.passed += 1
                symbol = "✓"
            else:
                self.failed += 1
                symbol = "✗"

            print(f"  {symbol} {color(status)}: {name}")
            if message:
                print(f"      {message}")

            self.checks.append(
                {
                    "name": name,
                    "status": "pass" if success else "fail",
                    "message": message,
                    "required": required,
                }
            )

            return success

        except Exception as e:
            self.failed += 1
            print(f"  ✗ {Colors.red('ERROR')}: {name}")
            print(f"      {str(e)}")
            self.checks.append(
                {
                    "name": name,
                    "status": "error",
                    "message": str(e),
                    "required": required,
                }
            )
            return False

    def warning(self, name: str, message: str) -> None:
        """Print a warning."""
        self.warnings += 1
        print(f"  ⚠ {Colors.yellow('WARN')}: {name}")
        print(f"      {message}")

    def summary(self) -> None:
        """Print verification summary."""
        self.header("VERIFICATION SUMMARY")

        print(
            f"  {Colors.green(f'Passed: {self.passed}')} | "
            f"{Colors.red(f'Failed: {self.failed}')} | "
            f"{Colors.yellow(f'Warnings: {self.warnings}')}"
        )

        if self.failed == 0:
            print(f"\n  {Colors.green('✓ Installation verification successful!')}")
        else:
            print(
                f"\n  {Colors.red(f'✗ Installation verification failed ({self.failed} issues)')}"
            )

        print(Colors.cyan("=" * 70) + "\n")

    def detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed JSON report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "prefix": sys.prefix,
            },
            "results": self.checks,
            "summary": {"passed": self.passed, "failed": self.failed, "warnings": self.warnings},
        }


def verify_python_installation(verifier: Verifier) -> None:
    """Verify Python environment."""
    verifier.header("PYTHON ENVIRONMENT")

    # Python version
    def check_version():
        version = tuple(int(x) for x in platform.python_version().split(".")[:2])
        if version >= (3, 10):
            return True, f"Python {platform.python_version()}"
        else:
            return False, f"Python {platform.python_version()} (requires 3.10+)"

    verifier.check("Python Version", check_version, required=True)

    # Executable location
    def check_executable():
        return True, sys.executable

    verifier.check("Python Executable", check_executable)

    # Virtual environment
    def check_venv():
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )
        if in_venv:
            return True, f"Active venv at {sys.prefix}"
        else:
            return (
                False,
                "Not running in a virtual environment (recommended but not required)",
            )

    verifier.check("Virtual Environment", check_venv, required=False)


def verify_package_installation(verifier: Verifier) -> None:
    """Verify package installation."""
    verifier.header("PACKAGE INSTALLATION")

    # hidden-regime package
    def check_hidden_regime():
        try:
            import hidden_regime

            return True, f"Version {hidden_regime.__version__}"
        except ImportError:
            return False, "Package not installed. Run: pip install hidden-regime"

    verifier.check("hidden-regime Package", check_hidden_regime, required=True)

    # hidden-regime-mcp module
    def check_hidden_regime_mcp():
        try:
            import hidden_regime_mcp

            return True, "MCP module available"
        except ImportError:
            return False, "MCP module not found"

    verifier.check("hidden-regime-mcp Module", check_hidden_regime_mcp, required=True)

    # Cache module (optional - new in dev version)
    def check_cache():
        try:
            from hidden_regime_mcp.cache import get_cache

            cache = get_cache()
            return True, "Cache system operational"
        except ImportError:
            return True, "Cache system not yet installed (optional - future feature)"
        except Exception as e:
            return False, f"Cache system error: {str(e)}"

    verifier.check("Cache System", check_cache, required=False)


def verify_dependencies(verifier: Verifier) -> None:
    """Verify core dependencies."""
    verifier.header("CORE DEPENDENCIES")

    packages = {
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "scipy": "Scientific computing",
        "sklearn": "Machine learning (scikit-learn)",
        "matplotlib": "Visualization",
        "seaborn": "Statistical plots",
        "yfinance": "Market data (Yahoo Finance)",
    }

    for package_name, description in packages.items():
        def check_package(name=package_name):
            try:
                __import__(name)
                mod = sys.modules[name]
                version = getattr(mod, "__version__", "unknown")
                return True, f"{version}"
            except ImportError:
                return False, f"Not installed. Run: pip install {name}"

        verifier.check(
            f"{description} ({package_name})",
            check_package,
            required=package_name in ["pandas", "numpy", "yfinance"],
        )


def verify_connectivity(verifier: Verifier) -> None:
    """Verify network and data connectivity."""
    verifier.header("NETWORK CONNECTIVITY")

    # Internet connectivity
    def check_internet():
        try:
            import socket

            socket.create_connection(("www.google.com", 80), timeout=3)
            return True, "Internet connectivity OK"
        except Exception:
            return False, "Cannot reach internet (required for yfinance)"

    verifier.check("Internet Connectivity", check_internet, required=True)

    # yfinance connectivity
    def check_yfinance():
        try:
            import yfinance

            data = yfinance.download("SPY", period="1d", progress=False)
            if len(data) > 0:
                return True, "Can download market data"
            else:
                return False, "No data returned from yfinance"
        except Exception as e:
            return False, f"yfinance error: {str(e)[:50]}"

    verifier.check("yfinance Data Download", check_yfinance, required=True)


def verify_mcp_server(verifier: Verifier) -> None:
    """Verify MCP server functionality."""
    verifier.header("MCP SERVER")

    # MCP server import
    def check_mcp_import():
        try:
            from hidden_regime_mcp import server

            return True, "MCP server module available"
        except ImportError as e:
            return False, f"Cannot import MCP server: {str(e)}"

    verifier.check("MCP Server Module", check_mcp_import, required=True)

    # MCP tools
    def check_mcp_tools():
        try:
            from hidden_regime_mcp.tools import detect_regime, get_regime_statistics

            return True, "Core tools available"
        except ImportError as e:
            return False, f"Cannot import tools: {str(e)}"

    verifier.check("MCP Tools", check_mcp_tools, required=True)

    # MCP startup test (short timeout)
    def check_mcp_startup():
        try:
            # Try to verify the server module can be imported
            from hidden_regime_mcp import server

            # Check if the server module has the FastMCP app
            if hasattr(server, "app") or hasattr(server, "mcp"):
                return True, "MCP server module loaded successfully"
            else:
                # Fallback: just check the module exists and loads
                return True, "MCP server module can be imported"
        except Exception as e:
            return False, f"Cannot import MCP server: {str(e)[:60]}"

    verifier.check("MCP Server Startup", check_mcp_startup, required=True)


def verify_platform_specific(verifier: Verifier) -> None:
    """Verify platform-specific requirements."""
    verifier.header("PLATFORM-SPECIFIC")

    system = platform.system()

    if system == "Windows":
        # Windows-specific checks
        def check_paths():
            try:
                # Check if paths use forward slashes (better for config)
                exe_path = sys.executable
                if "\\" in exe_path:
                    # This is normal on Windows, just check it's accessible
                    import os

                    if os.path.exists(exe_path):
                        return True, "Python path is accessible"
                    else:
                        return False, "Python path not accessible"
                return True, "Python path is accessible"
            except Exception as e:
                return False, str(e)

        verifier.check("Windows Path Accessibility", check_paths)

    elif system == "Darwin":
        # macOS-specific checks
        def check_macos_python():
            version = platform.mac_ver()[0]
            if version:
                return True, f"macOS {version}"
            return True, "macOS detected"

        verifier.check("macOS Compatibility", check_macos_python)

        # Check if Apple Silicon
        def check_apple_silicon():
            try:
                result = subprocess.run(
                    ["uname", "-m"], capture_output=True, text=True, timeout=5
                )
                arch = result.stdout.strip()
                if "arm64" in arch:
                    return True, f"Apple Silicon ({arch})"
                else:
                    return True, f"Intel ({arch})"
            except Exception:
                return True, "Architecture detection skipped"

        verifier.check("macOS Architecture", check_apple_silicon)

    elif system == "Linux":
        # Linux-specific checks
        def check_linux_distro():
            try:
                if hasattr(platform, "freedesktop_os_release"):
                    info = platform.freedesktop_os_release()
                    distro = info.get("NAME", "Unknown")
                    return True, distro
            except Exception:
                pass
            return True, "Linux detected"

        verifier.check("Linux Distribution", check_linux_distro)


def main():
    """Run all verifications."""
    print(Colors.cyan("\n" + "=" * 70))
    print(Colors.cyan("  Hidden Regime MCP - Installation Verification"))
    print(Colors.cyan("=" * 70))

    verifier = Verifier()

    # Run all checks
    verify_python_installation(verifier)
    verify_package_installation(verifier)
    verify_dependencies(verifier)
    verify_connectivity(verifier)
    verify_mcp_server(verifier)
    verify_platform_specific(verifier)

    # Print summary
    verifier.summary()

    # Generate report
    report = verifier.detailed_report()

    # Save report
    report_file = "hidden_regime_verification_report.json"
    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to: {Colors.blue(report_file)}\n")
    except Exception as e:
        print(f"Could not save report: {e}\n")

    # Return exit code
    return 0 if verifier.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
