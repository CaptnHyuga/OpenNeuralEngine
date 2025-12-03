"""Comprehensive test runner for OpenNeuralEngine.

This script provides convenient commands to run different test suites.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def run_pytest(pytest_args: list[str]) -> int:
    """Invoke pytest directly to avoid spawning subprocesses."""
    full_cmd = ['pytest', *pytest_args]
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(full_cmd)}")
    print(f"{'='*70}\n")
    
    return pytest.main(pytest_args)


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenNeuralEngine test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all
  
  # Run only unit tests
  python run_tests.py --unit
  
  # Run with coverage
  python run_tests.py --coverage
  
  # Run specific test file
  python run_tests.py --file tests/test_layers.py
  
  # Run fast tests only (skip slow tests)
  python run_tests.py --fast
  
  # Run with verbose output
  python run_tests.py --all -v
        """
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all tests"
    )
    
    parser.add_argument(
        "--unit", "-u",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run only GPU tests"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage report"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Run specific test file"
    )
    
    parser.add_argument(
        "--marker", "-m",
        type=str,
        help="Run tests with specific marker (e.g., 'unit', 'slow', 'gpu')"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    pytest_args: list[str] = []
    
    # Add verbosity
    if args.verbose:
        pytest_args.append("-vv")
    
    # Add fail fast
    if args.failfast:
        pytest_args.append("-x")
    
    # Add parallel execution
    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # Add coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add markers
    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"])
    elif args.gpu:
        pytest_args.extend(["-m", "gpu"])
    elif args.marker:
        pytest_args.extend(["-m", args.marker])
    
    # Skip slow tests
    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    
    # Specific file
    if args.file:
        pytest_args.append(args.file)
    elif not (args.unit or args.integration or args.gpu or args.marker):
        # Run all tests if no specific marker selected
        pytest_args.append("tests/")
    
    # Run tests
    exit_code = run_pytest(pytest_args)
    
    # Print summary
    print(f"\n{'='*70}")
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print(f"{'='*70}\n")
    
    # Show coverage report location if generated
    if args.coverage:
        coverage_html = Path("htmlcov/index.html")
        if coverage_html.exists():
            print(f"📊 Coverage report: {coverage_html.absolute()}\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
