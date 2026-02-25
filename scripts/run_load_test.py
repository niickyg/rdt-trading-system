#!/usr/bin/env python3
"""
Load Test Runner for RDT Trading System

CLI wrapper for Locust load testing with:
- Configurable users, spawn rate, and duration
- HTML report generation
- CI/CD integration
- Performance threshold validation

Usage:
    # Basic usage
    python scripts/run_load_test.py

    # Custom configuration
    python scripts/run_load_test.py --users 100 --spawn-rate 10 --duration 300

    # CI/CD mode with thresholds
    python scripts/run_load_test.py --ci --fail-ratio 0.01 --p95-threshold 500

    # Generate HTML report
    python scripts/run_load_test.py --html-report results/load_test.html

    # Web UI mode
    python scripts/run_load_test.py --web
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run load tests for RDT Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --users 50 --spawn-rate 5 --duration 60
    %(prog)s --ci --fail-ratio 0.01
    %(prog)s --web --host http://localhost:5000
        """
    )

    # Target configuration
    parser.add_argument(
        '--host', '-H',
        default=os.getenv('API_BASE_URL', 'http://localhost:5000'),
        help='Target host URL (default: http://localhost:5000)'
    )

    # Load configuration
    parser.add_argument(
        '--users', '-u',
        type=int,
        default=50,
        help='Total number of simulated users (default: 50)'
    )

    parser.add_argument(
        '--spawn-rate', '-r',
        type=float,
        default=5.0,
        help='Users spawned per second (default: 5.0)'
    )

    parser.add_argument(
        '--duration', '-t',
        type=int,
        default=60,
        help='Test duration in seconds (default: 60)'
    )

    # Test mode
    parser.add_argument(
        '--web',
        action='store_true',
        help='Start Locust web UI instead of headless mode'
    )

    parser.add_argument(
        '--web-port',
        type=int,
        default=8089,
        help='Port for Locust web UI (default: 8089)'
    )

    # CI/CD configuration
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: strict thresholds, non-zero exit on failure'
    )

    parser.add_argument(
        '--fail-ratio',
        type=float,
        default=0.05,
        help='Maximum acceptable failure ratio (default: 0.05 = 5%%)'
    )

    parser.add_argument(
        '--p95-threshold',
        type=float,
        default=1000.0,
        help='Maximum acceptable P95 response time in ms (default: 1000)'
    )

    parser.add_argument(
        '--p99-threshold',
        type=float,
        default=2000.0,
        help='Maximum acceptable P99 response time in ms (default: 2000)'
    )

    # Output configuration
    parser.add_argument(
        '--html-report',
        type=str,
        help='Path for HTML report output'
    )

    parser.add_argument(
        '--csv-prefix',
        type=str,
        help='Prefix for CSV output files'
    )

    parser.add_argument(
        '--json-report',
        type=str,
        help='Path for JSON report output'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/load_tests',
        help='Directory for test output (default: results/load_tests)'
    )

    # User classes
    parser.add_argument(
        '--user-classes',
        type=str,
        nargs='+',
        default=['APIUser', 'DashboardUser', 'MixedUser'],
        help='Locust user classes to run (default: APIUser DashboardUser MixedUser)'
    )

    # Test shape
    parser.add_argument(
        '--shape',
        choices=['constant', 'stages', 'spike'],
        default='constant',
        help='Load test shape (default: constant)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )

    return parser.parse_args()


def check_locust_installed() -> bool:
    """Check if Locust is installed."""
    try:
        import locust
        return True
    except ImportError:
        return False


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_timestamp() -> str:
    """Generate timestamp for file names."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def build_locust_command(args) -> List[str]:
    """Build Locust command from arguments."""
    locustfile = PROJECT_ROOT / 'tests' / 'performance' / 'locustfile.py'

    cmd = [
        sys.executable, '-m', 'locust',
        '-f', str(locustfile),
        '--host', args.host,
    ]

    if args.web:
        # Web UI mode
        cmd.extend(['--web-port', str(args.web_port)])
    else:
        # Headless mode
        cmd.extend([
            '--headless',
            '-u', str(args.users),
            '-r', str(args.spawn_rate),
            '-t', f'{args.duration}s',
        ])

    # Add HTML report if specified
    if args.html_report:
        cmd.extend(['--html', args.html_report])
    elif not args.web:
        # Generate default HTML report
        output_dir = ensure_output_dir(args.output_dir)
        timestamp = generate_timestamp()
        html_path = output_dir / f'load_test_{timestamp}.html'
        cmd.extend(['--html', str(html_path)])

    # Add CSV output if specified
    if args.csv_prefix:
        cmd.extend(['--csv', args.csv_prefix])

    # Add user classes
    if args.user_classes:
        for user_class in args.user_classes:
            cmd.extend(['--class-picker' if args.web else '--class', user_class])

    return cmd


def run_locust(cmd: List[str], args) -> subprocess.CompletedProcess:
    """Run Locust with the given command."""
    if args.verbose:
        print(f"Running: {' '.join(cmd)}")

    return subprocess.run(
        cmd,
        capture_output=not args.verbose,
        text=True,
        cwd=str(PROJECT_ROOT)
    )


def parse_locust_stats(csv_prefix: str) -> Optional[Dict[str, Any]]:
    """Parse Locust stats from CSV output."""
    stats_file = Path(f"{csv_prefix}_stats.csv")
    if not stats_file.exists():
        return None

    try:
        import csv

        stats = {
            'endpoints': [],
            'total_requests': 0,
            'total_failures': 0,
            'avg_response_time': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0,
        }

        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Name'] == 'Aggregated':
                    stats['total_requests'] = int(row.get('Request Count', 0))
                    stats['total_failures'] = int(row.get('Failure Count', 0))
                    stats['avg_response_time'] = float(row.get('Average Response Time', 0))
                    stats['p50'] = float(row.get('50%', 0))
                    stats['p95'] = float(row.get('95%', 0))
                    stats['p99'] = float(row.get('99%', 0))
                else:
                    stats['endpoints'].append({
                        'name': row['Name'],
                        'requests': int(row.get('Request Count', 0)),
                        'failures': int(row.get('Failure Count', 0)),
                        'avg_time': float(row.get('Average Response Time', 0)),
                        'p95': float(row.get('95%', 0)),
                    })

        return stats

    except Exception as e:
        print(f"Error parsing stats: {e}")
        return None


def validate_results(stats: Dict[str, Any], args) -> bool:
    """Validate load test results against thresholds."""
    passed = True
    issues = []

    # Check failure ratio
    if stats['total_requests'] > 0:
        fail_ratio = stats['total_failures'] / stats['total_requests']
        if fail_ratio > args.fail_ratio:
            passed = False
            issues.append(f"Failure ratio {fail_ratio:.2%} exceeds threshold {args.fail_ratio:.2%}")

    # Check P95 response time
    if stats['p95'] > args.p95_threshold:
        passed = False
        issues.append(f"P95 response time {stats['p95']:.0f}ms exceeds threshold {args.p95_threshold}ms")

    # Check P99 response time
    if stats['p99'] > args.p99_threshold:
        passed = False
        issues.append(f"P99 response time {stats['p99']:.0f}ms exceeds threshold {args.p99_threshold}ms")

    return passed, issues


def print_summary(stats: Dict[str, Any], passed: bool, issues: List[str], args):
    """Print test summary."""
    if args.quiet:
        print("PASS" if passed else "FAIL")
        return

    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)

    print(f"\nTarget: {args.host}")
    print(f"Users: {args.users}")
    print(f"Spawn Rate: {args.spawn_rate}/s")
    print(f"Duration: {args.duration}s")

    print(f"\nResults:")
    print(f"  Total Requests: {stats['total_requests']:,}")
    print(f"  Failed Requests: {stats['total_failures']:,}")
    if stats['total_requests'] > 0:
        print(f"  Failure Rate: {stats['total_failures'] / stats['total_requests']:.2%}")
    print(f"  Avg Response Time: {stats['avg_response_time']:.0f}ms")
    print(f"  P50 Response Time: {stats['p50']:.0f}ms")
    print(f"  P95 Response Time: {stats['p95']:.0f}ms")
    print(f"  P99 Response Time: {stats['p99']:.0f}ms")

    if stats['endpoints']:
        print("\nEndpoint Stats:")
        for ep in stats['endpoints'][:10]:  # Top 10
            print(f"  {ep['name']}: {ep['requests']} reqs, {ep['avg_time']:.0f}ms avg, P95={ep['p95']:.0f}ms")

    print("\n" + "-" * 60)
    if passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for issue in issues:
            print(f"  - {issue}")
    print("=" * 60)


def save_json_report(stats: Dict[str, Any], passed: bool, issues: List[str], args):
    """Save results as JSON report."""
    if not args.json_report:
        output_dir = ensure_output_dir(args.output_dir)
        timestamp = generate_timestamp()
        args.json_report = str(output_dir / f'load_test_{timestamp}.json')

    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'host': args.host,
            'users': args.users,
            'spawn_rate': args.spawn_rate,
            'duration': args.duration,
        },
        'thresholds': {
            'fail_ratio': args.fail_ratio,
            'p95_threshold': args.p95_threshold,
            'p99_threshold': args.p99_threshold,
        },
        'results': stats,
        'passed': passed,
        'issues': issues,
    }

    with open(args.json_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nJSON report saved to: {args.json_report}")


def run_quick_smoke_test(args) -> bool:
    """Run a quick smoke test before full load test."""
    print("Running smoke test...")

    try:
        import requests

        # Test health endpoint
        response = requests.get(f"{args.host}/api/v1/health", timeout=10)
        if response.status_code != 200:
            print(f"Smoke test failed: health endpoint returned {response.status_code}")
            return False

        # Test status endpoint
        response = requests.get(f"{args.host}/api/v1/status", timeout=10)
        if response.status_code != 200:
            print(f"Smoke test failed: status endpoint returned {response.status_code}")
            return False

        print("Smoke test passed")
        return True

    except requests.exceptions.ConnectionError:
        print(f"Smoke test failed: Cannot connect to {args.host}")
        return False
    except Exception as e:
        print(f"Smoke test failed: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_args()

    # Check Locust is installed
    if not check_locust_installed():
        print("ERROR: Locust is not installed.")
        print("Install with: pip install locust")
        return 1

    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 60)
        print("RDT Trading System Load Test")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Target Host: {args.host}")
        print(f"  Users: {args.users}")
        print(f"  Spawn Rate: {args.spawn_rate}/s")
        print(f"  Duration: {args.duration}s")
        if args.ci:
            print(f"  CI Mode: Enabled")
            print(f"  Fail Ratio Threshold: {args.fail_ratio:.2%}")
            print(f"  P95 Threshold: {args.p95_threshold}ms")

    # Run smoke test in CI mode
    if args.ci and not args.web:
        if not run_quick_smoke_test(args):
            return 1

    # Prepare output directory
    output_dir = ensure_output_dir(args.output_dir)
    timestamp = generate_timestamp()

    # Set up CSV output for stats parsing
    if not args.csv_prefix and not args.web:
        args.csv_prefix = str(output_dir / f'load_test_{timestamp}')

    # Build and run Locust command
    cmd = build_locust_command(args)

    if not args.quiet:
        print(f"\nStarting load test...")
        if args.web:
            print(f"Web UI available at: http://localhost:{args.web_port}")

    result = run_locust(cmd, args)

    if args.web:
        # Web UI mode - just return the exit code
        return result.returncode

    # Parse results
    stats = parse_locust_stats(args.csv_prefix)

    if stats is None:
        print("WARNING: Could not parse test statistics")
        stats = {
            'endpoints': [],
            'total_requests': 0,
            'total_failures': 0,
            'avg_response_time': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0,
        }

    # Validate results
    passed, issues = validate_results(stats, args)

    # Print summary
    print_summary(stats, passed, issues, args)

    # Save JSON report
    if args.json_report or args.ci:
        save_json_report(stats, passed, issues, args)

    # Return exit code
    if args.ci:
        return 0 if passed else 1

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
