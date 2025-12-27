#!/usr/bin/env python3
"""
Convenience script to run both GPT and Claude analyses automatically.

Usage:
    python analyze_both.py [--date YYYY-MM-DD]
"""

import subprocess
import sys
import argparse
import datetime as dt

def main():
    parser = argparse.ArgumentParser(description="Run both GPT and Claude analyses")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date folder (YYYY-MM-DD). Defaults to today's date."
    )
    
    args = parser.parse_args()
    
    if args.date:
        date_str = args.date
    else:
        date_str = dt.date.today().strftime("%Y-%m-%d")
    
    print(f"Running analyses for date: {date_str}\n")
    
    # Run GPT analysis
    print("=" * 60)
    print("Running GPT-5.2 analysis...")
    print("=" * 60)
    result1 = subprocess.run(
        [sys.executable, "analyze_packets.py", "--date", date_str, "--provider", "openai"],
        capture_output=False
    )
    
    if result1.returncode != 0:
        print(f"\nError: GPT analysis failed with code {result1.returncode}")
        return result1.returncode
    
    print("\n")
    
    # Run Claude analysis
    print("=" * 60)
    print("Running Claude Sonnet 4.5 analysis...")
    print("=" * 60)
    result2 = subprocess.run(
        [sys.executable, "analyze_packets.py", "--date", date_str, "--provider", "anthropic", "--model", "claude-sonnet-4-5-20250929"],
        capture_output=False
    )
    
    if result2.returncode != 0:
        print(f"\nError: Claude analysis failed with code {result2.returncode}")
        return result2.returncode
    
    print("\n" + "=" * 60)
    print("Both analyses completed successfully!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

