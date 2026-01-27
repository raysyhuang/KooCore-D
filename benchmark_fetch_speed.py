"""
Benchmark: Yahoo Finance vs Polygon Fetching Speed

This script compares data fetching performance between Yahoo Finance and Polygon APIs.
Tests single ticker, small batch, and large batch scenarios.
"""

import os
import time
import statistics
from datetime import datetime, timedelta
from typing import Callable
import pandas as pd

# Disable yfinance progress bars for cleaner output
import yfinance as yf
yf.set_tz_cache_location("/tmp/yf_cache")

from src.core.yf import download_daily_range, get_ticker_df
from src.core.polygon import fetch_polygon_daily, download_polygon_batch


def get_polygon_api_key() -> str | None:
    """Get Polygon API key from environment."""
    return os.environ.get("POLYGON_API_KEY")


def benchmark_function(func: Callable, name: str, iterations: int = 3) -> dict:
    """
    Benchmark a function multiple times and return statistics.
    
    Returns dict with: mean, min, max, std, iterations, times
    """
    times = []
    errors = []
    
    for i in range(iterations):
        try:
            start = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # Small delay between iterations to be nice to APIs
            if i < iterations - 1:
                time.sleep(0.5)
        except Exception as e:
            errors.append(str(e))
    
    if not times:
        return {
            "name": name,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "iterations": iterations,
            "successful": 0,
            "errors": errors,
        }
    
    return {
        "name": name,
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "iterations": iterations,
        "successful": len(times),
        "errors": errors,
        "times": times,
    }


def print_benchmark_result(result: dict, indent: str = "  "):
    """Pretty print benchmark results."""
    if result["mean"] is None:
        print(f"{indent}FAILED - Errors: {result['errors'][:2]}")
        return
    
    print(f"{indent}Mean: {result['mean']:.3f}s")
    print(f"{indent}Min:  {result['min']:.3f}s")
    print(f"{indent}Max:  {result['max']:.3f}s")
    if result["std"]:
        print(f"{indent}Std:  {result['std']:.3f}s")
    print(f"{indent}Success: {result['successful']}/{result['iterations']}")


def test_single_ticker(ticker: str, lookback_days: int, api_key: str | None):
    """Test fetching a single ticker from both sources."""
    print(f"\n{'='*60}")
    print(f"TEST: Single Ticker ({ticker}) - {lookback_days} days lookback")
    print(f"{'='*60}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Yahoo Finance test
    def yf_fetch():
        data, report = download_daily_range(
            [ticker], 
            start_str, 
            end_str, 
            use_cache=False,  # Disable cache for fair comparison
            progress=False,
        )
        df = get_ticker_df(data, ticker)
        return df
    
    print("\n[Yahoo Finance]")
    yf_result = benchmark_function(yf_fetch, f"YF-{ticker}")
    print_benchmark_result(yf_result)
    
    # Polygon test
    if api_key:
        def polygon_fetch():
            df = fetch_polygon_daily(ticker, lookback_days, end_str, api_key)
            return df
        
        print("\n[Polygon]")
        polygon_result = benchmark_function(polygon_fetch, f"Polygon-{ticker}")
        print_benchmark_result(polygon_result)
        
        # Comparison
        if yf_result["mean"] and polygon_result["mean"]:
            speedup = yf_result["mean"] / polygon_result["mean"]
            winner = "Polygon" if speedup > 1 else "Yahoo Finance"
            print(f"\n>>> {winner} is {abs(speedup):.2f}x faster for single ticker")
    else:
        print("\n[Polygon] SKIPPED - No API key")
    
    return yf_result, polygon_result if api_key else None


def test_batch_tickers(tickers: list[str], lookback_days: int, api_key: str | None):
    """Test fetching multiple tickers from both sources."""
    n = len(tickers)
    print(f"\n{'='*60}")
    print(f"TEST: Batch ({n} tickers) - {lookback_days} days lookback")
    print(f"Tickers: {', '.join(tickers[:5])}{'...' if n > 5 else ''}")
    print(f"{'='*60}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Yahoo Finance test (batch download)
    def yf_fetch():
        data, report = download_daily_range(
            tickers, 
            start_str, 
            end_str, 
            use_cache=False,
            progress=False,
        )
        # Extract all ticker data to ensure it's actually processed
        results = {t: get_ticker_df(data, t) for t in tickers}
        valid = sum(1 for df in results.values() if not df.empty)
        return results, valid
    
    print("\n[Yahoo Finance]")
    yf_result = benchmark_function(yf_fetch, f"YF-batch-{n}")
    print_benchmark_result(yf_result)
    
    # Polygon test (parallel download)
    if api_key:
        def polygon_fetch():
            results = download_polygon_batch(
                tickers, 
                lookback_days, 
                end_str, 
                api_key,
                max_workers=8,
            )
            valid = sum(1 for df in results.values() if not df.empty)
            return results, valid
        
        print("\n[Polygon]")
        polygon_result = benchmark_function(polygon_fetch, f"Polygon-batch-{n}")
        print_benchmark_result(polygon_result)
        
        # Comparison
        if yf_result["mean"] and polygon_result["mean"]:
            speedup = yf_result["mean"] / polygon_result["mean"]
            winner = "Polygon" if speedup > 1 else "Yahoo Finance"
            faster_ratio = max(speedup, 1/speedup)
            print(f"\n>>> {winner} is {faster_ratio:.2f}x faster for {n} tickers")
            
            # Per-ticker timing
            yf_per = yf_result["mean"] / n
            poly_per = polygon_result["mean"] / n
            print(f"    YF per ticker:      {yf_per*1000:.1f}ms")
            print(f"    Polygon per ticker: {poly_per*1000:.1f}ms")
    else:
        print("\n[Polygon] SKIPPED - No API key")
    
    return yf_result, polygon_result if api_key else None


def test_rate_limits(api_key: str | None):
    """Test how the APIs handle rapid requests (rate limit behavior)."""
    print(f"\n{'='*60}")
    print("TEST: Rate Limit Behavior (10 rapid requests)")
    print(f"{'='*60}")
    
    ticker = "AAPL"
    lookback_days = 30
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Yahoo Finance - 10 rapid requests
    print("\n[Yahoo Finance - 10 rapid sequential requests]")
    yf_times = []
    for i in range(10):
        start = time.perf_counter()
        try:
            data, _ = download_daily_range(
                [ticker], 
                (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
                end_date,
                use_cache=False,
                progress=False,
            )
            elapsed = time.perf_counter() - start
            yf_times.append(elapsed)
        except Exception as e:
            yf_times.append(None)
            print(f"  Request {i+1}: ERROR - {e}")
    
    valid_yf = [t for t in yf_times if t is not None]
    if valid_yf:
        print(f"  Success: {len(valid_yf)}/10")
        print(f"  Mean: {statistics.mean(valid_yf):.3f}s")
        print(f"  Total: {sum(valid_yf):.3f}s")
    
    # Polygon - 10 rapid requests
    if api_key:
        print("\n[Polygon - 10 rapid sequential requests]")
        poly_times = []
        for i in range(10):
            start = time.perf_counter()
            try:
                df = fetch_polygon_daily(ticker, lookback_days, end_date, api_key)
                elapsed = time.perf_counter() - start
                poly_times.append(elapsed)
            except Exception as e:
                poly_times.append(None)
                print(f"  Request {i+1}: ERROR - {e}")
        
        valid_poly = [t for t in poly_times if t is not None]
        if valid_poly:
            print(f"  Success: {len(valid_poly)}/10")
            print(f"  Mean: {statistics.mean(valid_poly):.3f}s")
            print(f"  Total: {sum(valid_poly):.3f}s")
    else:
        print("\n[Polygon] SKIPPED - No API key")


def main():
    print("="*60)
    print(" BENCHMARK: Yahoo Finance vs Polygon Fetching Speed")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    api_key = get_polygon_api_key()
    if api_key:
        print(f"Polygon API Key: {'*' * 8}...{api_key[-4:]}")
    else:
        print("Polygon API Key: NOT FOUND (set POLYGON_API_KEY env var)")
        print("         Only Yahoo Finance tests will run.")
    
    # Test configurations
    test_tickers_small = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    test_tickers_medium = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
        "JPM", "V", "UNH", "HD", "PG", "MA", "DIS", "PYPL", "NFLX", "ADBE",
        "CRM", "INTC"
    ]
    test_tickers_large = test_tickers_medium + [
        "CSCO", "VZ", "T", "PFE", "MRK", "ABT", "NKE", "KO", "PEP", "WMT",
        "CVX", "XOM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "IBM",
        "ORCL", "QCOM", "AMD", "TXN", "AVGO", "MU", "AMAT", "LRCX", "NOW", "SNOW"
    ]
    
    results = {}
    
    # Test 1: Single ticker
    print("\n" + "="*60)
    print(" SECTION 1: Single Ticker Tests")
    print("="*60)
    results["single_30d"] = test_single_ticker("AAPL", 30, api_key)
    results["single_180d"] = test_single_ticker("AAPL", 180, api_key)
    
    # Test 2: Small batch (5 tickers)
    print("\n" + "="*60)
    print(" SECTION 2: Small Batch Tests (5 tickers)")
    print("="*60)
    results["batch_5_60d"] = test_batch_tickers(test_tickers_small, 60, api_key)
    
    # Test 3: Medium batch (20 tickers)
    print("\n" + "="*60)
    print(" SECTION 3: Medium Batch Tests (20 tickers)")
    print("="*60)
    results["batch_20_60d"] = test_batch_tickers(test_tickers_medium, 60, api_key)
    
    # Test 4: Large batch (50 tickers)
    print("\n" + "="*60)
    print(" SECTION 4: Large Batch Tests (50 tickers)")
    print("="*60)
    results["batch_50_60d"] = test_batch_tickers(test_tickers_large, 60, api_key)
    
    # Test 5: Rate limits
    print("\n" + "="*60)
    print(" SECTION 5: Rate Limit Behavior")
    print("="*60)
    test_rate_limits(api_key)
    
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    print("\n| Test               | Yahoo Finance | Polygon    | Winner   | Speedup |")
    print("|" + "-"*20 + "|" + "-"*15 + "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*9 + "|")
    
    for name, (yf_res, poly_res) in results.items():
        yf_time = f"{yf_res['mean']:.3f}s" if yf_res and yf_res['mean'] else "N/A"
        poly_time = f"{poly_res['mean']:.3f}s" if poly_res and poly_res['mean'] else "N/A"
        
        if yf_res and yf_res['mean'] and poly_res and poly_res['mean']:
            speedup = yf_res['mean'] / poly_res['mean']
            winner = "Polygon" if speedup > 1.05 else ("YF" if speedup < 0.95 else "Tie")
            speedup_str = f"{max(speedup, 1/speedup):.2f}x"
        else:
            winner = "N/A"
            speedup_str = "N/A"
        
        print(f"| {name:18} | {yf_time:13} | {poly_time:10} | {winner:8} | {speedup_str:7} |")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
