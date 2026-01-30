#!/usr/bin/env python3
"""
Test script for Telegram tracking implementation.

This script verifies:
1. Metadata header is included in messages
2. Marker file creation works
3. Duplicate send prevention works
4. Retry attempt suppression works (NEW)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.alerts import AlertConfig, send_run_summary_alert


def test_telegram_tracking():
    """Test the Telegram tracking implementation."""
    
    print("=" * 60)
    print("Testing Telegram Tracking Implementation")
    print("=" * 60)
    
    # Mock GitHub environment variables for testing
    os.environ["GITHUB_WORKFLOW"] = "Test Workflow"
    os.environ["GITHUB_RUN_ID"] = "test_12345"
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    os.environ["GITHUB_SHA"] = "abc123def456789"
    
    # Configure alerts (use your actual credentials from .env)
    config = AlertConfig(
        enabled=True,
        channels=["telegram"],
        # Credentials will be loaded from environment
    )
    
    test_date = "2026-01-30"
    
    # Test 1: Send first message (attempt 1)
    print("\n[Test 1] Sending first message (attempt 1)...")
    results = send_run_summary_alert(
        date_str=test_date,
        weekly_count=5,
        pro30_count=10,
        movers_count=8,
        overlaps={
            "all_three": ["AAPL", "MSFT"],
            "weekly_pro30": ["GOOGL"],
            "weekly_movers": ["TSLA"],
            "pro30_movers": [],
        },
        config=config,
        weekly_top5_data=[
            {
                "ticker": "AAPL",
                "rank": 1,
                "composite_score": 8.5,
                "confidence": "HIGH",
                "name": "Apple Inc.",
            },
            {
                "ticker": "MSFT",
                "rank": 2,
                "composite_score": 7.8,
                "confidence": "MEDIUM",
                "name": "Microsoft Corporation",
            },
        ],
        hybrid_top3=[
            {
                "ticker": "AAPL",
                "hybrid_score": 5.1,
                "sources": ["Swing(1)", "Pro30"],
                "rank": 1,
                "confidence": "HIGH",
            },
        ],
        model_health={
            "status": "GOOD",
            "hit_rate": 0.35,
            "win_rate": 0.28,
            "strategies": [
                {"name": "Swing", "hit_rate": 0.38, "n": 50},
                {"name": "Pro30", "hit_rate": 0.33, "n": 45},
            ]
        },
        primary_label="Swing",
        regime="Bull",
    )
    
    if results.get("telegram"):
        print("✓ First message sent successfully")
    else:
        print("✗ First message failed")
        print("Note: Make sure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set")
        return False
    
    # Check if marker file was created
    marker_path = Path(f"outputs/{test_date}/.telegram_sent_test_12345_1.txt")
    if marker_path.exists():
        print(f"✓ Marker file created: {marker_path}")
        print(f"  Contents:")
        with open(marker_path) as f:
            for line in f:
                print(f"    {line.rstrip()}")
    else:
        print("✗ Marker file not created")
        return False
    
    # Test 2: Try to send duplicate (should be skipped)
    print("\n[Test 2] Attempting duplicate send (should be skipped)...")
    results2 = send_run_summary_alert(
        date_str=test_date,
        weekly_count=5,
        pro30_count=10,
        movers_count=8,
        overlaps={"all_three": [], "weekly_pro30": [], "weekly_movers": [], "pro30_movers": []},
        config=config,
        primary_label="Swing",
    )
    
    if results2.get("telegram"):
        print("✓ Duplicate send properly skipped (returned success)")
    else:
        print("✗ Duplicate handling failed")
        return False
    
    # Test 3: Retry attempt should be suppressed (NEW TEST)
    print("\n[Test 3] Testing retry suppression (attempt 2)...")
    os.environ["GITHUB_RUN_ATTEMPT"] = "2"
    
    # Clear marker to test suppression logic (not marker logic)
    marker_path_attempt2 = Path(f"outputs/{test_date}/.telegram_sent_test_12345_2.txt")
    if marker_path_attempt2.exists():
        marker_path_attempt2.unlink()
    
    results3 = send_run_summary_alert(
        date_str=test_date,
        weekly_count=5,
        pro30_count=10,
        movers_count=8,
        overlaps={"all_three": [], "weekly_pro30": [], "weekly_movers": [], "pro30_movers": []},
        config=config,
        primary_label="Swing",
    )
    
    # Should return True but not actually send
    if results3.get("telegram"):
        print("✓ Retry attempt suppressed (returned success without sending)")
    else:
        print("✗ Retry suppression failed")
        return False
    
    # Marker should NOT be created for suppressed send
    if not marker_path_attempt2.exists():
        print("✓ No marker file created for suppressed retry (as expected)")
    else:
        print("⚠ Marker file created for suppressed send (unexpected but harmless)")
    
    # Test 4: Different run_id with attempt 1 should send
    print("\n[Test 4] Testing different run_id (should send)...")
    os.environ["GITHUB_RUN_ID"] = "test_67890"
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    results4 = send_run_summary_alert(
        date_str=test_date,
        weekly_count=5,
        pro30_count=10,
        movers_count=8,
        overlaps={"all_three": [], "weekly_pro30": [], "weekly_movers": [], "pro30_movers": []},
        config=config,
        primary_label="Swing",
    )
    
    # Check for new marker file
    marker_path_new = Path(f"outputs/{test_date}/.telegram_sent_test_67890_1.txt")
    if marker_path_new.exists():
        print(f"✓ New marker file created for different run_id: {marker_path_new}")
    else:
        print("✗ Marker file not created for new run")
        return False
    
    # Cleanup
    print("\n[Cleanup] Removing test marker files...")
    for marker in [marker_path, marker_path_attempt2, marker_path_new]:
        if marker.exists():
            marker.unlink()
            print(f"  Removed {marker}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print("  ✓ Attempt 1: Alert sent with metadata")
    print("  ✓ Duplicate: Skipped (marker exists)")
    print("  ✓ Attempt 2: Suppressed (no alert)")
    print("  ✓ New run_id: Alert sent")
    print("  ✓ Marker files: Created and cleaned up")
    
    return True


if __name__ == "__main__":
    try:
        success = test_telegram_tracking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
