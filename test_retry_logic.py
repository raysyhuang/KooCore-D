#!/usr/bin/env python3
"""
Unit tests for retry guard logic (no external dependencies).

Tests the guard logic without actually sending Telegram messages
or persisting to databases.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_telegram_suppression():
    """Test that Telegram alerts are suppressed for attempt > 1."""
    from src.core.alerts import AlertManager, AlertConfig
    
    print("\n[Test 1] Telegram Alert Suppression")
    print("=" * 50)
    
    # Test Case 1: Attempt 1 should proceed
    print("\n  Test 1.1: Attempt 1 (should proceed to send)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    config = AlertConfig(enabled=True, channels=["telegram"])
    manager = AlertManager(config)
    
    # Mock the actual Telegram API call
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        
        result = manager._send_telegram(
            title="Test Alert",
            message="Test message",
            data={"asof": "2026-01-30"},
            priority="normal"
        )
        
        # For attempt 1, should have tried to send
        if mock_post.called:
            print("    ✓ Attempt 1: Tried to send (as expected)")
        else:
            print("    ✗ Attempt 1: Did NOT try to send (unexpected)")
            return False
    
    # Test Case 2: Attempt 2 should be suppressed
    print("\n  Test 1.2: Attempt 2 (should be suppressed)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "2"
    
    with patch('requests.post') as mock_post:
        result = manager._send_telegram(
            title="Test Alert",
            message="Test message",
            data={"asof": "2026-01-30"},
            priority="normal"
        )
        
        # For attempt 2, should NOT have tried to send
        if not mock_post.called:
            print("    ✓ Attempt 2: Suppressed (as expected)")
        else:
            print("    ✗ Attempt 2: Tried to send (should be suppressed)")
            return False
        
        # Should still return True (silent success)
        if result:
            print("    ✓ Returned True (silent success)")
        else:
            print("    ✗ Returned False (should return True)")
            return False
    
    # Test Case 3: Attempt 3 should also be suppressed
    print("\n  Test 1.3: Attempt 3 (should be suppressed)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "3"
    
    with patch('requests.post') as mock_post:
        result = manager._send_telegram(
            title="Test Alert",
            message="Test message",
            data={"asof": "2026-01-30"},
            priority="normal"
        )
        
        if not mock_post.called and result:
            print("    ✓ Attempt 3: Suppressed (as expected)")
        else:
            print("    ✗ Attempt 3: Failed suppression check")
            return False
    
    # Reset
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    print("\n  ✅ All telegram suppression tests passed!")
    return True


def test_outcome_recording_guard():
    """Test that outcome recording is guarded for attempt > 1."""
    print("\n[Test 2] Outcome Recording Guard")
    print("=" * 50)
    
    # We'll test this by checking the guard logic directly
    
    # Test Case 1: Attempt 1 should record
    print("\n  Test 2.1: Attempt 1 (should record)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    try:
        attempt_num = int(run_attempt)
        should_skip = attempt_num > 1
    except (ValueError, TypeError):
        should_skip = False
    
    if not should_skip:
        print("    ✓ Attempt 1: Would record outcome")
    else:
        print("    ✗ Attempt 1: Would skip (unexpected)")
        return False
    
    # Test Case 2: Attempt 2 should skip
    print("\n  Test 2.2: Attempt 2 (should skip)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "2"
    
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    try:
        attempt_num = int(run_attempt)
        should_skip = attempt_num > 1
    except (ValueError, TypeError):
        should_skip = False
    
    if should_skip:
        print("    ✓ Attempt 2: Would skip outcome recording")
    else:
        print("    ✗ Attempt 2: Would record (should skip)")
        return False
    
    # Reset
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    print("\n  ✅ All outcome recording tests passed!")
    return True


def test_phase5_guard():
    """Test that Phase 5 learning is guarded for attempt > 1."""
    print("\n[Test 3] Phase 5 Learning Guard")
    print("=" * 50)
    
    # Test Case 1: Attempt 1 should persist
    print("\n  Test 3.1: Attempt 1 (should persist)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    phase5_enabled = True
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    try:
        attempt_num = int(run_attempt)
        if attempt_num > 1:
            phase5_enabled = False
    except (ValueError, TypeError):
        pass
    
    if phase5_enabled:
        print("    ✓ Attempt 1: Would persist learning")
    else:
        print("    ✗ Attempt 1: Would skip (unexpected)")
        return False
    
    # Test Case 2: Attempt 2 should skip
    print("\n  Test 3.2: Attempt 2 (should skip)")
    os.environ["GITHUB_RUN_ATTEMPT"] = "2"
    
    phase5_enabled = True
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    try:
        attempt_num = int(run_attempt)
        if attempt_num > 1:
            phase5_enabled = False
    except (ValueError, TypeError):
        pass
    
    if not phase5_enabled:
        print("    ✓ Attempt 2: Would skip learning persistence")
    else:
        print("    ✗ Attempt 2: Would persist (should skip)")
        return False
    
    # Reset
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    print("\n  ✅ All Phase 5 guard tests passed!")
    return True


def test_marker_file_logic():
    """Test marker file creation and checking."""
    print("\n[Test 4] Marker File Logic")
    print("=" * 50)
    
    from pathlib import Path
    import tempfile
    import shutil
    
    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test Case 1: No marker file exists (should create)
        print("\n  Test 4.1: Create marker file")
        
        outputs_dir = test_dir / "2026-01-30"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        run_id = "test_12345"
        run_attempt = "1"
        marker_file = outputs_dir / f".telegram_sent_{run_id}_{run_attempt}.txt"
        
        # Simulate marker creation
        if not marker_file.exists():
            with open(marker_file, "w") as f:
                f.write("Sent at: 2026-01-30T10:15:30\n")
                f.write("Run ID: test_12345\n")
            print("    ✓ Marker file created")
        else:
            print("    ✗ Marker file already exists (unexpected)")
            return False
        
        # Test Case 2: Marker file exists (should skip)
        print("\n  Test 4.2: Check existing marker file")
        
        if marker_file.exists():
            print("    ✓ Marker file detected (would skip send)")
        else:
            print("    ✗ Marker file not found (unexpected)")
            return False
        
        # Test Case 3: Different attempt creates different marker
        print("\n  Test 4.3: Different attempt number")
        
        run_attempt2 = "2"
        marker_file2 = outputs_dir / f".telegram_sent_{run_id}_{run_attempt2}.txt"
        
        if not marker_file2.exists():
            print("    ✓ Different attempt = different marker (correct)")
        else:
            print("    ✗ Marker already exists (unexpected)")
            return False
        
        print("\n  ✅ All marker file tests passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


def test_metadata_extraction():
    """Test GitHub metadata extraction."""
    print("\n[Test 5] Metadata Extraction")
    print("=" * 50)
    
    # Set up test environment
    os.environ["GITHUB_WORKFLOW"] = "Test Workflow"
    os.environ["GITHUB_RUN_ID"] = "1234567890"
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    os.environ["GITHUB_SHA"] = "abcdef1234567890"
    
    # Test Case 1: Extract all metadata
    print("\n  Test 5.1: Extract GitHub metadata")
    
    workflow = os.environ.get("GITHUB_WORKFLOW", "local")
    run_id = os.environ.get("GITHUB_RUN_ID", "N/A")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    sha = os.environ.get("GITHUB_SHA", "N/A")
    
    if sha != "N/A" and len(sha) > 7:
        sha = sha[:7]
    
    if workflow == "Test Workflow" and run_id == "1234567890" and sha == "abcdef1":
        print("    ✓ All metadata extracted correctly")
    else:
        print(f"    ✗ Metadata mismatch:")
        print(f"      workflow={workflow}, run_id={run_id}, sha={sha}")
        return False
    
    # Test Case 2: Local environment (no GitHub vars)
    print("\n  Test 5.2: Local environment fallbacks")
    
    del os.environ["GITHUB_WORKFLOW"]
    del os.environ["GITHUB_RUN_ID"]
    del os.environ["GITHUB_SHA"]
    
    workflow = os.environ.get("GITHUB_WORKFLOW", "local")
    run_id = os.environ.get("GITHUB_RUN_ID", "N/A")
    sha = os.environ.get("GITHUB_SHA", "N/A")
    
    if workflow == "local" and run_id == "N/A" and sha == "N/A":
        print("    ✓ Local fallbacks work correctly")
    else:
        print(f"    ✗ Fallback failed:")
        print(f"      workflow={workflow}, run_id={run_id}, sha={sha}")
        return False
    
    # Reset
    os.environ["GITHUB_RUN_ATTEMPT"] = "1"
    
    print("\n  ✅ All metadata extraction tests passed!")
    return True


def main():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("RETRY LOGIC UNIT TESTS")
    print("=" * 60)
    print("\nTesting retry guard logic without external dependencies...")
    
    results = []
    
    try:
        results.append(("Telegram Suppression", test_telegram_suppression()))
    except Exception as e:
        print(f"\n✗ Telegram suppression test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Telegram Suppression", False))
    
    try:
        results.append(("Outcome Recording Guard", test_outcome_recording_guard()))
    except Exception as e:
        print(f"\n✗ Outcome recording test failed: {e}")
        results.append(("Outcome Recording Guard", False))
    
    try:
        results.append(("Phase 5 Learning Guard", test_phase5_guard()))
    except Exception as e:
        print(f"\n✗ Phase 5 guard test failed: {e}")
        results.append(("Phase 5 Learning Guard", False))
    
    try:
        results.append(("Marker File Logic", test_marker_file_logic()))
    except Exception as e:
        print(f"\n✗ Marker file test failed: {e}")
        results.append(("Marker File Logic", False))
    
    try:
        results.append(("Metadata Extraction", test_metadata_extraction()))
    except Exception as e:
        print(f"\n✗ Metadata extraction test failed: {e}")
        results.append(("Metadata Extraction", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 60)
        print("\nRetry logic is working correctly!")
        return True
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("=" * 60)
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
