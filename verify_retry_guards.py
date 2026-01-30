#!/usr/bin/env python3
"""
Verification script to ensure all retry guards are properly implemented.

This script checks:
1. Alert suppression in _send_telegram()
2. Phase 5 learning guard in cmd_all()
3. Outcome recording guard in _record_outcome()
"""

import sys
from pathlib import Path
import re


def check_file_for_guard(filepath: Path, guard_pattern: str, context: str) -> bool:
    """Check if a file contains the expected guard pattern."""
    try:
        content = filepath.read_text()
        
        if guard_pattern in content:
            print(f"✓ {context}: Guard found")
            return True
        else:
            print(f"✗ {context}: Guard MISSING")
            return False
    except Exception as e:
        print(f"✗ {context}: Error reading file: {e}")
        return False


def verify_guards():
    """Verify all retry guards are in place."""
    
    print("=" * 60)
    print("Retry Guard Verification")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    all_good = True
    
    # Check 1: Alert suppression in alerts.py
    print("\n[1/3] Checking Telegram alert suppression...")
    alerts_file = project_root / "src" / "core" / "alerts.py"
    
    if not alerts_file.exists():
        print(f"✗ File not found: {alerts_file}")
        all_good = False
    else:
        # Look for the suppression logic (now centralized)
        content = alerts_file.read_text()
        
        checks = [
            ("from src.core.retry_guard import is_retry_attempt", "Imports retry guard"),
            ("if is_retry_attempt():", "Checks for retry attempt"),
            ("log_retry_suppression", "Logs suppression"),
        ]
        
        for pattern, desc in checks:
            if pattern in content:
                print(f"  ✓ {desc}")
            else:
                print(f"  ✗ {desc} - MISSING")
                all_good = False
    
    # Check 2: Phase 5 learning guard in all.py
    print("\n[2/3] Checking Phase 5 learning guard...")
    all_file = project_root / "src" / "commands" / "all.py"
    
    if not all_file.exists():
        print(f"✗ File not found: {all_file}")
        all_good = False
    else:
        content = all_file.read_text()
        
        phase5_section = re.search(
            r'phase5_enabled.*?persist_learning',
            content,
            re.DOTALL
        )
        
        if phase5_section:
            section_text = phase5_section.group(0)
            
            if "is_retry_attempt()" in section_text and "log_retry_suppression" in section_text:
                print(f"  ✓ Guard found before persist_learning (centralized)")
            elif "GITHUB_RUN_ATTEMPT" in section_text and "attempt_num > 1" in section_text:
                print(f"  ✓ Guard found before persist_learning (old style)")
            else:
                print(f"  ✗ Guard NOT found before persist_learning")
                all_good = False
        else:
            print(f"  ⚠ Could not locate Phase 5 section (manual check needed)")
    
    # Check 3: Outcome recording guard in tracker.py
    print("\n[3/3] Checking outcome recording guard...")
    tracker_file = project_root / "src" / "features" / "positions" / "tracker.py"
    
    if not tracker_file.exists():
        print(f"✗ File not found: {tracker_file}")
        all_good = False
    else:
        content = tracker_file.read_text()
        
        # Look for _record_outcome function
        record_outcome_match = re.search(
            r'def _record_outcome\(self, pos.*?\n(?:.*?\n){0,30}',
            content,
            re.DOTALL
        )
        
        if record_outcome_match:
            func_start = record_outcome_match.group(0)
            
            if "is_retry_attempt()" in func_start and "log_retry_suppression" in func_start:
                print(f"  ✓ Guard found in _record_outcome() (centralized)")
            elif "GITHUB_RUN_ATTEMPT" in func_start and "attempt_num > 1" in func_start:
                print(f"  ✓ Guard found in _record_outcome() (old style)")
            else:
                print(f"  ✗ Guard NOT found in _record_outcome()")
                all_good = False
        else:
            print(f"  ⚠ Could not locate _record_outcome() (manual check needed)")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL GUARDS VERIFIED")
        print("=" * 60)
        print("\nRetry handling is properly implemented:")
        print("  ✓ Telegram alerts suppressed for attempt > 1")
        print("  ✓ Phase 5 learning guarded")
        print("  ✓ Outcome recording guarded")
        print("\nYou're ready to deploy!")
        return True
    else:
        print("⚠️  SOME GUARDS MISSING OR INCOMPLETE")
        print("=" * 60)
        print("\nPlease review the failures above and ensure all guards are in place.")
        return False


def show_guard_locations():
    """Show where to find each guard in the codebase."""
    print("\n" + "=" * 60)
    print("Guard Locations Reference")
    print("=" * 60)
    
    guards = [
        {
            "name": "Telegram Alert Suppression",
            "file": "src/core/alerts.py",
            "function": "_send_telegram()",
            "line_hint": "After getting GITHUB_RUN_ATTEMPT, before checking marker file",
            "pattern": "if attempt_num > 1: ... return True",
        },
        {
            "name": "Phase 5 Learning Guard",
            "file": "src/commands/all.py",
            "function": "cmd_all()",
            "line_hint": "Before if phase5_enabled: block",
            "pattern": "if attempt_num > 1: ... phase5_enabled = False",
        },
        {
            "name": "Outcome Recording Guard",
            "file": "src/features/positions/tracker.py",
            "function": "_record_outcome()",
            "line_hint": "At start of function, before validating entry_price",
            "pattern": "if attempt_num > 1: ... return",
        },
    ]
    
    for i, guard in enumerate(guards, 1):
        print(f"\n{i}. {guard['name']}")
        print(f"   File: {guard['file']}")
        print(f"   Function: {guard['function']}")
        print(f"   Location: {guard['line_hint']}")
        print(f"   Pattern: {guard['pattern']}")


if __name__ == "__main__":
    print("\nRetry Guard Verification Script")
    print("This script checks that all retry guards are properly implemented.\n")
    
    success = verify_guards()
    
    if not success:
        show_guard_locations()
        print("\n" + "=" * 60)
        print("Need help? See RETRY_HANDLING_SUMMARY.md for details.")
        print("=" * 60)
    
    sys.exit(0 if success else 1)
