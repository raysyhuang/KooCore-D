# Retry Handling Flow Diagram

## 📊 Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Workflow Starts                       │
│                  (Auto Run, Phase 5, etc.)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Check GITHUB_RUN_ATTEMPT│
                └────────┬───────────────┘
                         │
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   attempt == 1                      attempt > 1
   [FIRST TRY]                       [RETRY]
        │                                 │
        │                                 │
        ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐
│ Full Operation   │              │ Partial Operation│
├──────────────────┤              ├──────────────────┤
│                  │              │                  │
│ ✅ Run Scan      │              │ ✅ Run Scan      │
│ ✅ Generate Data │              │ ✅ Generate Data │
│ ✅ Create Report │              │ ✅ Create Report │
│                  │              │                  │
│ ✅ Send Telegram │              │ ❌ SKIP Telegram │
│    Alert         │              │    Alert         │
│                  │              │    (suppressed)  │
│ ✅ Record        │              │ ❌ SKIP Record   │
│    Outcomes      │              │    Outcomes      │
│                  │              │    (guarded)     │
│ ✅ Persist       │              │ ❌ SKIP Persist  │
│    Learning      │              │    Learning      │
│                  │              │    (guarded)     │
│ ✅ Create Marker │              │                  │
│    File          │              │                  │
└──────────────────┘              └──────────────────┘
        │                                 │
        │                                 │
        ▼                                 ▼
┌──────────────────┐              ┌──────────────────┐
│ User receives:   │              │ User receives:   │
│ • 1 Telegram msg │              │ • Nothing new    │
│ • Full metadata  │              │ • Logs only      │
│ • Clean outcomes │              │                  │
└──────────────────┘              └──────────────────┘
```

## 🔐 Guard Locations

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUARD LAYER 1                           │
│                    Telegram Alert Sending                       │
│                   (src/core/alerts.py)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  def _send_telegram():                                          │
│      run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")   │
│      attempt_num = int(run_attempt)                             │
│                                                                 │
│      if attempt_num > 1:  🛡️                                   │
│          logger.info("Suppressing Telegram alert...")           │
│          return True  # Silent success                          │
│                                                                 │
│      # ... rest of send logic ...                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                         GUARD LAYER 2                           │
│                     Phase 5 Learning                            │
│                   (src/commands/all.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  def cmd_all():                                                 │
│      # ... scan logic ...                                       │
│                                                                 │
│      if phase5_enabled:                                         │
│          run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")│
│          attempt_num = int(run_attempt)                         │
│                                                                 │
│          if attempt_num > 1:  🛡️                               │
│              logger.info("Skipping Phase 5 persistence...")     │
│              phase5_enabled = False                             │
│                                                                 │
│          if phase5_enabled:                                     │
│              persist_learning(records)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                         GUARD LAYER 3                           │
│                    Position Outcomes                            │
│              (src/features/positions/tracker.py)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  def _record_outcome(pos):                                      │
│      run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")   │
│      attempt_num = int(run_attempt)                             │
│                                                                 │
│      if attempt_num > 1:  🛡️                                   │
│          logger.info(f"Skipping outcome for {pos.ticker}...")   │
│          return                                                 │
│                                                                 │
│      # ... outcome recording logic ...                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📈 Outcome Impact

### Before Guards (with retries)

```
Run 1 (attempt 1):
  ✅ Alert sent
  ✅ AAPL outcome recorded: hit_7pct=True

Run 1 (attempt 2 - RETRY):
  ✅ Alert sent ← DUPLICATE! ❌
  ✅ AAPL outcome recorded: hit_7pct=True ← DUPLICATE! ❌

Database:
  - AAPL, 2026-01-30, hit_7pct=True  ← entry 1
  - AAPL, 2026-01-30, hit_7pct=True  ← entry 2 (DUPLICATE)

Result: Inflated hit rate! 🚨
```

### After Guards (with retries)

```
Run 1 (attempt 1):
  ✅ Alert sent
  ✅ AAPL outcome recorded: hit_7pct=True

Run 1 (attempt 2 - RETRY):
  ❌ Alert suppressed (guarded)
  ❌ AAPL outcome skipped (guarded)

Database:
  - AAPL, 2026-01-30, hit_7pct=True  ← single entry ✅

Result: Accurate hit rate! ✅
```

## 🎯 Message Examples

### Attempt 1 Message
```
📈 Scan Complete: 2026-01-30 (Swing)

━━━━━━━━━━━━━━━━━━━━━━━
🔍 Run Metadata:
  workflow: Auto Run (Daily)
  run_id: 1234567890
  attempt: 1                    ← FIRST ATTEMPT
  sha: abc1234
  run_started_utc: 2026-01-30T23:15:45
  asof: 2026-01-30
  report_path: outputs/2026-01-30/report_2026-01-30.html
━━━━━━━━━━━━━━━━━━━━━━━

===================================
SCAN COMPLETE - 2026-01-30
===================================
Regime: Bull
...
```

### Attempt 2 (No Message - Suppressed)
```
[Logs only]
INFO: Suppressing Telegram alert (retry attempt=2, run_id=1234567890)
INFO: Skipping outcome persistence for AAPL (retry attempt=2)
INFO: Skipping Phase 5 outcome persistence for retry attempt=2
```

## 🧪 Testing Commands

```bash
# ─────────────────────────────────────────────────────────
# Test 1: Normal run (attempt 1)
# ─────────────────────────────────────────────────────────
unset GITHUB_RUN_ATTEMPT  # or export GITHUB_RUN_ATTEMPT=1
python main.py all --config config/default.yaml

Expected:
  ✅ Telegram alert sent
  ✅ Outcomes recorded
  ✅ "Recorded outcome for AAPL" in logs


# ─────────────────────────────────────────────────────────
# Test 2: Retry simulation (attempt 2)
# ─────────────────────────────────────────────────────────
export GITHUB_RUN_ATTEMPT=2
python main.py all --config config/default.yaml

Expected:
  ❌ No Telegram alert
  ❌ No outcomes recorded
  ✅ "Suppressing Telegram alert" in logs
  ✅ "Skipping outcome persistence" in logs


# ─────────────────────────────────────────────────────────
# Test 3: Full verification
# ─────────────────────────────────────────────────────────
python verify_retry_guards.py

Expected:
  ✅ ALL GUARDS VERIFIED


# ─────────────────────────────────────────────────────────
# Test 4: Full integration test
# ─────────────────────────────────────────────────────────
python test_telegram_tracking.py

Expected:
  ✅ Attempt 1: Alert sent
  ✅ Duplicate: Skipped
  ✅ Attempt 2: Suppressed
  ✅ New run_id: Alert sent
```

## 📋 Checklist for Production

```
Before Retry:
  ❌ Multiple alerts per logical run
  ❌ Duplicate outcomes in database
  ❌ Inflated statistics
  ❌ Confusion about which run sent what

After Retry Guards:
  ✅ Single alert per logical run (attempt 1 only)
  ✅ Clean outcome data (no duplicates)
  ✅ Accurate statistics
  ✅ Full metadata in every message

Implementation:
  ✅ 3 guard layers implemented
  ✅ All guards verified (automated)
  ✅ No linter errors
  ✅ Documentation complete
  ✅ Test scripts created
  ✅ Ready to deploy
```

---

**Visual Guide Last Updated:** 2026-01-30  
**Status:** Complete  
**All Guards:** ✅ Verified & Tested
