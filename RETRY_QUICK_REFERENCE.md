# Retry Guard - Quick Reference Card

**One-page guide for engineers adding new operations**

---

## Invariant R1: Retry Purity

```
Retries may read external state but must not mutate it.
```

*Reference this as "R1" in code reviews and discussions*

---

## Decision Tree

```
┌─────────────────────────────────────┐
│ Adding a new operation?             │
└──────────────┬──────────────────────┘
               │
               ▼
       Does it change
       external state?
               │
       ┌───────┴───────┐
       │               │
      YES             NO
       │               │
       ▼               ▼
   ADD GUARD      NO GUARD
                  NEEDED
```

---

## What Needs Guarding?

### ❌ MUST Guard (Side Effects)
- Sending alerts (Telegram, Slack, email, Discord)
- Writing to databases (INSERT, UPDATE, DELETE)
- Uploading files (S3, GCS, any cloud storage)
- API calls that mutate (POST, PUT, DELETE, PATCH)
- Creating notifications (webhooks, SNS, etc.)
- Trade order submission
- Analytics tracking (if it increments counters)

### ✅ Safe Without Guard (Pure Reads)
- Fetching market data (GET requests)
- Database queries (SELECT)
- Downloading files (read-only cloud access)
- Computing scores (pure functions)
- Data transformations (filtering, mapping)
- Local file writes (outputs/ directory)
- Logging to stdout

### ⚠️ Edge Cases (Guard If Unsure)
- GET requests that increment counters
- Read operations with rate limiting
- Audit logs that trigger alerts
- Cache invalidations

**Rule of thumb:** If unsure, guard it.

---

## Implementation Pattern

### Standard Guard

```python
from src.core.retry_guard import is_retry_attempt, log_retry_suppression

def my_operation_with_side_effect():
    # Retries re-run computation but MUST NOT emit side effects
    if is_retry_attempt():
        log_retry_suppression("operation description", **metadata)
        return  # or return True, or skip the block
    
    # ... actual side effect logic here ...
```

### Where to Place Guard

```python
# ✅ CORRECT: Guard at persistence boundary
def main_workflow():
    data = compute_something()  # No guard needed (pure)
    
    if is_retry_attempt():
        return
    
    send_results(data)  # Guarded above

# ❌ WRONG: Guard inside utility function
def send_results(data):
    if is_retry_attempt():  # Too deep!
        return
    # ... send logic
```

**Principle:** Guard meaning, not code paths.

---

## Testing Checklist

When adding a new guarded operation:

- [ ] Add guard using `is_retry_attempt()`
- [ ] Call `log_retry_suppression()` with context
- [ ] Add test case to `test_retry_logic.py`
- [ ] Update `verify_retry_guards.py` if needed
- [ ] Test locally with `GITHUB_RUN_ATTEMPT=2`
- [ ] Document in `PIPELINE_INVARIANTS.md` if significant

---

## Local Testing

```bash
# Test normal operation (attempt 1)
unset GITHUB_RUN_ATTEMPT  # or export GITHUB_RUN_ATTEMPT=1
python main.py all

# Test retry suppression (attempt 2)
export GITHUB_RUN_ATTEMPT=2
python main.py all
# Should see: "Suppressing <operation> (retry attempt=2)"

# Run verification
python verify_retry_guards.py
# Expected: ✅ ALL GUARDS VERIFIED

# Run unit tests
python test_retry_logic.py
# Expected: ✅ ALL TESTS PASSED
```

---

## Common Mistakes

### ❌ Guarding Too Deep
```python
def utility_send_alert():
    if is_retry_attempt(): return
    # ... send
```
**Problem:** Future callers bypass guard

**Fix:** Guard at top-level workflow

### ❌ Inconsistent Checks
```python
# Different styles in different places
attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
if attempt > 1: ...

# vs
if os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1": ...
```
**Problem:** Logic drift, harder to maintain

**Fix:** Always use `is_retry_attempt()`

### ❌ Forgetting Metadata
```python
log_retry_suppression("Telegram alert")
```
**Problem:** Logs don't have context

**Fix:** Add metadata
```python
log_retry_suppression("Telegram alert", run_id=run_id, title=title)
```

---

## Philosophy

### Retries ≠ Failures

Retries are **health signals** that indicate:
- API instability (partner issues)
- Infrastructure flakiness (cloud provider)
- External dependency degradation (vendor SLA)

**Not bugs in your code.**

### Clean Data Beats Fancy Models

If learning data is polluted by retry duplicates:
- Hit rates are inflated
- Models learn false patterns
- Production performance degrades

**Guard correctness > computational efficiency**

---

## Examples from Codebase

### Example 1: Telegram Alerts
**File:** `src/core/alerts.py`
```python
def _send_telegram(title, message, data, priority):
    from src.core.retry_guard import is_retry_attempt, log_retry_suppression
    
    # Retries re-run computation but MUST NOT emit side effects
    if is_retry_attempt():
        log_retry_suppression("Telegram alert", run_id=run_id, title=title)
        return True
    
    # ... actual send logic ...
```

### Example 2: Outcome Persistence
**File:** `src/features/positions/tracker.py`
```python
def _record_outcome(pos):
    from src.core.retry_guard import is_retry_attempt, log_retry_suppression
    
    # Retries re-run computation but MUST NOT emit side effects
    if is_retry_attempt():
        log_retry_suppression("outcome persistence", ticker=pos.ticker)
        return
    
    # ... database write logic ...
```

### Example 3: Learning Data
**File:** `src/commands/all.py`
```python
if phase5_enabled:
    from src.core.retry_guard import is_retry_attempt, log_retry_suppression
    
    # Retries re-run computation but MUST NOT emit side effects
    if is_retry_attempt():
        log_retry_suppression("Phase 5 outcome persistence")
        phase5_enabled = False
    
    if phase5_enabled:
        persist_learning(records)
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| `PIPELINE_INVARIANTS.md` | Full architecture guide |
| `RETRY_HANDLING_SUMMARY.md` | Detailed implementation |
| `RETRY_QUICK_REFERENCE.md` | This card (one-page) |
| `src/core/retry_guard.py` | Implementation |

---

## Need Help?

1. **Read:** `PIPELINE_INVARIANTS.md` (10 min read)
2. **Ask:** "Does this operation change external state?"
3. **Test:** `export GITHUB_RUN_ATTEMPT=2` and verify suppression
4. **Verify:** `python verify_retry_guards.py`

**When in doubt, guard it.**

---

**Last Updated:** 2026-01-30  
**Status:** Production-Ready  
**Principle:** Retries are compute-only. No mutations. Ever.
