# Pipeline Invariants

## Core Principle: Retry Semantics

```
Retries re-run computation but MUST NOT emit side effects.
```

### Invariant R1: Retry Purity

```
Retries may read external state but must not mutate it.
```

This is the foundational invariant that governs all persistence boundaries in the pipeline.

**Why "R1"?** This naming convention makes the invariant:
- **Referenceable** in code reviews ("Does this violate R1?")
- **Extensible** for future invariants (R2: Ordering, R3: Monotonicity, etc.)
- **Formal** without being heavyweight

**What this means:**
- ✅ **Read operations** are safe to retry (market data fetches, database queries)
- ❌ **Write operations** must be guarded (alerts, database writes, API calls)
- ✅ **Pure computation** is safe to retry (scoring, transformations)
- ❌ **External communication** must be guarded (webhooks, emails)

---

## What Are Side Effects?

Side effects are operations that **change external state** or **communicate with external systems**:

### ❌ Side Effects (Must Be Guarded)
- Sending Telegram/Slack/Discord/Email alerts
- Writing to outcome database
- Persisting learning data
- Uploading to S3/cloud storage
- Making external API calls that mutate state
- Creating webhook notifications

### ✅ Computation (Safe to Retry)
- Fetching market data (read-only)
- Running analysis pipelines
- Computing scores and rankings
- Generating reports (local files)
- Reading from databases
- Data transformations

---

## The Retry Guard Pattern

### Standard Implementation

Every side effect must be guarded with:

```python
from src.core.retry_guard import is_retry_attempt, log_retry_suppression

# Invariant R1: Retry Purity - no mutations on retry
if is_retry_attempt():
    log_retry_suppression("description of side effect", **metadata)
    return  # or skip the block
```

**Optional:** Reference R1 in comments for clarity in code reviews.

### Why This Works

1. **Centralized Logic:** Single source of truth (`src/core/retry_guard.py`)
2. **Consistent Behavior:** Same check everywhere
3. **Audit Trail:** All suppressions are logged
4. **Silent Success:** Returns True/continues to avoid breaking workflows

---

## Current Protected Boundaries

### 1. Alert Notifications
**Location:** `src/core/alerts.py` - `_send_telegram()`

```python
# Retries re-run computation but MUST NOT emit side effects
if is_retry_attempt():
    log_retry_suppression("Telegram alert", run_id=run_id, title=title)
    return True
```

**Why:** Prevents alert spam when workflows retry.

### 2. Outcome Persistence
**Location:** `src/features/positions/tracker.py` - `_record_outcome()`

```python
# Retries re-run computation but MUST NOT emit side effects
if is_retry_attempt():
    log_retry_suppression("outcome persistence", ticker=pos.ticker)
    return
```

**Why:** Prevents inflated hit rates from duplicate outcomes.

### 3. Learning Data
**Location:** `src/commands/all.py` - Phase 5 learning block

```python
# Retries re-run computation but MUST NOT emit side effects
if is_retry_attempt():
    log_retry_suppression("Phase 5 outcome persistence")
    phase5_enabled = False
```

**Why:** Keeps learning data clean and accurate.

---

## Verification

### Automated Checks
```bash
# Verify all guards are in place
python verify_retry_guards.py

# Run unit tests
python test_retry_logic.py
```

### Manual Inspection
```bash
# Check for retry suppressions in logs
grep "Suppressing.*retry attempt" outputs/logs/*.log

# Verify no duplicate outcomes
python -c "
from src.core.outcome_db import get_outcome_db
db = get_outcome_db()
df = db.get_training_data()
dupes = df[df.duplicated(['ticker', 'pick_date'], keep=False)]
print(f'Duplicate outcomes: {len(dupes)}')
"
```

---

## When to Add Guards

Ask yourself: **"Does this operation change external state?"**

### Examples:

**✅ Needs Guard:**
- Sending a webhook
- Writing to database
- Uploading a file
- Sending an email
- Creating a trade order

**❌ No Guard Needed:**
- Reading market data (GET requests, SELECT queries)
- Computing a score (pure functions)
- Filtering a dataframe (data transformations)
- Generating a local report (writing to outputs/)
- Logging to stdout (observability)

**⚠️ Edge Case: Hidden Side Effects**

Some "read" operations have hidden side effects:
- Analytics tracking APIs (increment usage counters)
- Rate-limited endpoints (consume quota even on GET)
- Audit systems (access logs that trigger alerts)

**Rule of thumb:** If a GET request increments a counter, sends a notification, or affects external state in any way, guard it.

---

## Marker Files vs Retry Guards

Two separate mechanisms with different purposes:

### Retry Guards (Environment-Based)
- **Purpose:** Prevent side effects on workflow retries
- **Mechanism:** Check `GITHUB_RUN_ATTEMPT` environment variable
- **Scope:** Entire workflow execution
- **Use Case:** "Don't send alert if this is a retry"

### Marker Files (File-Based)
- **Purpose:** Prevent duplicate sends within same attempt
- **Mechanism:** Check for file existence
- **Scope:** Specific operation (e.g., Telegram send)
- **Use Case:** "Don't send alert if already sent this run"

**Both are needed:**
```
                ┌─────────────────────┐
                │ Workflow Execution  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Is retry? (Guard)   │ ← Retry Guard
                └─────┬──────────┬─────┘
                  Yes │          │ No
                      │          │
                      ▼          ▼
                   Skip    ┌─────────────┐
                           │ Marker file? │ ← Marker File
                           └──┬────────┬──┘
                         Yes  │        │ No
                              │        │
                              ▼        ▼
                           Skip     Send Alert
```

**Invariant:** Marker files are never created for retries (because retries are suppressed before marker logic).

---

## Metrics & Monitoring (Future)

### Retries Are Health Signals, Not Failures

**Important:** Retries are not code bugs — they indicate environmental conditions.

High retry rates signal:
- **API Instability:** External dependencies failing temporarily
- **Infrastructure Flakiness:** Transient network/compute issues  
- **Partner SLA Degradation:** Third-party service uptime problems

**This is valuable information for:**
- Capacity planning
- Vendor negotiation
- Infrastructure investment decisions
- SLA monitoring

### Retry Frequency as Health Signal

Track:
- Daily retry count by workflow
- Workflows that retry most often
- Correlation with external failures (API outages, etc.)
- Time-of-day patterns (market hours vs off-hours)

Implementation:
```python
from src.core.retry_guard import should_collect_retry_metrics, get_retry_attempt_number

if should_collect_retry_metrics():
    # Log retry event with context
    metrics.increment("workflow.retries", tags={
        "workflow": os.getenv("GITHUB_WORKFLOW"),
        "attempt": get_retry_attempt_number(),
        "hour": datetime.utcnow().hour,
    })
```

**Status:** Not yet implemented, but logging infrastructure is ready.

**Philosophy:** Treat retry metrics as operational intelligence, not error rates.

---

## Common Pitfalls

### ❌ Wrong: Guard at wrong layer
```python
def send_alert():
    if is_retry_attempt():
        return
    # ... alert logic

def main():
    send_alert()  # Guard too deep - might miss other callers
```

### ✅ Right: Guard at persistence boundary
```python
def send_alert():
    # ... alert logic (pure)

def main():
    if is_retry_attempt():
        return
    send_alert()  # Guard at top level
```

### ❌ Wrong: Inconsistent checks
```python
# Different places doing it differently
attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
if attempt > 1: ...

# vs
if os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1": ...
```

### ✅ Right: Centralized helper
```python
from src.core.retry_guard import is_retry_attempt

if is_retry_attempt(): ...
```

---

## Using Invariants in Practice

### Code Review Language

**Instead of:**
> "Don't forget to check for retries here"

**Say:**
> "Does this violate R1?" (shorter, clearer, referenceable)

**When reviewing PRs:**
- ✅ "This operation reads data only → R1 compliant"
- ❌ "This sends alerts without retry guard → violates R1"
- ⚠️ "This GET has side effects → needs R1 guard"

### Common Phrases
- "Guard against R1 violations"
- "This respects R1 (read-only)"
- "R1 check: Does this mutate external state?"

### Future Invariants (Placeholder)

This naming scheme allows for future additions:
- **R2:** Ordering invariants (if needed)
- **R3:** Monotonicity guarantees (if needed)
- **R4:** Causality constraints (if needed)

For now, R1 is the only formal invariant.

---

## Philosophy

This pattern follows distributed systems principles:

1. **Idempotency:** Running the same operation multiple times has the same effect as running it once
2. **Separation of Concerns:** Computation (pure) vs side effects (guarded)
3. **Defense in Depth:** Multiple guards at different boundaries
4. **Observable:** All suppressions are logged

**Result:** Clean data beats fancy models.

If your learning data is polluted by retry duplicates, no amount of hyperparameter tuning will save you.

---

## Future Extensions

This pattern can be applied to:

- **Email alerts** (`send_email()`)
- **Slack notifications** (`send_slack()`)
- **Discord webhooks** (`send_discord()`)
- **Trade order submission** (`submit_order()`)
- **Cloud uploads** (`upload_to_s3()`)
- **Database writes** (any other persistence)

General rule:
```python
def any_side_effect_operation():
    if is_retry_attempt():
        log_retry_suppression("operation name")
        return
    
    # ... actual side effect logic
```

---

## Testing Checklist

When adding a new side effect:

- [ ] Add retry guard before the operation
- [ ] Log suppression with `log_retry_suppression()`
- [ ] Add test case to `test_retry_logic.py`
- [ ] Update `verify_retry_guards.py` to check new guard
- [ ] Document in this file

---

## References

- **Implementation:** `src/core/retry_guard.py`
- **Documentation:** `RETRY_HANDLING_SUMMARY.md`, `TELEGRAM_TRACKING.md`
- **Tests:** `test_retry_logic.py`, `verify_retry_guards.py`
- **Design Review:** See user feedback in git history

---

## Quick Deployment Check

After deploying, verify the system is working:

```bash
# 1. Verify guards
python verify_retry_guards.py

# 2. Run tests
python test_retry_logic.py

# 3. Check logs after next workflow run
grep "Suppressing.*retry attempt" outputs/logs/*.log

# 4. Verify no duplicate outcomes
python -c "
from src.core.outcome_db import get_outcome_db
db = get_outcome_db()
df = db.get_training_data()
dupes = df[df.duplicated(['ticker', 'pick_date'], keep=False)]
print(f'Duplicates: {len(dupes)}')  # Should be 0
"
```

---

**Last Updated:** 2026-01-30  
**Status:** ✅ Production-Ready  
**Principle:** Retries are compute-only. No side effects. Ever.
