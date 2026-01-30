"""
Retry Guard - Centralized retry detection for GitHub Actions workflows.

Core Principle:
    Retries re-run computation but MUST NOT emit side effects.

Invariant R1: Retry Purity
    Retries may read external state but must not mutate it.

Side Effects (must be guarded):
    Any operations that mutate external state or communicate results externally:
    - Sending alerts (Telegram, Slack, email)
    - Writing to databases (outcomes, learning data)
    - Uploading to cloud storage (S3, GCS)
    - Making API calls that mutate state (POST, PUT, DELETE)
    - Creating webhook notifications
    - Submitting trade orders

Safe Operations (allowed on retry):
    Operations that only read state without mutation:
    - Fetching market data (GET requests)
    - Reading from databases (SELECT queries)
    - Downloading from cloud storage
    - Computing scores and transformations
    - Generating local reports

This module provides a single source of truth for detecting workflow retries,
ensuring consistent behavior across all persistence boundaries.
"""

import os
import logging

logger = logging.getLogger(__name__)


def is_retry_attempt() -> bool:
    """
    Check if current execution is a workflow retry (attempt > 1).
    
    Returns:
        True if this is a retry attempt (attempt > 1)
        False if this is the first attempt or not in GitHub Actions
    
    Note:
        Retries should execute computation but never emit side effects:
        - No Telegram alerts
        - No outcome persistence
        - No learning data writes
        - No external API calls with side effects
    
    Usage:
        if is_retry_attempt():
            logger.info("Skipping side effect (retry attempt)")
            return
    """
    try:
        attempt_num = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
        return attempt_num > 1
    except (ValueError, TypeError):
        # If conversion fails, assume first attempt (safe default)
        return False


def get_retry_attempt_number() -> int:
    """
    Get the current retry attempt number.
    
    Returns:
        Attempt number (1 for first attempt, 2+ for retries)
        Defaults to 1 if not in GitHub Actions or parse fails
    """
    try:
        return int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    except (ValueError, TypeError):
        return 1


def log_retry_suppression(context: str, **metadata) -> None:
    """
    Log that a side effect was suppressed due to retry.
    
    Args:
        context: Description of what was suppressed (e.g., "Telegram alert", "outcome persistence")
        **metadata: Additional context to log (e.g., ticker, run_id)
    """
    attempt = get_retry_attempt_number()
    log_parts = [f"Suppressing {context} (retry attempt={attempt})"]
    
    if metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        log_parts.append(f"[{meta_str}]")
    
    logger.info(" ".join(log_parts))


# Optional: Future metrics collection
def should_collect_retry_metrics() -> bool:
    """
    Check if retry metrics should be collected.
    
    Note: Retries are not failures — they are health signals.
    
    Future enhancement: Track retry frequency as pipeline health signal:
    - API instability detector (external deps failing)
    - Infrastructure flakiness metric (transient errors)
    - External dependency SLA monitor (partner uptime)
    
    High retry rates indicate environmental issues, not code bugs.
    
    For now, always returns False (metrics not yet implemented).
    """
    return False
