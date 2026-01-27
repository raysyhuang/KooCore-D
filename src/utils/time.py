"""
Time utilities.

Single source of "now" for UTC-aware timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return ISO string with +00:00 offset."""
    return utc_now().isoformat()


def utc_now_iso_z() -> str:
    """Return ISO string with trailing Z."""
    return utc_now().isoformat().replace("+00:00", "Z")


def utc_now_timestamp() -> float:
    """Return UTC timestamp as float seconds."""
    return utc_now().timestamp()


def utc_today() -> datetime.date:
    """Return current UTC date."""
    return utc_now().date()
