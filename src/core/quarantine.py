"""
Ticker quarantine utilities.

Tracks symbols that repeatedly fail data downloads (delisted, missing data, etc.)
and temporarily excludes them from the universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json


DEFAULT_QUARANTINE_PATH = Path("data/bad_tickers.json")


def _utcnow() -> datetime:
    return datetime.utcnow()


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", ""))
    except Exception:
        return None


def _load_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_raw(records: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_quarantine(path: Path | str = DEFAULT_QUARANTINE_PATH) -> dict:
    """Load and prune quarantine records."""
    path = Path(path)
    records = _load_raw(path)
    now = _utcnow()
    pruned = {}
    for ticker, data in records.items():
        until = _parse_ts(data.get("until"))
        if until and until > now:
            pruned[ticker] = data
    if pruned != records:
        _save_raw(pruned, path)
    return pruned


def get_quarantined_tickers(path: Path | str = DEFAULT_QUARANTINE_PATH) -> set[str]:
    records = load_quarantine(path)
    return {t.upper() for t in records.keys()}


def record_bad_tickers(
    tickers: list[str],
    reasons: dict[str, str] | None = None,
    *,
    days: int = 7,
    source: str = "unknown",
    path: Path | str = DEFAULT_QUARANTINE_PATH,
) -> None:
    """Record tickers into quarantine for a fixed number of days."""
    if not tickers:
        return
    path = Path(path)
    records = _load_raw(path)
    now = _utcnow()
    until = (now + timedelta(days=days)).isoformat() + "Z"
    for t in tickers:
        if not t:
            continue
        key = str(t).upper()
        rec = records.get(key, {})
        rec["until"] = until
        rec["last_seen"] = now.isoformat() + "Z"
        rec["source"] = source
        reason = (reasons or {}).get(t) or (reasons or {}).get(key)
        if reason:
            rec["reason"] = str(reason)[:200]
        rec["count"] = int(rec.get("count") or 0) + 1
        records[key] = rec
    _save_raw(records, path)
