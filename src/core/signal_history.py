"""
Signal history utilities for persistence gating.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json


DEFAULT_SIGNAL_HISTORY_PATH = Path("data/signal_history.json")


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def load_signal_history(path: Path | str = DEFAULT_SIGNAL_HISTORY_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_signal_history(history: dict, path: Path | str = DEFAULT_SIGNAL_HISTORY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def update_signal_history(
    history: dict,
    date_str: str,
    signals: list[dict],
    *,
    max_days: int = 30,
    source: str = "swing",
) -> dict:
    """Append signals for date_str and prune history."""
    dt = _parse_date(date_str)
    if dt is None:
        return history
    for item in signals:
        ticker = str(item.get("ticker", "")).upper()
        if not ticker:
            continue
        entry = {
            "date": date_str,
            "score": float(item.get("swing_score") or item.get("technical_score") or 0),
            "source": source,
        }
        history.setdefault(ticker, [])
        history[ticker].append(entry)
    cutoff = dt - timedelta(days=max_days)
    pruned = {}
    for ticker, entries in history.items():
        kept = []
        for e in entries:
            d = _parse_date(e.get("date", ""))
            if d and d >= cutoff:
                kept.append(e)
        if kept:
            pruned[ticker] = kept
    return pruned


def check_persistence(
    history: dict,
    ticker: str,
    date_str: str,
    *,
    lookback_days: int = 3,
    min_hits: int = 2,
    min_score_improve: float = 0.0,
    current_score: float | None = None,
    allow_new: bool = False,
    min_score_new: float | None = None,
) -> dict:
    """Check if ticker has persisted in the recent window."""
    ticker = str(ticker).upper()
    dt = _parse_date(date_str)
    if not dt:
        return {"passed": True, "hits": 0, "last_score": None}
    window_start = dt - timedelta(days=lookback_days)
    entries = history.get(ticker, [])
    hits = 0
    last_score = None
    last_dt = None
    for e in entries:
        d = _parse_date(e.get("date", ""))
        if not d:
            continue
        if window_start <= d < dt:
            hits += 1
            if last_dt is None or d > last_dt:
                last_dt = d
                last_score = e.get("score")
    improved = False
    if current_score is not None and last_score is not None:
        improved = (float(current_score) - float(last_score)) >= float(min_score_improve)
    passed = hits >= min_hits or improved
    if not passed and allow_new and hits == 0:
        try:
            score_val = float(current_score) if current_score is not None else None
        except Exception:
            score_val = None
        if score_val is not None:
            threshold = float(min_score_new) if min_score_new is not None else 0.0
            if score_val >= threshold:
                passed = True
    return {
        "passed": passed,
        "hits": hits,
        "last_score": last_score,
        "improved": improved,
    }
