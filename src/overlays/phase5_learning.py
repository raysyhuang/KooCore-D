"""
Phase 5 passive learning overlay.
"""

from __future__ import annotations


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def record_phase5_learning(outcomes, signals, context, cfg):
    """
    Passive learning recorder.
    Does nothing unless cfg.enabled is True.
    """
    if not _get(cfg, "enabled", False):
        return []

    records = []

    for o in outcomes or []:
        record = {
            "ticker": o.get("ticker"),
            "hit_7pct": o.get("hit_7pct"),
            "max_return": o.get("max_return_pct", o.get("max_return")),
            "source": o.get("source"),
            "regime": (context or {}).get("regime"),
            "signals": o.get("signals", []),
        }
        records.append(record)

    return records
