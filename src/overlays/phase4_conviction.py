"""
Phase 4 optional conviction/sizing overlay for trade plans.
"""

from __future__ import annotations


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def apply_phase4_overlay(trade_plan, signals, context, cfg):
    """
    Phase 4 overlay.
    Must return trade_plan unchanged if cfg.enabled is False.
    """
    if not _get(cfg, "enabled", False):
        return trade_plan

    plan = trade_plan.copy()
    positions = plan.get("positions", []) if isinstance(plan, dict) else plan

    # Conviction scoring only used when enabled
    conviction_cfg = _get(cfg, "conviction", {})
    if _get(conviction_cfg, "enabled", False):
        for p in positions:
            score = 0
            score += 2 if p.get("source") == "pro30" else 0
            score += 2 if p.get("weekly_overlap") else 0
            p["conviction_score"] = score

    # Sizing overlay
    sizing_cfg = _get(cfg, "sizing", {})
    if _get(sizing_cfg, "enabled", False):
        tiers = _get(sizing_cfg, "tiers", {}) or {}
        for p in positions:
            tier = str(p.get("conviction_score", 0))
            if tier in tiers:
                p["target_weight"] = tiers[tier]

    if isinstance(plan, dict):
        plan["positions"] = positions
        return plan
    return positions
