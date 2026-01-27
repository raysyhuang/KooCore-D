"""
Phase 3 optional post-filters for candidates.
"""

from __future__ import annotations


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def apply_phase3_filters(candidates_df, context, cfg):
    """
    Phase 3 overlay.
    Must return candidates_df unchanged if cfg.enabled is False.
    """
    if not _get(cfg, "enabled", False):
        return candidates_df

    df = candidates_df.copy()

    movers_cfg = _get(cfg, "movers_filters", {})
    if _get(movers_cfg, "enabled", False):
        min_price = _get(movers_cfg, "min_price", 0)
        rsi_min = _get(movers_cfg, "rsi_min", None)
        rsi_max = _get(movers_cfg, "rsi_max", None)
        atr_pct_max = _get(movers_cfg, "atr_pct_max", None)

        if "close" in df:
            df = df[df["close"] >= min_price]

        if "rsi" in df and rsi_min is not None and rsi_max is not None:
            df = df[(df["rsi"] >= rsi_min) & (df["rsi"] <= rsi_max)]

        if "atr_pct" in df and atr_pct_max is not None:
            df = df[df["atr_pct"] <= atr_pct_max]

    # cooldown / regime filters are NO-OP unless enabled
    return df
