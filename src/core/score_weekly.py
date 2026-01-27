# src/core/score_weekly.py
"""
Weekly scoring logic using FeatureSet only (no pandas import).

This module is intentionally isolated from DataFrame operations to prevent
any possibility of lookahead bias. All data must come through FeatureSet.

IMPORTANT: Do NOT add 'import pandas' to this file.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from .types import FeatureSet


@dataclass(frozen=True)
class ScoreResult:
    """Result of scoring a FeatureSet."""
    score: float
    cap_applied: Optional[float]
    evidence: Dict[str, Any]
    data_gaps: List[str]


def score_weekly(features: FeatureSet) -> ScoreResult:
    """
    Score a ticker based on its FeatureSet for weekly momentum strategy.
    
    Scoring criteria (max 10 points):
    - Within 5% of 52W high: +2.0
    - Volume ratio >= 1.5: +2.0
    - RSI in 50-70 range: +2.0
    - Above MA10/MA20/MA50: +2.0
    - Realized vol >= 20%: +2.0
    
    If any data is missing, score is capped at 6.0.
    
    Args:
        features: FeatureSet containing all computed features
    
    Returns:
        ScoreResult with score, evidence, and any data gaps
    """
    gaps: List[str] = []
    points = 0.0
    e: Dict[str, Any] = {}

    # Check for minimum data
    if features.last_close is None:
        return ScoreResult(
            score=0.0,
            cap_applied=6.0,
            evidence={},
            data_gaps=["Insufficient price data"]
        )

    # 1) Within 5% of 52W high (+2.0)
    if features.dist_52w_high_pct is None:
        gaps.append("52W high calc missing")
        e["dist_to_52w_high_pct"] = None
        e["within_5pct_52w_high"] = None
    else:
        e["dist_to_52w_high_pct"] = round(features.dist_52w_high_pct, 2)
        if features.dist_52w_high_pct <= 5.0:
            points += 2.0
            e["within_5pct_52w_high"] = True
        else:
            e["within_5pct_52w_high"] = False

    # 2) Volume ratio >= 1.5 (+2.0)
    if features.vol_ratio_3_20 is None:
        gaps.append("volume ratio missing")
        e["volume_ratio_3d_to_20d"] = None
    else:
        e["volume_ratio_3d_to_20d"] = round(features.vol_ratio_3_20, 2)
        if features.vol_ratio_3_20 >= 1.5:
            points += 2.0

    # 3) RSI in 50-70 band (+2.0)
    if features.rsi14 is None:
        gaps.append("RSI missing")
        e["rsi14"] = None
    else:
        e["rsi14"] = round(features.rsi14, 2)
        if 50.0 <= features.rsi14 <= 70.0:
            points += 2.0

    # 4) MA stack - price above all three MAs (+2.0)
    if features.ma10 is None or features.ma20 is None or features.ma50 is None:
        gaps.append("MA missing")
        e["above_ma10_ma20_ma50"] = None
    else:
        above = (
            features.last_close > features.ma10 and
            features.last_close > features.ma20 and
            features.last_close > features.ma50
        )
        e["ma10"] = round(features.ma10, 2)
        e["ma20"] = round(features.ma20, 2)
        e["ma50"] = round(features.ma50, 2)
        e["above_ma10_ma20_ma50"] = above
        if above:
            points += 2.0

    # 5) Realized volatility >= 20% (+2.0)
    if features.realized_vol_5d_ann_pct is None:
        gaps.append("realized vol missing")
        e["realized_vol_5d_ann_pct"] = None
    else:
        e["realized_vol_5d_ann_pct"] = round(features.realized_vol_5d_ann_pct, 2)
        if features.realized_vol_5d_ann_pct >= 20.0:
            points += 2.0

    # Cap score and finalize
    score = min(10.0, points)
    cap = None
    if gaps:
        cap = 6.0
        score = min(score, cap)

    return ScoreResult(
        score=round(score, 2),
        cap_applied=cap,
        evidence=e,
        data_gaps=gaps
    )
