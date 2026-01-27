# src/core/momentum_norm.py
"""
ATR-normalized momentum adjustments.

Reduces scores for:
- Ultra-high ATR% names (junk/penny stock behavior)
- Very extended moves (late-stage chasing)

Boosts scores for:
- Moderate volatility (tradeable momentum)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .types import FeatureSet


@dataclass(frozen=True)
class MomentumNormResult:
    """Result of momentum normalization adjustment."""
    score_adj: float
    evidence: Dict[str, Any]


def momentum_atr_adjust(features: FeatureSet) -> MomentumNormResult:
    """
    Calculate ATR-based momentum adjustments.
    
    Adjustments:
    - ATR% >= 12: -1.5 (high volatility junk)
    - ATR% >= 8: -0.75 (elevated volatility)
    - ATR% < 8: +0.25 (moderate volatility bonus)
    - 5d return > 2x ATR: -1.0 (extended/chasing penalty)
    
    Args:
        features: FeatureSet with ATR and return data
    
    Returns:
        MomentumNormResult with adjustment and evidence
    """
    e: Dict[str, Any] = {}
    
    # Check for required data
    if features.last_close is None or features.atr14 is None or features.atr14 <= 0:
        return MomentumNormResult(
            score_adj=0.0,
            evidence={"note": "missing atr or price data"}
        )

    # Calculate ATR as percentage of price
    atr_pct = (features.atr14 / features.last_close) * 100.0
    e["atr_pct"] = round(atr_pct, 2)

    adj = 0.0
    
    # Volatility-based adjustments
    if atr_pct >= 12:
        adj -= 1.5
        e["volatility_adj"] = -1.5
        e["volatility_note"] = "high_volatility_penalty"
    elif atr_pct >= 8:
        adj -= 0.75
        e["volatility_adj"] = -0.75
        e["volatility_note"] = "elevated_volatility_penalty"
    else:
        adj += 0.25
        e["volatility_adj"] = 0.25
        e["volatility_note"] = "moderate_volatility_bonus"

    # Extension penalty - penalize very extended 5d moves
    if features.ret_5d is not None and atr_pct > 0:
        ret5_atr = features.ret_5d / atr_pct
        e["ret5_atr_units"] = round(ret5_atr, 2)
        
        if ret5_atr > 2.0:
            adj -= 1.0
            e["extension_adj"] = -1.0
            e["extension_note"] = "extended_move_penalty"
        else:
            e["extension_adj"] = 0.0
    else:
        e["ret5_atr_units"] = None

    return MomentumNormResult(
        score_adj=round(adj, 2),
        evidence=e
    )
