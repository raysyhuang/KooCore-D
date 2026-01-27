"""
Conviction Ranker Module

Combines candidates from all scanners, scores with adaptive weights,
and returns the top 1-3 highest conviction picks.

OPTIMIZED Jan 2026:
- Pro30: 50% hit rate (best performer)
- Weekly Rank 1: 38% hit rate
- Weekly Rank 2: 29% hit rate
- Movers: 0% hit rate (disabled)
- Confluence (overlap) is highest conviction signal
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """A candidate with conviction scoring."""
    ticker: str
    conviction_score: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    sources: List[str]
    overlap_count: int
    
    # Original data from scanners
    composite_score: Optional[float] = None
    technical_score: Optional[float] = None
    rank: Optional[int] = None  # For weekly picks
    
    # Features for learning
    rsi14: Optional[float] = None
    volume_ratio_3d_to_20d: Optional[float] = None
    dist_to_52w_high_pct: Optional[float] = None
    realized_vol_5d_ann_pct: Optional[float] = None
    above_ma10: Optional[bool] = None
    above_ma20: Optional[bool] = None
    above_ma50: Optional[bool] = None
    sector: Optional[str] = None
    market_cap_usd: Optional[float] = None
    
    # Entry information
    current_price: Optional[float] = None
    name: Optional[str] = None
    
    # Model notes
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "ticker": self.ticker,
            "conviction_score": self.conviction_score,
            "confidence": self.confidence,
            "sources": self.sources,
            "overlap_count": self.overlap_count,
            "composite_score": self.composite_score,
            "technical_score": self.technical_score,
            "rank": self.rank,
            "current_price": self.current_price,
            "name": self.name,
            "sector": self.sector,
            "notes": self.notes,
        }


def merge_and_dedupe(
    weekly_picks: List[Dict[str, Any]],
    pro30_picks: List[str],
    movers_picks: List[str],
    weekly_features: Optional[Dict[str, Dict]] = None,
    pro30_features: Optional[Dict[str, Dict]] = None,
    movers_features: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge candidates from all sources and deduplicate.
    
    For tickers appearing in multiple sources, combine their features
    and track the overlap.
    
    Args:
        weekly_picks: List of weekly scanner dicts
        pro30_picks: List of Pro30 ticker strings
        movers_picks: List of movers ticker strings
        weekly_features: Optional dict mapping ticker -> features for weekly
        pro30_features: Optional dict mapping ticker -> features for pro30
        movers_features: Optional dict mapping ticker -> features for movers
    
    Returns:
        List of merged candidate dicts with overlap tracking
    """
    weekly_features = weekly_features or {}
    pro30_features = pro30_features or {}
    movers_features = movers_features or {}
    
    # Build sets for overlap detection
    weekly_tickers = {p.get("ticker") for p in weekly_picks if p.get("ticker")}
    pro30_set = set(pro30_picks or [])
    movers_set = set(movers_picks or [])
    
    # Track all candidates
    candidates: Dict[str, Dict[str, Any]] = {}
    
    # Process weekly picks (richest data)
    for pick in weekly_picks:
        ticker = pick.get("ticker")
        if not ticker:
            continue
        
        features = weekly_features.get(ticker, {})
        
        candidates[ticker] = {
            "ticker": ticker,
            "sources": ["weekly_top5"],
            "overlap_count": 1,
            "composite_score": pick.get("composite_score"),
            "technical_score": features.get("technical_score") or pick.get("technical_score"),
            "rank": pick.get("rank"),
            "current_price": pick.get("current_price"),
            "name": pick.get("name"),
            "rsi14": features.get("rsi14"),
            "volume_ratio_3d_to_20d": features.get("volume_ratio_3d_to_20d"),
            "dist_to_52w_high_pct": features.get("dist_to_52w_high_pct"),
            "realized_vol_5d_ann_pct": features.get("realized_vol_5d_ann_pct"),
            "above_ma10": features.get("above_ma10"),
            "above_ma20": features.get("above_ma20"),
            "above_ma50": features.get("above_ma50"),
            "sector": features.get("sector") or pick.get("sector"),
            "market_cap_usd": features.get("market_cap_usd"),
        }
        
        # Check for overlaps
        if ticker in pro30_set:
            candidates[ticker]["sources"].append("pro30")
            candidates[ticker]["overlap_count"] += 1
        if ticker in movers_set:
            candidates[ticker]["sources"].append("movers")
            candidates[ticker]["overlap_count"] += 1
    
    # Process Pro30 picks
    for ticker in pro30_picks or []:
        if ticker in candidates:
            continue  # Already added from weekly
        
        features = pro30_features.get(ticker, {})
        
        candidates[ticker] = {
            "ticker": ticker,
            "sources": ["pro30"],
            "overlap_count": 1,
            "composite_score": features.get("composite_score"),
            "technical_score": features.get("technical_score"),
            "rank": None,
            "current_price": features.get("current_price"),
            "name": features.get("name"),
            "rsi14": features.get("rsi14"),
            "volume_ratio_3d_to_20d": features.get("volume_ratio_3d_to_20d"),
            "dist_to_52w_high_pct": features.get("dist_to_52w_high_pct"),
            "realized_vol_5d_ann_pct": features.get("realized_vol_5d_ann_pct"),
            "above_ma10": features.get("above_ma10"),
            "above_ma20": features.get("above_ma20"),
            "above_ma50": features.get("above_ma50"),
            "sector": features.get("sector"),
            "market_cap_usd": features.get("market_cap_usd"),
        }
        
        # Check for overlaps
        if ticker in weekly_tickers:
            candidates[ticker]["sources"].append("weekly_top5")
            candidates[ticker]["overlap_count"] += 1
        if ticker in movers_set:
            candidates[ticker]["sources"].append("movers")
            candidates[ticker]["overlap_count"] += 1
    
    # Process movers picks
    for ticker in movers_picks or []:
        if ticker in candidates:
            continue  # Already added
        
        features = movers_features.get(ticker, {})
        
        candidates[ticker] = {
            "ticker": ticker,
            "sources": ["movers"],
            "overlap_count": 1,
            "composite_score": features.get("composite_score"),
            "technical_score": features.get("technical_score"),
            "rank": None,
            "current_price": features.get("current_price"),
            "name": features.get("name"),
            "rsi14": features.get("rsi14"),
            "volume_ratio_3d_to_20d": features.get("volume_ratio_3d_to_20d"),
            "dist_to_52w_high_pct": features.get("dist_to_52w_high_pct"),
            "realized_vol_5d_ann_pct": features.get("realized_vol_5d_ann_pct"),
            "above_ma10": features.get("above_ma10"),
            "above_ma20": features.get("above_ma20"),
            "above_ma50": features.get("above_ma50"),
            "sector": features.get("sector"),
            "market_cap_usd": features.get("market_cap_usd"),
        }
        
        # Check for overlaps
        if ticker in weekly_tickers:
            candidates[ticker]["sources"].append("weekly_top5")
            candidates[ticker]["overlap_count"] += 1
        if ticker in pro30_set:
            candidates[ticker]["sources"].append("pro30")
            candidates[ticker]["overlap_count"] += 1
    
    return list(candidates.values())


def apply_confidence_cutoff(
    ranked: List[RankedCandidate],
    max_picks: int = 3,
    min_confidence: str = "MEDIUM",
) -> List[RankedCandidate]:
    """
    Apply confidence cutoff and limit picks.
    
    Args:
        ranked: List of ranked candidates (sorted by score)
        max_picks: Maximum number of picks to return
        min_confidence: Minimum confidence level ("HIGH", "MEDIUM", "LOW")
    
    Returns:
        Filtered list of top picks
    """
    confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_conf_value = confidence_order.get(min_confidence, 1)
    
    filtered = []
    for candidate in ranked:
        conf_value = confidence_order.get(candidate.confidence, 0)
        if conf_value >= min_conf_value:
            filtered.append(candidate)
        
        if len(filtered) >= max_picks:
            break
    
    return filtered


def rank_candidates(
    weekly_picks: List[Dict[str, Any]],
    pro30_picks: List[str],
    movers_picks: List[str],
    scorer: Optional[Any] = None,
    weekly_features: Optional[Dict[str, Dict]] = None,
    pro30_features: Optional[Dict[str, Dict]] = None,
    movers_features: Optional[Dict[str, Dict]] = None,
    max_picks: int = 3,
    min_confidence: str = "MEDIUM",
) -> Dict[str, Any]:
    """
    Combine all candidates, score with adaptive weights,
    return top 1-3 with confidence levels.
    
    Args:
        weekly_picks: List of weekly scanner dicts
        pro30_picks: List of Pro30 ticker strings
        movers_picks: List of movers ticker strings
        scorer: AdaptiveScorer instance (uses default if None)
        weekly_features: Optional features dict for weekly picks
        pro30_features: Optional features dict for pro30 picks
        movers_features: Optional features dict for movers picks
        max_picks: Maximum picks to return (default: 3)
        min_confidence: Minimum confidence for inclusion
    
    Returns:
        Dict with top_picks, all_candidates, model_notes
    """
    # Get or create scorer
    if scorer is None:
        from src.core.adaptive_scorer import get_adaptive_scorer
        scorer = get_adaptive_scorer()
    
    # Merge and dedupe candidates
    all_candidates = merge_and_dedupe(
        weekly_picks=weekly_picks,
        pro30_picks=pro30_picks,
        movers_picks=movers_picks,
        weekly_features=weekly_features,
        pro30_features=pro30_features,
        movers_features=movers_features,
    )
    
    if not all_candidates:
        return {
            "top_picks": [],
            "all_candidates": [],
            "model_info": scorer.get_model_info(),
            "model_notes": ["No candidates found"],
        }
    
    # Score all candidates
    ranked_candidates: List[RankedCandidate] = []
    
    for c in all_candidates:
        # Build features dict for scoring
        features = {
            "ticker": c["ticker"],
            "source": c["sources"][0] if c["sources"] else "",
            "composite_score": c.get("composite_score"),
            "technical_score": c.get("technical_score"),
            "overlap_count": c.get("overlap_count", 1),
            "overlap_sources": c.get("sources", []),
            "rank": c.get("rank"),
            "rsi14": c.get("rsi14"),
            "volume_ratio_3d_to_20d": c.get("volume_ratio_3d_to_20d"),
            "sector": c.get("sector"),
        }
        
        # Compute conviction score and confidence
        conviction_score = scorer.score(features)
        confidence = scorer.compute_confidence(features)
        
        # Build notes (OPTIMIZED Jan 2026 with updated hit rates)
        notes = []
        if c.get("overlap_count", 1) >= 3:
            notes.append("ALL-THREE overlap (highest conviction)")
        elif c.get("overlap_count", 1) >= 2:
            sources_str = "+".join(c.get("sources", []))
            notes.append(f"{sources_str} overlap")
        
        if c.get("rank") == 1:
            notes.append("Rank 1 weekly pick (38% hit rate)")
        elif c.get("rank") == 2:
            notes.append("Rank 2 weekly pick (29% hit rate)")
        
        if "pro30" in c.get("sources", []):
            notes.append("Pro30 pattern (50% hit rate - BEST)")
        
        # RSI sweet spot indicator
        rsi = c.get("rsi14")
        if rsi is not None:
            if 55 <= rsi <= 65:
                notes.append("RSI in sweet spot (55-65)")
            elif rsi > 70:
                notes.append("⚠️ RSI overbought (>70)")
        
        # Volume confirmation indicator
        vol_ratio = c.get("volume_ratio_3d_to_20d")
        if vol_ratio is not None and vol_ratio >= 2.0:
            notes.append("Strong volume confirmation (≥2x)")
        
        candidate = RankedCandidate(
            ticker=c["ticker"],
            conviction_score=conviction_score,
            confidence=confidence,
            sources=c.get("sources", []),
            overlap_count=c.get("overlap_count", 1),
            composite_score=c.get("composite_score"),
            technical_score=c.get("technical_score"),
            rank=c.get("rank"),
            rsi14=c.get("rsi14"),
            volume_ratio_3d_to_20d=c.get("volume_ratio_3d_to_20d"),
            dist_to_52w_high_pct=c.get("dist_to_52w_high_pct"),
            realized_vol_5d_ann_pct=c.get("realized_vol_5d_ann_pct"),
            above_ma10=c.get("above_ma10"),
            above_ma20=c.get("above_ma20"),
            above_ma50=c.get("above_ma50"),
            sector=c.get("sector"),
            market_cap_usd=c.get("market_cap_usd"),
            current_price=c.get("current_price"),
            name=c.get("name"),
            notes=notes,
        )
        
        ranked_candidates.append(candidate)
    
    # Sort by conviction score (descending)
    ranked_candidates.sort(key=lambda x: x.conviction_score, reverse=True)
    
    # Apply confidence cutoff
    top_picks = apply_confidence_cutoff(
        ranked_candidates,
        max_picks=max_picks,
        min_confidence=min_confidence,
    )
    
    # Generate model notes
    model_info = scorer.get_model_info()
    model_notes = []
    
    # Count overlaps
    all_three_count = sum(1 for c in ranked_candidates if c.overlap_count >= 3)
    weekly_pro30_count = sum(
        1 for c in ranked_candidates 
        if "weekly_top5" in c.sources and "pro30" in c.sources
    )
    
    if all_three_count > 0:
        model_notes.append(f"{all_three_count} ALL-THREE overlap(s) today (highest conviction)")
    elif weekly_pro30_count > 0:
        model_notes.append(f"{weekly_pro30_count} Weekly+Pro30 overlap(s) today")
    else:
        model_notes.append("No ALL-THREE overlaps today")
    
    # Source breakdown (OPTIMIZED with correct hit rates)
    pro30_in_top = sum(1 for c in top_picks if "pro30" in c.sources)
    if pro30_in_top > 0:
        model_notes.append(f"{pro30_in_top} of {len(top_picks)} top picks from Pro30 (50% hit rate - BEST)")
    
    # Rank breakdown
    rank1_in_top = sum(1 for c in top_picks if c.rank == 1)
    if rank1_in_top > 0:
        model_notes.append(f"{rank1_in_top} Rank 1 pick(s) (38% hit rate)")
    
    # RSI warnings
    overbought_count = sum(1 for c in top_picks if c.rsi14 is not None and c.rsi14 > 70)
    if overbought_count > 0:
        model_notes.append(f"⚠️ {overbought_count} pick(s) with RSI > 70 (overbought warning)")
    
    # Model training status
    if model_info.get("needs_training"):
        model_notes.append(f"Model needs training (only {model_info.get('observations', 0)} outcomes)")
    
    return {
        "top_picks": [c.to_dict() for c in top_picks],
        "all_candidates": [c.to_dict() for c in ranked_candidates],
        "model_info": model_info,
        "model_notes": model_notes,
        "summary": {
            "total_candidates": len(ranked_candidates),
            "top_picks_count": len(top_picks),
            "all_three_overlaps": all_three_count,
            "weekly_pro30_overlaps": weekly_pro30_count,
        }
    }


def format_conviction_picks(result: Dict[str, Any]) -> str:
    """
    Format conviction picks for display.
    
    Returns formatted string for logging/display.
    """
    lines = []
    
    model_info = result.get("model_info", {})
    model_version = model_info.get("version", 1)
    hit_rate = model_info.get("overall_hit_rate", 0)
    
    lines.append("=" * 60)
    lines.append(f"TOP CONVICTION PICKS (Model v{model_version}, {hit_rate*100:.0f}% historical hit rate)")
    lines.append("=" * 60)
    
    top_picks = result.get("top_picks", [])
    
    if not top_picks:
        lines.append("  No picks meet confidence threshold today.")
    else:
        for i, pick in enumerate(top_picks, 1):
            ticker = pick.get("ticker", "?")
            score = pick.get("conviction_score", 0)
            confidence = pick.get("confidence", "?")
            notes = pick.get("notes", [])
            notes_str = f" ({', '.join(notes)})" if notes else ""
            
            lines.append(f"  {i}. {ticker:<6} - Score: {score:.1f} | Confidence: {confidence}{notes_str}")
    
    # Model notes
    model_notes = result.get("model_notes", [])
    if model_notes:
        lines.append("")
        lines.append("Model Notes:")
        for note in model_notes:
            lines.append(f"  - {note}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
