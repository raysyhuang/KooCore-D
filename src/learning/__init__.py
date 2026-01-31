"""
Phase-5 Learning Infrastructure

Provides:
- phase5_schema: Canonical data structures for learning
- phase5_store: JSONL-based append-only storage  
- phase5_resolver: Outcome resolution
- phase5_analyzer: Comprehensive analysis

Usage:
    from src.learning import (
        Phase5Row,
        Phase5Store,
        Phase5Resolver,
        Phase5Analyzer,
        build_phase5_row,
    )
"""

from src.learning.phase5_schema import (
    Phase5Row,
    Phase5Identity,
    Phase5Signals,
    Phase5HybridContext,
    Phase5TechnicalBuckets,
    Phase5CatalystFlags,
    Phase5Outcome,
    bucket_rsi,
    bucket_distance_52w,
    bucket_liquidity,
    bucket_volatility,
    derive_trend_state,
    bucket_pro30_rank,
    bucket_primary_rank,
    validate_row,
)

from src.learning.phase5_store import (
    Phase5Store,
    get_phase5_store,
)

from src.learning.phase5_resolver import (
    Phase5Resolver,
    get_phase5_resolver,
)

from src.learning.phase5_analyzer import (
    Phase5Analyzer,
    get_phase5_analyzer,
)


__all__ = [
    # Schema
    "Phase5Row",
    "Phase5Identity",
    "Phase5Signals",
    "Phase5HybridContext",
    "Phase5TechnicalBuckets",
    "Phase5CatalystFlags",
    "Phase5Outcome",
    # Store
    "Phase5Store",
    "get_phase5_store",
    # Resolver
    "Phase5Resolver",
    "get_phase5_resolver",
    # Analyzer
    "Phase5Analyzer",
    "get_phase5_analyzer",
    # Builder
    "build_phase5_row",
]


def build_phase5_row(
    scan_date: str,
    ticker: str,
    primary_strategy: str,
    regime: str,
    *,
    # Signal flags
    in_swing_top5: bool = False,
    swing_rank: int = None,
    in_weekly_top5: bool = False,
    weekly_rank: int = None,
    in_pro30: bool = False,
    pro30_rank: int = None,
    pro30_score: float = None,
    in_movers: bool = False,
    in_confluence: bool = False,
    confluence_score: int = None,
    overlap_primary_pro30: bool = False,
    overlap_all_three: bool = False,
    # Hybrid context
    hybrid_score: float = 0.0,
    hybrid_rank: int = 0,
    hybrid_sources: list = None,
    weights_snapshot: dict = None,
    in_hybrid_top3: bool = False,
    in_conviction_picks: bool = False,
    # Technical buckets (raw values, will be bucketed)
    rsi: float = None,
    atr_pct: float = None,
    adv20: float = None,
    distance_52w_pct: float = None,
    above_ma20: bool = False,
    above_ma50: bool = False,
    ret_20d: float = None,
    # Catalyst
    has_known_catalyst: bool = False,
    catalyst_type: str = None,
    has_earnings_within_7d: bool = False,
    # Metadata
    run_id: str = None,
) -> Phase5Row:
    """
    Build a Phase5Row from raw inputs.
    
    Convenience function that handles bucketing and structure creation.
    """
    from datetime import datetime
    
    # Build identity
    identity = Phase5Identity(
        scan_date=scan_date,
        ticker=ticker,
        primary_strategy=primary_strategy,
        regime=regime,
    )
    
    # Build signals
    signals = Phase5Signals(
        in_swing_top5=in_swing_top5,
        swing_rank=swing_rank,
        in_weekly_top5=in_weekly_top5,
        weekly_rank=weekly_rank,
        in_pro30=in_pro30,
        pro30_rank=pro30_rank,
        pro30_score=pro30_score,
        in_movers=in_movers,
        in_confluence=in_confluence,
        confluence_score=confluence_score,
        overlap_primary_pro30=overlap_primary_pro30,
        overlap_all_three=overlap_all_three,
    )
    
    # Build hybrid context
    hybrid = Phase5HybridContext(
        hybrid_score=hybrid_score,
        hybrid_rank=hybrid_rank,
        hybrid_sources=hybrid_sources or [],
        weights_snapshot=weights_snapshot or {},
        in_hybrid_top3=in_hybrid_top3,
        in_conviction_picks=in_conviction_picks,
    )
    
    # Build technical buckets
    technicals = Phase5TechnicalBuckets(
        trend_state=derive_trend_state(above_ma20, above_ma50, ret_20d),
        rsi_bucket=bucket_rsi(rsi),
        distance_52w_bucket=bucket_distance_52w(distance_52w_pct),
        liquidity_bucket=bucket_liquidity(adv20),
        volatility_bucket=bucket_volatility(atr_pct),
        _raw_rsi=rsi,
        _raw_atr_pct=atr_pct,
        _raw_adv20=adv20,
    )
    
    # Build catalyst flags
    catalyst = Phase5CatalystFlags(
        has_known_catalyst=has_known_catalyst,
        catalyst_type=catalyst_type,
        has_earnings_within_7d=has_earnings_within_7d,
    )
    
    # Build row
    return Phase5Row(
        identity=identity,
        signals=signals,
        hybrid=hybrid,
        technicals=technicals,
        catalyst=catalyst,
        outcome=None,  # Resolved later
        run_id=run_id,
        created_at=datetime.now().isoformat(),
    )
