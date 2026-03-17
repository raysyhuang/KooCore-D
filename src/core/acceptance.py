"""App-level acceptance layer — portfolio decision above engine scoring.

Engines are idea generators. This module decides WHEN to participate
and HOW MANY names to allow, optimizing for precision (high hit/win rate)
over quantity.

Three concepts:
  1. Day Quality Score — should we trade today at all?
  2. Candidate Tier — A/B/C classification per pick
  3. Book Decision — how many picks survive given day quality + tier mix

IMPORTANT: This layer must receive the FULL eligible candidate set
(post-dedupe, post-score-floor, post-hard-vetoes) — not a pre-truncated
list. It IS the allocator; truncation happens here.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DayQuality:
    """Market context assessment for a single trading day."""
    breadth_pct: float          # % of universe above SMA20
    regime: str                 # bull/choppy/bear
    signal_density: float       # eligible_candidates / universe_size
    top_score: float            # best candidate composite score
    score: float = 0.0         # overall day quality (0-100)
    components: dict = field(default_factory=dict)


@dataclass
class AcceptanceResult:
    """Output of the acceptance layer."""
    day_quality: DayQuality
    mode: str                   # "abstain", "selective", "full"
    accepted: list[tuple[str, object, str]]  # (engine, signal, tier)
    rejected: list[tuple[str, object, str, str]]  # (engine, signal, tier, reason)
    abstained: bool = False
    eligible_count: int = 0     # total candidates seen before acceptance


def score_day_quality(
    breadth_pct: float,
    regime: str,
    eligible_count: int,
    universe_size: int,
    top_score: float,
    sector_concentration: float = 0.0,
    engine_concentration: float = 0.0,
) -> DayQuality:
    """Score overall day quality from market context.

    Components (sum to 100):
      - Breadth (25 pts): % of universe above SMA20
      - Regime (20 pts): bull=20, choppy=10, bear=0
      - Signal density (20 pts): ratio of eligible candidates to universe
      - Top score (25 pts): quality of best candidate
      - Diversity (10 pts): penalize sector/engine concentration

    Args:
        eligible_count: total candidates that passed engine scoring + hard vetoes.
            Must be the UNCAPPED count, not a truncated list size.
    """
    # Breadth: 0-25 pts. Linear from 0.30 (0 pts) to 0.60 (25 pts)
    if breadth_pct >= 0.60:
        breadth_score = 25.0
    elif breadth_pct <= 0.30:
        breadth_score = 0.0
    else:
        breadth_score = (breadth_pct - 0.30) / 0.30 * 25.0

    # Regime: 0-20 pts
    regime_scores = {"bull": 20.0, "choppy": 10.0, "bear": 0.0}
    regime_score = regime_scores.get(regime, 5.0)

    # Signal density: 0-20 pts. ≥2% of universe generating signals = 20
    density = eligible_count / max(universe_size, 1)
    if density >= 0.02:
        density_score = 20.0
    elif density >= 0.005:
        density_score = (density - 0.005) / 0.015 * 20.0
    else:
        density_score = 0.0

    # Top score: 0-25 pts. Linear from 65 (0 pts) to 90 (25 pts)
    if top_score >= 90:
        top_score_pts = 25.0
    elif top_score <= 65:
        top_score_pts = 0.0
    else:
        top_score_pts = (top_score - 65) / 25 * 25.0

    # Diversity: 0-10 pts. Penalize if one sector or engine dominates
    # sector_concentration = fraction of candidates from top sector
    # engine_concentration = fraction of candidates from one engine
    diversity_score = 10.0
    if sector_concentration > 0.50:
        diversity_score -= min(5.0, (sector_concentration - 0.50) * 20)
    if engine_concentration > 0.80:
        diversity_score -= min(5.0, (engine_concentration - 0.80) * 25)
    diversity_score = max(0.0, diversity_score)

    total = breadth_score + regime_score + density_score + top_score_pts + diversity_score

    components = {
        "breadth": round(breadth_score, 1),
        "regime": round(regime_score, 1),
        "signal_density": round(density_score, 1),
        "top_score": round(top_score_pts, 1),
        "diversity": round(diversity_score, 1),
    }

    return DayQuality(
        breadth_pct=breadth_pct,
        regime=regime,
        signal_density=round(density, 4),
        top_score=top_score,
        score=round(total, 1),
        components=components,
    )


def classify_tier(score: float, components: dict) -> str:
    """Classify a candidate into A/B/C tier.

    Tier A: score >= 80 AND all component scores >= 50 (conviction pick)
    Tier B: score >= 72 AND at least 3 components >= 50 (solid setup)
    Tier C: everything else that passed engine floors (marginal)
    """
    component_values = list(components.values())
    above_50_count = sum(1 for v in component_values if v >= 50)

    if score >= 80 and all(v >= 50 for v in component_values):
        return "A"
    elif score >= 72 and above_50_count >= 3:
        return "B"
    return "C"


def _compute_concentration(
    candidates: list[tuple[str, object]],
    info_map: dict[str, dict] | None = None,
) -> tuple[float, float]:
    """Compute sector and engine concentration from the full candidate set.

    Returns (sector_concentration, engine_concentration) as fractions.
    """
    if not candidates:
        return 0.0, 0.0

    n = len(candidates)

    # Engine concentration: fraction from most common engine
    engine_counts = Counter(engine for engine, _ in candidates)
    engine_concentration = max(engine_counts.values()) / n

    # Sector concentration: fraction from most common sector
    if info_map:
        sectors = [
            (info_map.get(sig.ticker, {}) or {}).get("industry", "unknown") or "unknown"
            for _, sig in candidates
        ]
        sector_counts = Counter(sectors)
        sector_concentration = max(sector_counts.values()) / n
    else:
        sector_concentration = 0.0

    return sector_concentration, engine_concentration


def run_acceptance(
    candidates: list[tuple[str, object]],
    breadth_pct: float,
    regime: str,
    universe_size: int,
    config: dict | None = None,
    info_map: dict[str, dict] | None = None,
) -> AcceptanceResult:
    """Run the acceptance layer on the FULL eligible candidate set.

    This is the allocator — it decides book size. Callers must NOT
    truncate the candidate list before passing it here.

    Args:
        candidates: FULL list of (engine, signal) tuples, sorted by score desc.
            Post-dedupe, post-score-floor, post-hard-vetoes (limit-down, sector cap).
        breadth_pct: fraction of universe above SMA20
        regime: market regime string
        universe_size: total tickers in universe (for density calculation)
        config: optional acceptance config overrides
        info_map: ticker -> metadata dict (for sector concentration)

    Returns:
        AcceptanceResult with accepted picks, rejected picks, and day quality
    """
    cfg = config or {}
    # Day quality thresholds
    dq_full_threshold = float(cfg.get("dq_full_threshold", 55))
    dq_selective_threshold = float(cfg.get("dq_selective_threshold", 35))
    # Max picks per quality band
    max_full = int(cfg.get("max_full", 5))
    max_selective = int(cfg.get("max_selective", 2))

    eligible_count = len(candidates)

    # Compute concentration from full candidate set
    sector_conc, engine_conc = _compute_concentration(candidates, info_map)

    # Score day quality using the FULL eligible count
    top_score = max((sig.score for _, sig in candidates), default=0)
    dq = score_day_quality(
        breadth_pct=breadth_pct,
        regime=regime,
        eligible_count=eligible_count,
        universe_size=universe_size,
        top_score=top_score,
        sector_concentration=sector_conc,
        engine_concentration=engine_conc,
    )

    # Classify all candidates
    tiered: list[tuple[str, object, str]] = []
    for engine, sig in candidates:
        tier = classify_tier(sig.score, sig.components)
        tiered.append((engine, sig, tier))

    accepted = []
    rejected = []

    if dq.score < dq_selective_threshold:
        # Full abstention
        mode = "abstain"
        for engine, sig, tier in tiered:
            rejected.append((engine, sig, tier, f"day_quality={dq.score:.0f} < {dq_selective_threshold}"))
        logger.info("Acceptance: ABSTAIN (day_quality=%.0f < %.0f, eligible=%d)",
                     dq.score, dq_selective_threshold, eligible_count)

    elif dq.score < dq_full_threshold:
        # Selective mode: only A-tier, capped
        mode = "selective"
        limit = max_selective
        for engine, sig, tier in tiered:
            if tier == "A" and len(accepted) < limit:
                accepted.append((engine, sig, tier))
            else:
                reason = f"selective_mode: tier={tier}" if tier != "A" else f"selective_cap ({limit})"
                rejected.append((engine, sig, tier, reason))
        logger.info("Acceptance: SELECTIVE (day_quality=%.0f, eligible=%d) — %d A-tier accepted",
                     dq.score, eligible_count, len(accepted))

    else:
        # Full mode: A + B tiers, capped
        mode = "full"
        limit = max_full
        for engine, sig, tier in tiered:
            if tier in ("A", "B") and len(accepted) < limit:
                accepted.append((engine, sig, tier))
            elif tier == "C":
                rejected.append((engine, sig, tier, "tier_C_rejected"))
            else:
                rejected.append((engine, sig, tier, f"cap ({limit})"))
        logger.info("Acceptance: FULL (day_quality=%.0f, eligible=%d) — %d accepted",
                     dq.score, eligible_count, len(accepted))

    return AcceptanceResult(
        day_quality=dq,
        mode=mode,
        accepted=accepted,
        rejected=rejected,
        abstained=len(accepted) == 0,
        eligible_count=eligible_count,
    )
