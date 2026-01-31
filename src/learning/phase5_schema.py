"""
Phase-5 Feature Schema (Authoritative)

Canonical data structures for learning signal attribution.
Scan-time only, deterministic, and future-proof.

Design principles:
- Only scan-time data (no future leakage)
- Deterministic & reproducible
- Low dimensionality first (expand later)
- Human-auditable (you must be able to read rows)

Schema size: ~45-55 columns when flattened
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional, Literal
from pathlib import Path


# =============================================================================
# Type Definitions
# =============================================================================

RegimeType = Literal["bull", "chop", "stress"]
StrategyType = Literal["Swing", "Weekly"]
OutcomeType = Literal["hit", "miss", "neutral"]
ExitReasonType = Literal["target_hit", "stop", "expiry", "manual"]
TrendStateType = Literal["strong_up", "weak_up", "flat", "down"]
RSIBucketType = Literal["<40", "40_50", "50_60", "60_70", ">70"]
DistanceBucketType = Literal["near", "mid", "far"]
LiquidityBucketType = Literal["high", "medium", "low"]
VolatilityBucketType = Literal["contracted", "normal", "expanded"]
CatalystType = Literal["earnings", "FDA", "guidance", "macro", "unknown"]


# =============================================================================
# Core Identity (Primary Key)
# =============================================================================

@dataclass
class Phase5Identity:
    """
    Primary key for a Phase-5 row.
    
    Canonical key: (scan_date, ticker, primary_strategy)
    
    One row = one candidate, one scan, one decision.
    """
    scan_date: str              # YYYY-MM-DD (trading date)
    ticker: str                 # Symbol
    primary_strategy: str       # "Swing" | "Weekly"
    regime: str                 # "bull" | "chop" | "stress"
    
    def get_key(self) -> tuple[str, str, str]:
        """Return canonical key for idempotency checks."""
        return (self.scan_date, self.ticker, self.primary_strategy)


# =============================================================================
# Signal Presence & Rank Attribution (MOST IMPORTANT)
# =============================================================================

@dataclass
class Phase5Signals:
    """
    Binary presence flags + ranks for each signal source.
    
    This is where learning power actually comes from.
    Phase-5 learns: "Which source mattered?"
    """
    # Swing signals
    in_swing_top5: bool = False
    swing_rank: Optional[int] = None
    
    # Weekly signals
    in_weekly_top5: bool = False
    weekly_rank: Optional[int] = None
    
    # Pro30 signals (with deterministic ranking)
    in_pro30: bool = False
    pro30_rank: Optional[int] = None
    pro30_score: Optional[float] = None  # Original Pro30 Score
    
    # Movers signals
    in_movers: bool = False
    
    # Confluence (multi-source agreement)
    in_confluence: bool = False
    confluence_score: Optional[int] = None
    
    # Overlap flags (derived from above)
    overlap_primary_pro30: bool = False
    overlap_all_three: bool = False


# =============================================================================
# Hybrid Decision Snapshot (Frozen Belief)
# =============================================================================

@dataclass
class Phase5HybridContext:
    """
    Snapshot of what the system believed at scan-time.
    
    Allows future re-weighting without hindsight bias.
    Lets learning judge: belief vs reality.
    """
    hybrid_score: float
    hybrid_rank: int
    
    # Sources that contributed (e.g., ["Swing(1)", "Pro30(r3)"])
    hybrid_sources: list[str] = field(default_factory=list)
    
    # Frozen config at scan-time
    weights_snapshot: dict = field(default_factory=dict)
    
    # Was this ticker selected for final output?
    in_hybrid_top3: bool = False
    in_conviction_picks: bool = False


# =============================================================================
# Compact Technical Context (Bucketed)
# =============================================================================

@dataclass
class Phase5TechnicalBuckets:
    """
    Coarse technical buckets, not raw indicator values.
    
    Stable across regimes, resistant to overfitting.
    """
    trend_state: str = "flat"           # strong_up | weak_up | flat | down
    rsi_bucket: str = "50_60"           # <40 | 40_50 | 50_60 | 60_70 | >70
    distance_52w_bucket: str = "mid"    # near | mid | far
    liquidity_bucket: str = "medium"    # high | medium | low
    volatility_bucket: str = "normal"   # contracted | normal | expanded
    
    # Raw values (for debugging only, not for learning)
    _raw_rsi: Optional[float] = None
    _raw_atr_pct: Optional[float] = None
    _raw_adv20: Optional[float] = None


# =============================================================================
# Catalyst Presence (Binary, Not Text)
# =============================================================================

@dataclass
class Phase5CatalystFlags:
    """
    Binary catalyst presence flags.
    
    LLM text is unstable; binary presence works statistically.
    """
    has_known_catalyst: bool = False
    catalyst_type: Optional[str] = None  # earnings | FDA | macro | guidance | unknown
    has_earnings_within_7d: bool = False


# =============================================================================
# Outcome Labels (Written Post-Hoc Only)
# =============================================================================

@dataclass
class Phase5Outcome:
    """
    Outcome data, written ONLY after resolution.
    
    Multiple labels enable richer learning.
    """
    # Primary outcome
    outcome_7d: str                     # hit | miss | neutral
    return_7d: float                    # Actual return
    
    # Risk metrics
    max_drawdown_7d: float
    max_gain_7d: Optional[float] = None
    
    # Target tracking
    days_to_target: Optional[int] = None
    exit_reason: str = "expiry"         # target_hit | stop | expiry | manual
    
    # Resolution metadata
    resolved_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None


# =============================================================================
# Full Row Wrapper
# =============================================================================

@dataclass
class Phase5Row:
    """
    Complete Phase-5 learning row.
    
    Serializes cleanly to JSON, CSV, Pandas, SQLite/DuckDB.
    """
    identity: Phase5Identity
    signals: Phase5Signals
    hybrid: Phase5HybridContext
    technicals: Phase5TechnicalBuckets
    catalyst: Phase5CatalystFlags
    outcome: Optional[Phase5Outcome] = None  # None until resolved
    
    # Metadata
    run_id: Optional[str] = None        # For traceability, not uniqueness
    created_at: Optional[str] = None    # ISO timestamp
    
    def get_key(self) -> tuple[str, str, str]:
        """Return canonical key for idempotency checks."""
        return self.identity.get_key()
    
    def is_resolved(self) -> bool:
        """Check if outcome has been resolved."""
        return self.outcome is not None
    
    def to_dict(self) -> dict:
        """Serialize to dictionary (for JSON)."""
        return asdict(self)
    
    def to_flat_dict(self) -> dict:
        """Flatten to single-level dict (for CSV/DataFrame)."""
        flat = {}
        
        # Identity
        flat["scan_date"] = self.identity.scan_date
        flat["ticker"] = self.identity.ticker
        flat["primary_strategy"] = self.identity.primary_strategy
        flat["regime"] = self.identity.regime
        
        # Signals
        flat["in_swing_top5"] = self.signals.in_swing_top5
        flat["swing_rank"] = self.signals.swing_rank
        flat["in_weekly_top5"] = self.signals.in_weekly_top5
        flat["weekly_rank"] = self.signals.weekly_rank
        flat["in_pro30"] = self.signals.in_pro30
        flat["pro30_rank"] = self.signals.pro30_rank
        flat["pro30_score"] = self.signals.pro30_score
        flat["in_movers"] = self.signals.in_movers
        flat["in_confluence"] = self.signals.in_confluence
        flat["confluence_score"] = self.signals.confluence_score
        flat["overlap_primary_pro30"] = self.signals.overlap_primary_pro30
        flat["overlap_all_three"] = self.signals.overlap_all_three
        
        # Hybrid
        flat["hybrid_score"] = self.hybrid.hybrid_score
        flat["hybrid_rank"] = self.hybrid.hybrid_rank
        flat["hybrid_sources"] = ",".join(self.hybrid.hybrid_sources)
        flat["in_hybrid_top3"] = self.hybrid.in_hybrid_top3
        flat["in_conviction_picks"] = self.hybrid.in_conviction_picks
        
        # Technicals
        flat["trend_state"] = self.technicals.trend_state
        flat["rsi_bucket"] = self.technicals.rsi_bucket
        flat["distance_52w_bucket"] = self.technicals.distance_52w_bucket
        flat["liquidity_bucket"] = self.technicals.liquidity_bucket
        flat["volatility_bucket"] = self.technicals.volatility_bucket
        
        # Catalyst
        flat["has_known_catalyst"] = self.catalyst.has_known_catalyst
        flat["catalyst_type"] = self.catalyst.catalyst_type
        flat["has_earnings_within_7d"] = self.catalyst.has_earnings_within_7d
        
        # Outcome (if resolved)
        if self.outcome:
            flat["outcome_7d"] = self.outcome.outcome_7d
            flat["return_7d"] = self.outcome.return_7d
            flat["max_drawdown_7d"] = self.outcome.max_drawdown_7d
            flat["max_gain_7d"] = self.outcome.max_gain_7d
            flat["days_to_target"] = self.outcome.days_to_target
            flat["exit_reason"] = self.outcome.exit_reason
            flat["resolved_date"] = self.outcome.resolved_date
        else:
            flat["outcome_7d"] = None
            flat["return_7d"] = None
            flat["max_drawdown_7d"] = None
            flat["max_gain_7d"] = None
            flat["days_to_target"] = None
            flat["exit_reason"] = None
            flat["resolved_date"] = None
        
        # Metadata
        flat["run_id"] = self.run_id
        flat["created_at"] = self.created_at
        
        return flat
    
    @classmethod
    def from_dict(cls, data: dict) -> "Phase5Row":
        """Deserialize from dictionary."""
        return cls(
            identity=Phase5Identity(**data["identity"]),
            signals=Phase5Signals(**data["signals"]),
            hybrid=Phase5HybridContext(**data["hybrid"]),
            technicals=Phase5TechnicalBuckets(**data["technicals"]),
            catalyst=Phase5CatalystFlags(**data["catalyst"]),
            outcome=Phase5Outcome(**data["outcome"]) if data.get("outcome") else None,
            run_id=data.get("run_id"),
            created_at=data.get("created_at"),
        )


# =============================================================================
# Bucketing Utilities
# =============================================================================

def bucket_rsi(rsi: Optional[float]) -> str:
    """Convert RSI value to bucket."""
    if rsi is None:
        return "50_60"  # Default
    if rsi < 40:
        return "<40"
    elif rsi < 50:
        return "40_50"
    elif rsi < 60:
        return "50_60"
    elif rsi < 70:
        return "60_70"
    else:
        return ">70"


def bucket_distance_52w(dist_pct: Optional[float]) -> str:
    """
    Convert distance to 52-week high (as %) to bucket.
    dist_pct = negative means below high, e.g., -5 means 5% below.
    """
    if dist_pct is None:
        return "mid"
    # Closer to 0 = nearer to 52w high
    abs_dist = abs(dist_pct)
    if abs_dist <= 5:
        return "near"
    elif abs_dist <= 15:
        return "mid"
    else:
        return "far"


def bucket_liquidity(adv20: Optional[float]) -> str:
    """
    Convert 20-day average dollar volume to bucket.
    """
    if adv20 is None:
        return "medium"
    if adv20 >= 50_000_000:  # $50M+
        return "high"
    elif adv20 >= 10_000_000:  # $10M+
        return "medium"
    else:
        return "low"


def bucket_volatility(atr_pct: Optional[float]) -> str:
    """
    Convert ATR% to volatility bucket.
    """
    if atr_pct is None:
        return "normal"
    if atr_pct < 2.0:
        return "contracted"
    elif atr_pct < 5.0:
        return "normal"
    else:
        return "expanded"


def derive_trend_state(
    above_ma20: bool,
    above_ma50: bool,
    ret_20d: Optional[float] = None
) -> str:
    """
    Derive trend state from MA alignment and momentum.
    """
    if above_ma20 and above_ma50:
        if ret_20d is not None and ret_20d > 10:
            return "strong_up"
        return "weak_up"
    elif above_ma20 or above_ma50:
        return "flat"
    else:
        return "down"


# =============================================================================
# Rank Bucketing (for learning features)
# =============================================================================

def bucket_pro30_rank(rank: Optional[int]) -> str:
    """Convert Pro30 rank to bucket for learning."""
    if rank is None:
        return "none"
    if rank <= 1:
        return "r1"
    elif rank <= 3:
        return "r2_3"
    elif rank <= 5:
        return "r4_5"
    elif rank <= 10:
        return "r6_10"
    elif rank <= 20:
        return "r11_20"
    else:
        return "r21+"


def bucket_primary_rank(rank: Optional[int]) -> str:
    """Convert primary (Swing/Weekly) rank to bucket."""
    if rank is None:
        return "none"
    if rank == 1:
        return "r1"
    elif rank == 2:
        return "r2"
    elif rank == 3:
        return "r3"
    elif rank <= 5:
        return "r4_5"
    else:
        return "r6+"


# =============================================================================
# Validation
# =============================================================================

REQUIRED_IDENTITY_FIELDS = {"scan_date", "ticker", "primary_strategy", "regime"}
VALID_REGIMES = {"bull", "chop", "stress"}
VALID_STRATEGIES = {"Swing", "Weekly"}
VALID_OUTCOMES = {"hit", "miss", "neutral"}


def validate_row(row: Phase5Row) -> list[str]:
    """
    Validate a Phase5Row and return list of errors (empty if valid).
    """
    errors = []
    
    # Identity validation
    if not row.identity.scan_date:
        errors.append("scan_date is required")
    if not row.identity.ticker:
        errors.append("ticker is required")
    if row.identity.primary_strategy not in VALID_STRATEGIES:
        errors.append(f"primary_strategy must be one of {VALID_STRATEGIES}")
    if row.identity.regime not in VALID_REGIMES:
        errors.append(f"regime must be one of {VALID_REGIMES}")
    
    # Outcome validation (if present)
    if row.outcome:
        if row.outcome.outcome_7d not in VALID_OUTCOMES:
            errors.append(f"outcome_7d must be one of {VALID_OUTCOMES}")
    
    return errors


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

def row_to_jsonl(row: Phase5Row) -> str:
    """Serialize row to JSONL (single line)."""
    return json.dumps(row.to_dict(), separators=(",", ":"))


def rows_to_jsonl(rows: list[Phase5Row]) -> str:
    """Serialize multiple rows to JSONL format."""
    return "\n".join(row_to_jsonl(row) for row in rows)


def row_from_jsonl(line: str) -> Phase5Row:
    """Deserialize row from JSONL line."""
    data = json.loads(line.strip())
    return Phase5Row.from_dict(data)


def load_jsonl_file(path: Path) -> list[Phase5Row]:
    """Load all rows from a JSONL file."""
    rows = []
    if not path.exists():
        return rows
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(row_from_jsonl(line))
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    # Log but don't fail on malformed lines
                    pass
    return rows


def save_jsonl_file(rows: list[Phase5Row], path: Path, append: bool = True):
    """Save rows to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(row_to_jsonl(row) + "\n")
