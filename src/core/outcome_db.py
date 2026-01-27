"""
Outcome Database Module

SQLite storage for picks and their outcomes to enable learning from historical performance.
Stores all features at pick time and tracks final outcomes for model improvement.
"""

from __future__ import annotations
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = "data/outcomes.db"


@dataclass
class PickRecord:
    """
    Complete record of a pick at entry time with all features.
    """
    # Core identification
    ticker: str
    pick_date: str  # YYYY-MM-DD
    source: str  # "weekly_top5", "pro30", "movers"
    
    # Entry details
    entry_price: float
    rank: Optional[int] = None  # For weekly picks
    
    # Technical features at pick time
    technical_score: Optional[float] = None
    rsi14: Optional[float] = None
    volume_ratio_3d_to_20d: Optional[float] = None
    dist_to_52w_high_pct: Optional[float] = None
    realized_vol_5d_ann_pct: Optional[float] = None
    above_ma10: Optional[bool] = None
    above_ma20: Optional[bool] = None
    above_ma50: Optional[bool] = None
    
    # Additional context
    sector: Optional[str] = None
    market_cap_usd: Optional[float] = None
    composite_score: Optional[float] = None
    
    # Overlap tracking (how many scanners flagged this ticker)
    overlap_count: int = 1
    overlap_sources: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.overlap_sources:
            self.overlap_sources = [self.source]


@dataclass
class OutcomeRecord:
    """
    Complete outcome record for a closed position.
    Combines entry features with final outcome metrics.
    """
    # Link to pick
    ticker: str
    pick_date: str
    source: str
    
    # Entry features (copied from PickRecord)
    entry_price: float
    rank: Optional[int] = None
    technical_score: Optional[float] = None
    rsi14: Optional[float] = None
    volume_ratio_3d_to_20d: Optional[float] = None
    dist_to_52w_high_pct: Optional[float] = None
    realized_vol_5d_ann_pct: Optional[float] = None
    above_ma10: Optional[bool] = None
    above_ma20: Optional[bool] = None
    above_ma50: Optional[bool] = None
    sector: Optional[str] = None
    market_cap_usd: Optional[float] = None
    composite_score: Optional[float] = None
    overlap_count: int = 1
    overlap_sources: List[str] = field(default_factory=list)
    
    # Outcome metrics
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "target_hit", "stopped", "expired", "manual"
    
    # Performance metrics
    max_price: Optional[float] = None
    max_return_pct: Optional[float] = None
    min_price: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    final_return_pct: Optional[float] = None
    days_held: int = 0
    days_to_peak: Optional[int] = None
    
    # Hit tracking (for model learning)
    hit_5pct: bool = False
    hit_7pct: bool = False
    hit_10pct: bool = False
    hit_15pct: bool = False
    
    # Metadata
    closed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.closed_at is None:
            self.closed_at = datetime.utcnow().isoformat() + "Z"


class OutcomeDatabase:
    """
    SQLite-based storage for picks and outcomes.
    
    Provides efficient querying for feature analysis and model training.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            # Picks table - stores all features at pick time
            conn.execute("""
                CREATE TABLE IF NOT EXISTS picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    pick_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    rank INTEGER,
                    technical_score REAL,
                    rsi14 REAL,
                    volume_ratio_3d_to_20d REAL,
                    dist_to_52w_high_pct REAL,
                    realized_vol_5d_ann_pct REAL,
                    above_ma10 INTEGER,
                    above_ma20 INTEGER,
                    above_ma50 INTEGER,
                    sector TEXT,
                    market_cap_usd REAL,
                    composite_score REAL,
                    overlap_count INTEGER DEFAULT 1,
                    overlap_sources TEXT,
                    created_at TEXT,
                    UNIQUE(ticker, pick_date, source)
                )
            """)
            
            # Outcomes table - stores completed position outcomes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    pick_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    rank INTEGER,
                    technical_score REAL,
                    rsi14 REAL,
                    volume_ratio_3d_to_20d REAL,
                    dist_to_52w_high_pct REAL,
                    realized_vol_5d_ann_pct REAL,
                    above_ma10 INTEGER,
                    above_ma20 INTEGER,
                    above_ma50 INTEGER,
                    sector TEXT,
                    market_cap_usd REAL,
                    composite_score REAL,
                    overlap_count INTEGER DEFAULT 1,
                    overlap_sources TEXT,
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    max_price REAL,
                    max_return_pct REAL,
                    min_price REAL,
                    max_drawdown_pct REAL,
                    final_return_pct REAL,
                    days_held INTEGER,
                    days_to_peak INTEGER,
                    hit_5pct INTEGER DEFAULT 0,
                    hit_7pct INTEGER DEFAULT 0,
                    hit_10pct INTEGER DEFAULT 0,
                    hit_15pct INTEGER DEFAULT 0,
                    closed_at TEXT,
                    UNIQUE(ticker, pick_date, source)
                )
            """)
            
            # Indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(pick_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_ticker ON picks(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_picks_source ON picks(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_date ON outcomes(pick_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_source ON outcomes(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_hit7 ON outcomes(hit_7pct)")
            
            conn.commit()
            logger.debug(f"Initialized outcome database at {self.db_path}")
    
    def store_pick(self, pick: PickRecord) -> bool:
        """
        Store a pick record with all features.
        
        Returns True if stored, False if already exists.
        """
        try:
            with self._get_conn() as conn:
                overlap_sources_json = json.dumps(pick.overlap_sources) if pick.overlap_sources else "[]"
                
                conn.execute("""
                    INSERT OR REPLACE INTO picks (
                        ticker, pick_date, source, entry_price, rank,
                        technical_score, rsi14, volume_ratio_3d_to_20d,
                        dist_to_52w_high_pct, realized_vol_5d_ann_pct,
                        above_ma10, above_ma20, above_ma50,
                        sector, market_cap_usd, composite_score,
                        overlap_count, overlap_sources, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pick.ticker, pick.pick_date, pick.source, pick.entry_price, pick.rank,
                    pick.technical_score, pick.rsi14, pick.volume_ratio_3d_to_20d,
                    pick.dist_to_52w_high_pct, pick.realized_vol_5d_ann_pct,
                    1 if pick.above_ma10 else 0 if pick.above_ma10 is not None else None,
                    1 if pick.above_ma20 else 0 if pick.above_ma20 is not None else None,
                    1 if pick.above_ma50 else 0 if pick.above_ma50 is not None else None,
                    pick.sector, pick.market_cap_usd, pick.composite_score,
                    pick.overlap_count, overlap_sources_json, pick.created_at
                ))
                conn.commit()
                logger.debug(f"Stored pick: {pick.ticker} ({pick.pick_date}, {pick.source})")
                return True
        except sqlite3.IntegrityError:
            logger.debug(f"Pick already exists: {pick.ticker} ({pick.pick_date}, {pick.source})")
            return False
        except Exception as e:
            logger.error(f"Failed to store pick: {e}")
            return False
    
    def store_picks_batch(self, picks: List[PickRecord]) -> int:
        """Store multiple picks efficiently. Returns count stored."""
        stored = 0
        for pick in picks:
            if self.store_pick(pick):
                stored += 1
        return stored
    
    def record_outcome(self, outcome: OutcomeRecord) -> bool:
        """
        Record the outcome for a completed position.
        
        Returns True if recorded successfully.
        """
        try:
            with self._get_conn() as conn:
                overlap_sources_json = json.dumps(outcome.overlap_sources) if outcome.overlap_sources else "[]"
                
                conn.execute("""
                    INSERT OR REPLACE INTO outcomes (
                        ticker, pick_date, source, entry_price, rank,
                        technical_score, rsi14, volume_ratio_3d_to_20d,
                        dist_to_52w_high_pct, realized_vol_5d_ann_pct,
                        above_ma10, above_ma20, above_ma50,
                        sector, market_cap_usd, composite_score,
                        overlap_count, overlap_sources,
                        exit_date, exit_price, exit_reason,
                        max_price, max_return_pct, min_price, max_drawdown_pct,
                        final_return_pct, days_held, days_to_peak,
                        hit_5pct, hit_7pct, hit_10pct, hit_15pct, closed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.ticker, outcome.pick_date, outcome.source, outcome.entry_price, outcome.rank,
                    outcome.technical_score, outcome.rsi14, outcome.volume_ratio_3d_to_20d,
                    outcome.dist_to_52w_high_pct, outcome.realized_vol_5d_ann_pct,
                    1 if outcome.above_ma10 else 0 if outcome.above_ma10 is not None else None,
                    1 if outcome.above_ma20 else 0 if outcome.above_ma20 is not None else None,
                    1 if outcome.above_ma50 else 0 if outcome.above_ma50 is not None else None,
                    outcome.sector, outcome.market_cap_usd, outcome.composite_score,
                    outcome.overlap_count, overlap_sources_json,
                    outcome.exit_date, outcome.exit_price, outcome.exit_reason,
                    outcome.max_price, outcome.max_return_pct, outcome.min_price, outcome.max_drawdown_pct,
                    outcome.final_return_pct, outcome.days_held, outcome.days_to_peak,
                    1 if outcome.hit_5pct else 0,
                    1 if outcome.hit_7pct else 0,
                    1 if outcome.hit_10pct else 0,
                    1 if outcome.hit_15pct else 0,
                    outcome.closed_at
                ))
                conn.commit()
                logger.info(f"Recorded outcome: {outcome.ticker} ({outcome.pick_date}) - hit_7pct={outcome.hit_7pct}")
                return True
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return False
    
    def get_pick(self, ticker: str, pick_date: str, source: str) -> Optional[PickRecord]:
        """Retrieve a specific pick record."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM picks 
                WHERE ticker = ? AND pick_date = ? AND source = ?
            """, (ticker, pick_date, source)).fetchone()
            
            if row:
                return self._row_to_pick(row)
            return None
    
    def get_outcome(self, ticker: str, pick_date: str, source: str) -> Optional[OutcomeRecord]:
        """Retrieve a specific outcome record."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM outcomes 
                WHERE ticker = ? AND pick_date = ? AND source = ?
            """, (ticker, pick_date, source)).fetchone()
            
            if row:
                return self._row_to_outcome(row)
            return None
    
    def get_training_data(
        self,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get outcome data as DataFrame for model training.
        
        Args:
            min_date: Minimum pick date (inclusive)
            max_date: Maximum pick date (inclusive)
            sources: Filter by source types
        
        Returns:
            DataFrame with all outcome records matching criteria
        """
        query = "SELECT * FROM outcomes WHERE 1=1"
        params = []
        
        if min_date:
            query += " AND pick_date >= ?"
            params.append(min_date)
        if max_date:
            query += " AND pick_date <= ?"
            params.append(max_date)
        if sources:
            placeholders = ",".join("?" * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)
        
        query += " ORDER BY pick_date DESC"
        
        with self._get_conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Convert boolean columns
        bool_cols = ["above_ma10", "above_ma20", "above_ma50", 
                     "hit_5pct", "hit_7pct", "hit_10pct", "hit_15pct"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Parse overlap_sources JSON
        if "overlap_sources" in df.columns:
            df["overlap_sources"] = df["overlap_sources"].apply(
                lambda x: json.loads(x) if x else []
            )
        
        return df
    
    def get_picks_without_outcomes(self, before_date: Optional[str] = None) -> List[PickRecord]:
        """
        Get picks that don't have recorded outcomes yet.
        
        Useful for finding positions that need outcome recording.
        """
        query = """
            SELECT p.* FROM picks p
            LEFT JOIN outcomes o ON p.ticker = o.ticker 
                AND p.pick_date = o.pick_date AND p.source = o.source
            WHERE o.id IS NULL
        """
        params = []
        
        if before_date:
            query += " AND p.pick_date < ?"
            params.append(before_date)
        
        query += " ORDER BY p.pick_date DESC"
        
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        
        return [self._row_to_pick(row) for row in rows]
    
    def get_outcome_stats(self) -> Dict[str, Any]:
        """Get summary statistics about outcomes."""
        with self._get_conn() as conn:
            total_picks = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
            total_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            
            if total_outcomes > 0:
                hit_7_count = conn.execute(
                    "SELECT COUNT(*) FROM outcomes WHERE hit_7pct = 1"
                ).fetchone()[0]
                hit_10_count = conn.execute(
                    "SELECT COUNT(*) FROM outcomes WHERE hit_10pct = 1"
                ).fetchone()[0]
                
                avg_return = conn.execute(
                    "SELECT AVG(max_return_pct) FROM outcomes WHERE max_return_pct IS NOT NULL"
                ).fetchone()[0] or 0
                
                avg_days = conn.execute(
                    "SELECT AVG(days_held) FROM outcomes WHERE days_held IS NOT NULL"
                ).fetchone()[0] or 0
                
                # By source breakdown
                source_stats = {}
                for row in conn.execute("""
                    SELECT source, 
                           COUNT(*) as count,
                           SUM(hit_7pct) as hits_7,
                           AVG(max_return_pct) as avg_return
                    FROM outcomes 
                    GROUP BY source
                """).fetchall():
                    source_stats[row["source"]] = {
                        "count": row["count"],
                        "hit_7pct_rate": row["hits_7"] / row["count"] if row["count"] > 0 else 0,
                        "avg_return": row["avg_return"] or 0
                    }
            else:
                hit_7_count = 0
                hit_10_count = 0
                avg_return = 0
                avg_days = 0
                source_stats = {}
        
        return {
            "total_picks": total_picks,
            "total_outcomes": total_outcomes,
            "pending_outcomes": total_picks - total_outcomes,
            "hit_7pct_rate": hit_7_count / total_outcomes if total_outcomes > 0 else 0,
            "hit_10pct_rate": hit_10_count / total_outcomes if total_outcomes > 0 else 0,
            "avg_max_return_pct": avg_return,
            "avg_days_held": avg_days,
            "by_source": source_stats,
        }
    
    def _row_to_pick(self, row: sqlite3.Row) -> PickRecord:
        """Convert database row to PickRecord."""
        overlap_sources = json.loads(row["overlap_sources"]) if row["overlap_sources"] else []
        
        return PickRecord(
            ticker=row["ticker"],
            pick_date=row["pick_date"],
            source=row["source"],
            entry_price=row["entry_price"],
            rank=row["rank"],
            technical_score=row["technical_score"],
            rsi14=row["rsi14"],
            volume_ratio_3d_to_20d=row["volume_ratio_3d_to_20d"],
            dist_to_52w_high_pct=row["dist_to_52w_high_pct"],
            realized_vol_5d_ann_pct=row["realized_vol_5d_ann_pct"],
            above_ma10=bool(row["above_ma10"]) if row["above_ma10"] is not None else None,
            above_ma20=bool(row["above_ma20"]) if row["above_ma20"] is not None else None,
            above_ma50=bool(row["above_ma50"]) if row["above_ma50"] is not None else None,
            sector=row["sector"],
            market_cap_usd=row["market_cap_usd"],
            composite_score=row["composite_score"],
            overlap_count=row["overlap_count"] or 1,
            overlap_sources=overlap_sources,
            created_at=row["created_at"],
        )
    
    def _row_to_outcome(self, row: sqlite3.Row) -> OutcomeRecord:
        """Convert database row to OutcomeRecord."""
        overlap_sources = json.loads(row["overlap_sources"]) if row["overlap_sources"] else []
        
        return OutcomeRecord(
            ticker=row["ticker"],
            pick_date=row["pick_date"],
            source=row["source"],
            entry_price=row["entry_price"],
            rank=row["rank"],
            technical_score=row["technical_score"],
            rsi14=row["rsi14"],
            volume_ratio_3d_to_20d=row["volume_ratio_3d_to_20d"],
            dist_to_52w_high_pct=row["dist_to_52w_high_pct"],
            realized_vol_5d_ann_pct=row["realized_vol_5d_ann_pct"],
            above_ma10=bool(row["above_ma10"]) if row["above_ma10"] is not None else None,
            above_ma20=bool(row["above_ma20"]) if row["above_ma20"] is not None else None,
            above_ma50=bool(row["above_ma50"]) if row["above_ma50"] is not None else None,
            sector=row["sector"],
            market_cap_usd=row["market_cap_usd"],
            composite_score=row["composite_score"],
            overlap_count=row["overlap_count"] or 1,
            overlap_sources=overlap_sources,
            exit_date=row["exit_date"],
            exit_price=row["exit_price"],
            exit_reason=row["exit_reason"],
            max_price=row["max_price"],
            max_return_pct=row["max_return_pct"],
            min_price=row["min_price"],
            max_drawdown_pct=row["max_drawdown_pct"],
            final_return_pct=row["final_return_pct"],
            days_held=row["days_held"] or 0,
            days_to_peak=row["days_to_peak"],
            hit_5pct=bool(row["hit_5pct"]),
            hit_7pct=bool(row["hit_7pct"]),
            hit_10pct=bool(row["hit_10pct"]),
            hit_15pct=bool(row["hit_15pct"]),
            closed_at=row["closed_at"],
        )


# Global instance
_db_instance: Optional[OutcomeDatabase] = None


def get_outcome_db(db_path: str = DB_PATH) -> OutcomeDatabase:
    """Get or create the global OutcomeDatabase instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = OutcomeDatabase(db_path)
    return _db_instance
