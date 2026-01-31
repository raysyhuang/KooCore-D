"""
Phase-5 Data Store

JSONL-based append-only storage with idempotency guarantees.

Storage layout:
    outputs/phase5/
        rows/phase5_rows_YYYY-MM-DD.jsonl     # Scan-time features
        outcomes/phase5_outcomes_YYYY-MM-DD.jsonl  # Post-hoc outcomes
        merged/phase5_merged.parquet          # Training dataset
        metrics/phase5_scorecards_YYYY-MM-DD.json  # Analysis results
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple

from src.learning.phase5_schema import (
    Phase5Row,
    Phase5Outcome,
    row_to_jsonl,
    row_from_jsonl,
    load_jsonl_file,
    validate_row,
)

logger = logging.getLogger(__name__)

# Default base path
DEFAULT_BASE_PATH = Path("outputs/phase5")


class Phase5Store:
    """
    Manages Phase-5 learning data with idempotency guarantees.
    
    Ground rules:
    - Scan-time features only go into rows (no future leakage)
    - Outcomes are appended later (resolved jobs)
    - Retries may recompute, but must not double-write
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = Path(base_path) if base_path else DEFAULT_BASE_PATH
        self.rows_dir = self.base_path / "rows"
        self.outcomes_dir = self.base_path / "outcomes"
        self.merged_dir = self.base_path / "merged"
        self.metrics_dir = self.base_path / "metrics"
        
        # Ensure directories exist
        for d in [self.rows_dir, self.outcomes_dir, self.merged_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Row Storage (Scan-Time Features)
    # =========================================================================
    
    def get_rows_file(self, scan_date: str) -> Path:
        """Get the JSONL file path for a given scan date."""
        return self.rows_dir / f"phase5_rows_{scan_date}.jsonl"
    
    def get_existing_keys(self, scan_date: str) -> Set[Tuple[str, str, str]]:
        """Load existing keys for a scan date (for idempotency check)."""
        path = self.get_rows_file(scan_date)
        keys = set()
        
        if not path.exists():
            return keys
        
        rows = load_jsonl_file(path)
        for row in rows:
            keys.add(row.get_key())
        
        return keys
    
    def write_rows(
        self,
        rows: list[Phase5Row],
        scan_date: str,
        dry_run: bool = False
    ) -> dict:
        """
        Write rows to JSONL with idempotency.
        
        Returns dict with counts: written, skipped, errors
        """
        result = {"written": 0, "skipped": 0, "errors": 0, "dry_run": dry_run}
        
        if not rows:
            return result
        
        # Load existing keys for this date
        existing_keys = self.get_existing_keys(scan_date)
        
        # Filter to new rows only
        new_rows = []
        for row in rows:
            # Validate
            errors = validate_row(row)
            if errors:
                logger.warning(f"Validation errors for {row.identity.ticker}: {errors}")
                result["errors"] += 1
                continue
            
            key = row.get_key()
            if key in existing_keys:
                logger.debug(f"Skipping duplicate: {key}")
                result["skipped"] += 1
            else:
                new_rows.append(row)
                existing_keys.add(key)  # Prevent duplicates within batch
        
        if dry_run:
            result["written"] = len(new_rows)
            logger.info(f"[DRY RUN] Would write {len(new_rows)} rows to {scan_date}")
            return result
        
        # Write new rows
        if new_rows:
            path = self.get_rows_file(scan_date)
            with open(path, "a", encoding="utf-8") as f:
                for row in new_rows:
                    f.write(row_to_jsonl(row) + "\n")
                    result["written"] += 1
            
            logger.info(f"Wrote {result['written']} Phase-5 rows to {path.name}")
        
        return result
    
    def load_rows(self, scan_date: Optional[str] = None) -> list[Phase5Row]:
        """
        Load rows from storage.
        
        If scan_date is None, loads all rows.
        """
        if scan_date:
            return load_jsonl_file(self.get_rows_file(scan_date))
        
        # Load all
        all_rows = []
        for path in sorted(self.rows_dir.glob("phase5_rows_*.jsonl")):
            all_rows.extend(load_jsonl_file(path))
        
        return all_rows
    
    # =========================================================================
    # Outcome Storage (Post-Hoc Resolution)
    # =========================================================================
    
    def get_outcomes_file(self, scan_date: str) -> Path:
        """Get the outcomes JSONL file for a scan date."""
        return self.outcomes_dir / f"phase5_outcomes_{scan_date}.jsonl"
    
    def get_existing_outcome_keys(self, scan_date: str) -> Set[Tuple[str, str, str]]:
        """Load existing outcome keys for a scan date."""
        path = self.get_outcomes_file(scan_date)
        keys = set()
        
        if not path.exists():
            return keys
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        key = (data["scan_date"], data["ticker"], data["primary_strategy"])
                        keys.add(key)
                    except (json.JSONDecodeError, KeyError):
                        pass
        
        return keys
    
    def write_outcomes(
        self,
        outcomes: list[dict],
        scan_date: str,
        dry_run: bool = False
    ) -> dict:
        """
        Write outcomes to JSONL with idempotency.
        
        Each outcome dict must have:
        - scan_date, ticker, primary_strategy (key)
        - outcome_7d, return_7d, max_drawdown_7d, etc.
        """
        result = {"written": 0, "skipped": 0, "errors": 0, "dry_run": dry_run}
        
        if not outcomes:
            return result
        
        existing_keys = self.get_existing_outcome_keys(scan_date)
        
        new_outcomes = []
        for outcome in outcomes:
            try:
                key = (outcome["scan_date"], outcome["ticker"], outcome["primary_strategy"])
                if key in existing_keys:
                    result["skipped"] += 1
                else:
                    new_outcomes.append(outcome)
                    existing_keys.add(key)
            except KeyError as e:
                logger.warning(f"Outcome missing key field: {e}")
                result["errors"] += 1
        
        if dry_run:
            result["written"] = len(new_outcomes)
            return result
        
        if new_outcomes:
            path = self.get_outcomes_file(scan_date)
            with open(path, "a", encoding="utf-8") as f:
                for outcome in new_outcomes:
                    f.write(json.dumps(outcome, separators=(",", ":")) + "\n")
                    result["written"] += 1
            
            logger.info(f"Wrote {result['written']} outcomes to {path.name}")
        
        return result
    
    def load_outcomes(self, scan_date: Optional[str] = None) -> list[dict]:
        """Load outcomes from storage."""
        outcomes = []
        
        if scan_date:
            paths = [self.get_outcomes_file(scan_date)]
        else:
            paths = sorted(self.outcomes_dir.glob("phase5_outcomes_*.jsonl"))
        
        for path in paths:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            outcomes.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        
        return outcomes
    
    # =========================================================================
    # Merged Dataset
    # =========================================================================
    
    def get_merged_path(self) -> Path:
        """Get path to merged parquet file."""
        return self.merged_dir / "phase5_merged.parquet"
    
    def merge_rows_and_outcomes(self) -> int:
        """
        Merge all rows with their outcomes into a single parquet file.
        
        Returns number of rows in merged dataset.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for merge operation")
            return 0
        
        # Load all rows
        all_rows = self.load_rows()
        if not all_rows:
            logger.warning("No rows to merge")
            return 0
        
        # Convert to flat dicts
        rows_data = [row.to_flat_dict() for row in all_rows]
        df = pd.DataFrame(rows_data)
        
        # Load all outcomes
        all_outcomes = self.load_outcomes()
        
        # Build outcome lookup
        outcome_map = {}
        for outcome in all_outcomes:
            key = (outcome["scan_date"], outcome["ticker"], outcome["primary_strategy"])
            outcome_map[key] = outcome
        
        # Merge outcomes into rows
        for idx, row in df.iterrows():
            key = (row["scan_date"], row["ticker"], row["primary_strategy"])
            if key in outcome_map:
                outcome = outcome_map[key]
                df.at[idx, "outcome_7d"] = outcome.get("outcome_7d")
                df.at[idx, "return_7d"] = outcome.get("return_7d")
                df.at[idx, "max_drawdown_7d"] = outcome.get("max_drawdown_7d")
                df.at[idx, "max_gain_7d"] = outcome.get("max_gain_7d")
                df.at[idx, "days_to_target"] = outcome.get("days_to_target")
                df.at[idx, "exit_reason"] = outcome.get("exit_reason")
                df.at[idx, "resolved_date"] = outcome.get("resolved_date")
        
        # Save
        path = self.get_merged_path()
        df.to_parquet(path, index=False)
        
        logger.info(f"Merged {len(df)} rows to {path}")
        
        return len(df)
    
    def load_merged(self):
        """Load merged parquet as DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required")
            return None
        
        path = self.get_merged_path()
        if not path.exists():
            logger.warning(f"Merged file not found: {path}")
            return pd.DataFrame()
        
        return pd.read_parquet(path)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        rows_files = list(self.rows_dir.glob("phase5_rows_*.jsonl"))
        outcome_files = list(self.outcomes_dir.glob("phase5_outcomes_*.jsonl"))
        
        total_rows = 0
        total_outcomes = 0
        
        for f in rows_files:
            total_rows += sum(1 for line in open(f) if line.strip())
        
        for f in outcome_files:
            total_outcomes += sum(1 for line in open(f) if line.strip())
        
        merged_path = self.get_merged_path()
        merged_rows = 0
        if merged_path.exists():
            try:
                import pandas as pd
                merged_rows = len(pd.read_parquet(merged_path))
            except Exception:
                pass
        
        return {
            "rows_files": len(rows_files),
            "total_rows": total_rows,
            "outcome_files": len(outcome_files),
            "total_outcomes": total_outcomes,
            "merged_rows": merged_rows,
            "resolution_rate": total_outcomes / max(total_rows, 1),
        }


# =============================================================================
# Singleton Access
# =============================================================================

_store_instance: Optional[Phase5Store] = None


def get_phase5_store(base_path: Optional[Path] = None) -> Phase5Store:
    """Get or create the Phase5Store singleton."""
    global _store_instance
    if _store_instance is None or base_path is not None:
        _store_instance = Phase5Store(base_path)
    return _store_instance


# =============================================================================
# Legacy Compatibility (deprecated)
# =============================================================================

def persist_learning(records, base_path="learning/phase5"):
    """
    DEPRECATED: Use Phase5Store.write_rows() instead.
    
    Kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "persist_learning is deprecated, use Phase5Store.write_rows()",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not records:
        return
    
    from src.utils.time import utc_now
    
    Path(base_path).mkdir(parents=True, exist_ok=True)
    fname = f"{utc_now().strftime('%Y-%m-%d')}.json"
    path = Path(base_path) / fname
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
