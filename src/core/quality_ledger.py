# src/core/quality_ledger.py
"""
Quality Ledger for tracking data quality through the pipeline.

Records validation stats, rejections, and exceptions for every ticker
processed, enabling audit trails and debugging of data issues.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass
class LedgerRow:
    """A single row in the quality ledger."""
    ticker: str
    stage: str
    provider_used: Optional[str] = None
    rows: Optional[int] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    missing_cols: Optional[str] = None
    dropped_bad_rows: Optional[int] = None
    dropped_future_rows: Optional[int] = None
    reject_reason: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None


class QualityLedger:
    """
    Accumulates quality/validation info for each ticker through pipeline stages.
    
    Usage:
        ledger = QualityLedger()
        ledger.add(LedgerRow(ticker="AAPL", stage="load_prices", ...))
        ledger.write_csv("outputs/quality_ledger.csv")
    """
    
    def __init__(self) -> None:
        self._rows: List[LedgerRow] = []

    def add(self, row: LedgerRow) -> None:
        """Add a ledger row."""
        self._rows.append(row)

    def add_exception(
        self,
        ticker: str,
        stage: str,
        exc: Exception,
        provider_used: Optional[str] = None
    ) -> None:
        """Convenience method to log an exception."""
        self.add(
            LedgerRow(
                ticker=ticker,
                stage=stage,
                provider_used=provider_used,
                exception_type=type(exc).__name__,
                exception_message=str(exc)[:500],
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """Convert ledger to DataFrame."""
        return pd.DataFrame([asdict(r) for r in self._rows])

    def write_csv(self, path: str) -> None:
        """Write ledger to CSV file."""
        df = self.to_frame()
        if df.empty:
            # Write header-only file if empty
            df = pd.DataFrame(columns=[f.name for f in LedgerRow.__dataclass_fields__.values()])
        df.to_csv(path, index=False)
    
    def get_rejection_summary(self) -> Dict[str, int]:
        """Get count of rejections by reason."""
        summary: Dict[str, int] = {}
        for row in self._rows:
            if row.reject_reason:
                summary[row.reject_reason] = summary.get(row.reject_reason, 0) + 1
        return summary
    
    def get_exception_count(self) -> int:
        """Get count of exceptions logged."""
        return sum(1 for r in self._rows if r.exception_type)
    
    def __len__(self) -> int:
        return len(self._rows)
