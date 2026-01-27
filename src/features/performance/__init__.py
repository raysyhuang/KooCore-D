"""
Performance / backtesting utilities.

This package is intentionally lightweight and only depends on existing output
artifacts under `outputs/YYYY-MM-DD/`.
"""

from .backtest import (
    DatePicks,
    iter_output_dates,
    load_picks_for_date,
)
from .calibration import (
    CalibrationSuggestion,
    build_calibration_suggestions,
    write_calibration_report,
)

__all__ = [
    "DatePicks",
    "iter_output_dates",
    "load_picks_for_date",
    "CalibrationSuggestion",
    "build_calibration_suggestions",
    "write_calibration_report",
]

