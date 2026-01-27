"""
Formatting Utilities

Safe HTML escaping, number/date/unit formatting.
"""

from __future__ import annotations
import html
from typing import Any, Optional
import pandas as pd


def escape_html(text: Any) -> str:
    """Safely escape HTML special characters."""
    if text is None:
        return ""
    return html.escape(str(text))


def fmt_num(x: Any, default: str = "—", decimals: int = 2) -> str:
    """Format number with commas and optional decimals."""
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return "1" if x else "0"
        val = float(x)
        if decimals == 0:
            return f"{val:,.0f}"
        return f"{val:,.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return default


def fmt_pct(x: Any, default: str = "—", decimals: int = 1) -> str:
    """Format percentage."""
    try:
        if x is None:
            return default
        val = float(x)
        return f"{val:.{decimals}f}%"
    except Exception:
        return default


def fmt_currency(x: Any, default: str = "—") -> str:
    """Format currency (assumes USD)."""
    try:
        if x is None:
            return default
        val = float(x)
        if val >= 1_000_000_000:
            return f"${val/1_000_000_000:.2f}B"
        elif val >= 1_000_000:
            return f"${val/1_000_000:.2f}M"
        elif val >= 1_000:
            return f"${val/1_000:.2f}K"
        else:
            return f"${val:.2f}"
    except Exception:
        return default


def fmt_date(date_str: Optional[str], default: str = "—") -> str:
    """Format date string for display."""
    if not date_str or date_str == "Unknown":
        return default
    try:
        # Try parsing ISO format
        if "T" in date_str:
            dt = pd.to_datetime(date_str.split("T")[0])
            return dt.strftime("%Y-%m-%d")
        dt = pd.to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return date_str if date_str else default


def fmt_verdict_badge(verdict: Optional[str]) -> tuple[str, str]:
    """
    Format verdict as badge HTML.
    
    Returns:
        Tuple of (badge_class, badge_text)
    """
    if not verdict:
        return ("badge", "UNKNOWN")
    
    verdict_upper = str(verdict).upper().strip()
    
    if verdict_upper == "BUY":
        return ("badge badge-buy", "BUY")
    elif verdict_upper == "WATCH":
        return ("badge badge-watch", "WATCH")
    elif verdict_upper == "IGNORE":
        return ("badge badge-ignore", "IGNORE")
    else:
        return ("badge", verdict_upper)


def fmt_confidence_badge(confidence: Optional[str]) -> tuple[str, str]:
    """
    Format confidence level as badge HTML.
    
    Returns:
        Tuple of (badge_class, badge_text)
    """
    if not confidence:
        return ("badge", "UNKNOWN")
    
    conf_upper = str(confidence).upper().strip()
    
    if "HIGH" in conf_upper:
        return ("badge badge-high", conf_upper)
    elif "MEDIUM" in conf_upper or "MODERATE" in conf_upper:
        return ("badge badge-medium", conf_upper)
    elif "SPECULATIVE" in conf_upper or "LOW" in conf_upper:
        return ("badge badge-speculative", conf_upper)
    else:
        return ("badge", conf_upper)

