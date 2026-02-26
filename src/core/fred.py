"""FRED macro data client — VIX, yield curve, credit spreads for KooCore-D.

Sync requests client matching KooCore-D's fully synchronous patterns.
Used for multi-factor regime detection (supplements yfinance SPY/VIX).
"""

import logging
import os
from datetime import date, timedelta

import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "vix": "VIXCLS",
    "yield_spread": "T10Y2Y",       # 10Y-2Y spread
    "fed_funds": "FEDFUNDS",
    "credit_spread": "BAA10Y",  # BAA-10Y corporate credit spread
}


def _api_key() -> str | None:
    return os.getenv("FRED_API_KEY")


def get_series(series_id: str, lookback_days: int = 30) -> float | None:
    """Fetch the latest value of a FRED series.

    Returns the most recent non-null observation as a float,
    or None on failure / missing key.
    """
    key = _api_key()
    if not key:
        return None

    from_date = date.today() - timedelta(days=lookback_days)

    try:
        resp = requests.get(
            FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
                "observation_start": str(from_date),
                "observation_end": str(date.today()),
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("FRED %s HTTP %d", series_id, resp.status_code)
            return None

        observations = resp.json().get("observations", [])
        for obs in reversed(observations):
            val = obs.get("value", ".")
            if val != ".":
                return float(val)
        return None
    except Exception as e:
        logger.warning("FRED %s error: %s", series_id, e)
        return None


def get_macro_snapshot() -> dict:
    """Return a macro snapshot dict with VIX, yield spread, fed funds, credit spread.

    Always returns a dict (possibly with None values) — never raises.
    Each field is fetched independently so a single failure doesn't block the rest.
    """
    vix = get_series(SERIES["vix"])
    spread = get_series(SERIES["yield_spread"])
    fed_funds = get_series(SERIES["fed_funds"])
    credit = get_series(SERIES["credit_spread"])

    snapshot = {
        "vix": vix,
        "yield_spread": spread,
        "fed_funds": fed_funds,
        "credit_spread": credit,
        "yield_curve_inverted": spread is not None and spread < 0,
        "credit_stress": credit is not None and credit > 2.0,
    }
    logger.info(
        "FRED macro: VIX=%s spread=%s fed_funds=%s credit=%s",
        f"{vix:.1f}" if vix else "N/A",
        f"{spread:.2f}" if spread else "N/A",
        f"{fed_funds:.2f}" if fed_funds else "N/A",
        f"{credit:.2f}" if credit else "N/A",
    )
    return snapshot
