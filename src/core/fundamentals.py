"""FMP fundamentals enrichment for KooCore-D weekly scanner.

Sync requests client. Fetches company profiles + key metrics from FMP
to add fundamental context to LLM scoring packets.

Budget-guarded: `fmp_fundamentals_max_tickers` limits API calls per run.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"


def _api_key() -> str | None:
    return os.getenv("FMP_API_KEY")


def fetch_fundamentals_batch(
    tickers: list[str],
    api_key: Optional[str] = None,
    max_tickers: int = 25,
) -> dict[str, dict]:
    """Fetch company profiles + key metrics for a batch of tickers.

    Returns {ticker: {profile + metrics}} for each ticker that succeeds.
    Budget-guarded: only fetches up to `max_tickers` symbols.
    Returns {} when API key is missing.
    """
    key = api_key or _api_key()
    if not key:
        return {}

    # Budget guard
    batch = tickers[:max_tickers]
    if len(tickers) > max_tickers:
        logger.info(
            "FMP fundamentals: capped to %d/%d tickers",
            max_tickers, len(tickers),
        )

    results: dict[str, dict] = {}
    for ticker in batch:
        try:
            profile = _fetch_profile(ticker, key)
            metrics = _fetch_key_metrics(ticker, key)
            if profile:
                combined = {**profile}
                if metrics:
                    combined["key_metrics"] = metrics
                combined["fundamental_score"] = compute_fundamental_score(combined)
                results[ticker] = combined
        except Exception as e:
            logger.debug("FMP fundamentals error for %s: %s", ticker, e)
            continue

    logger.info("FMP fundamentals: fetched %d/%d tickers", len(results), len(batch))
    return results


def _fetch_profile(ticker: str, api_key: str) -> dict | None:
    """Fetch company profile from FMP."""
    try:
        resp = requests.get(
            f"{FMP_BASE}/profile",
            params={"apikey": api_key, "symbol": ticker},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _fetch_key_metrics(ticker: str, api_key: str) -> dict | None:
    """Fetch key financial metrics from FMP."""
    try:
        resp = requests.get(
            f"{FMP_BASE}/key-metrics",
            params={"apikey": api_key, "symbol": ticker, "period": "annual"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def compute_fundamental_score(profile: dict) -> float:
    """Compute a 0-10 fundamental quality score from FMP profile + metrics.

    Factors:
    - Revenue growth (from profile)
    - Profit margin
    - P/E relative to reasonable range
    - Market cap tier
    """
    score = 5.0  # neutral baseline

    # Market cap tier bonus
    mktcap = profile.get("mktCap") or profile.get("marketCap") or 0
    if mktcap > 50_000_000_000:      # mega cap
        score += 1.0
    elif mktcap > 10_000_000_000:    # large cap
        score += 0.5

    # Profit margin
    margin = profile.get("lastDiv")  # FMP profile doesn't have margin directly
    # Try key_metrics if available
    metrics = profile.get("key_metrics", {})

    # P/E reasonableness (if available)
    pe = None
    for key in ("peRatio", "priceEarningsRatio"):
        val = metrics.get(key) if metrics else None
        if val is None:
            val = profile.get(key)
        if val is not None:
            try:
                pe = float(val)
                break
            except (ValueError, TypeError):
                pass

    if pe is not None:
        if 5 < pe < 20:
            score += 1.5  # value sweet spot
        elif 20 <= pe < 35:
            score += 0.5  # reasonable growth
        elif pe > 60 or pe < 0:
            score -= 1.0  # expensive or unprofitable

    # Revenue growth (from key metrics)
    rev_growth = metrics.get("revenuePerShare") if metrics else None
    if rev_growth is not None:
        try:
            rg = float(rev_growth)
            if rg > 0:
                score += 0.5
        except (ValueError, TypeError):
            pass

    # ROE (return on equity)
    roe = metrics.get("roe") if metrics else None
    if roe is not None:
        try:
            roe_val = float(roe)
            if roe_val > 0.15:
                score += 1.0
            elif roe_val > 0.10:
                score += 0.5
        except (ValueError, TypeError):
            pass

    return round(max(0.0, min(10.0, score)), 1)
