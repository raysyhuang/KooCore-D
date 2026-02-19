"""Engine endpoint — standardized /api/engine/results + /api/engine/ingest.

Maps KooCore-D hybrid analysis to the EngineResultPayload contract.
Heroku filesystem is ephemeral, so GitHub Actions pushes results via /api/engine/ingest,
and /api/engine/results serves the latest stored results from memory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/engine", tags=["engine"])

# In-memory store for latest results (Heroku ephemeral filesystem)
_latest_result: dict | None = None


class EnginePick(BaseModel):
    ticker: str
    strategy: str
    entry_price: float
    stop_loss: float | None = None
    target_price: float | None = None
    confidence: float
    holding_period_days: int
    thesis: str | None = None
    risk_factors: list[str] = []
    raw_score: float | None = None
    metadata: dict = {}


class EngineResultPayload(BaseModel):
    engine_name: str
    engine_version: str
    run_date: str
    run_timestamp: str
    regime: str | None = None
    picks: list[EnginePick]
    candidates_screened: int
    pipeline_duration_s: float | None = None
    status: str = "success"


class IngestPayload(BaseModel):
    """Raw hybrid analysis data pushed from GitHub Actions."""
    hybrid_analysis: dict
    run_date: str
    pipeline_duration_s: float | None = None


def _map_hybrid_to_payload(hybrid: dict, run_date: str, duration: float | None = None) -> dict:
    """Convert KooCore-D hybrid_analysis JSON to EngineResultPayload format."""
    picks = []

    # Map hybrid_top3 (best picks across all models)
    hybrid_top3 = hybrid.get("hybrid_top3", [])
    weighted_picks = hybrid.get("weighted_picks", [])

    for item in hybrid_top3:
        ticker = item.get("ticker", "")
        composite_score = item.get("composite_score") or item.get("hybrid_score", 0)
        sources = item.get("sources", [])
        strategy = "hybrid_" + "_".join(sources) if sources else "hybrid"

        # Normalize confidence: composite_score is 0-10, multiply by 10
        confidence = min(max(composite_score * 10, 0), 100)

        picks.append({
            "ticker": ticker,
            "strategy": strategy,
            "entry_price": item.get("current_price", 0),
            "stop_loss": None,
            "target_price": item.get("target", {}).get("target_price_for_10pct"),
            "confidence": round(confidence, 1),
            "holding_period_days": 14,  # KooCore-D typical hold: 7-30d
            "thesis": item.get("verdict") or item.get("confidence"),
            "risk_factors": [],
            "raw_score": composite_score,
            "metadata": {
                "sources": sources,
                "rank": item.get("rank"),
                "scores": item.get("scores", {}),
            },
        })

    # Also include additional weighted_picks not already in top3
    top3_tickers = {p["ticker"] for p in picks}
    for item in weighted_picks:
        ticker = item.get("ticker", "")
        if ticker in top3_tickers:
            continue
        composite_score = item.get("hybrid_score", 0)
        if composite_score < 3:  # Skip low-scoring picks
            break
        sources = item.get("sources", [])
        confidence = min(max(composite_score * 10, 0), 100)
        picks.append({
            "ticker": ticker,
            "strategy": "swing" if "pro30" in str(sources) else "momentum",
            "entry_price": item.get("current_price", 0),
            "stop_loss": None,
            "target_price": None,
            "confidence": round(confidence, 1),
            "holding_period_days": 14,
            "thesis": None,
            "risk_factors": [],
            "raw_score": composite_score,
            "metadata": {"sources": sources},
        })
        if len(picks) >= 10:
            break

    summary = hybrid.get("summary", {})
    total_screened = (
        summary.get("weekly_top5_count", 0)
        + summary.get("pro30_candidates_count", 0)
        + summary.get("movers_count", 0)
    )

    return {
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": run_date,
        "run_timestamp": datetime.utcnow().isoformat(),
        "regime": None,  # KooCore-D uses regime-gating internally
        "picks": picks,
        "candidates_screened": total_screened,
        "pipeline_duration_s": duration,
        "status": "success",
    }


@router.get("/results")
async def get_engine_results():
    """Return latest engine results in standardized format."""
    if _latest_result:
        return _latest_result

    # Fallback: try to read from filesystem (local dev)
    from pathlib import Path
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        dates = sorted(
            [d.name for d in outputs_dir.iterdir() if d.is_dir() and len(d.name) == 10],
            reverse=True,
        )
        for date_str in dates:
            hybrid_path = outputs_dir / date_str / f"hybrid_analysis_{date_str}.json"
            if hybrid_path.exists():
                with open(hybrid_path) as f:
                    hybrid = json.load(f)
                return _map_hybrid_to_payload(hybrid, date_str)

    raise HTTPException(404, "No engine results available")


@router.post("/ingest")
async def ingest_results(
    payload: IngestPayload,
    x_engine_key: Optional[str] = Header(None),
):
    """Receive results pushed from GitHub Actions after scan completes."""
    import os

    expected_key = os.environ.get("ENGINE_API_KEY", "")
    if expected_key and x_engine_key != expected_key:
        raise HTTPException(403, "Invalid API key")

    global _latest_result

    _latest_result = _map_hybrid_to_payload(
        payload.hybrid_analysis,
        payload.run_date,
        payload.pipeline_duration_s,
    )

    logger.info(
        "Ingested KooCore-D results for %s: %d picks",
        payload.run_date, len(_latest_result.get("picks", [])),
    )
    return {"status": "ok", "picks_count": len(_latest_result.get("picks", []))}
