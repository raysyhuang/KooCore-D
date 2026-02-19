"""Engine endpoint — standardized /api/engine/results + /api/engine/ingest.

Maps KooCore-D hybrid analysis to the EngineResultPayload contract.
Heroku filesystem is ephemeral, so GitHub Actions pushes results via /api/engine/ingest,
and /api/engine/results serves the latest stored results from memory.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
import requests

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/engine", tags=["engine"])

# In-memory store for latest results (Heroku ephemeral filesystem)
_latest_result: dict | None = None
_LATEST_RESULT_FILE = Path("/tmp/koocore_latest_engine_result.json")
_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _load_latest_from_disk() -> dict | None:
    """Load latest ingested payload from local tmp cache (shared by workers)."""
    try:
        if not _LATEST_RESULT_FILE.exists():
            return None
        with _LATEST_RESULT_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and payload.get("engine_name") == "koocore_d":
            return payload
    except Exception as e:
        logger.warning("Failed to load latest engine payload from disk cache: %s", e)
    return None


def _save_latest_to_disk(payload: dict) -> None:
    """Persist latest payload so all workers serve the same run_date."""
    try:
        _LATEST_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LATEST_RESULT_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.warning("Failed to persist latest engine payload to disk cache: %s", e)


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


def _risk_profile_from_sources(sources: list[str]) -> tuple[float, float, str]:
    """Infer risk/target defaults from contributing source models."""
    src = {str(s).lower() for s in (sources or [])}
    if "pro30" in src:
        return 0.08, 0.18, "momentum"
    if "movers" in src:
        return 0.04, 0.08, "breakout"
    return 0.05, 0.10, "swing"


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
        risk_pct, reward_pct, fallback_strategy = _risk_profile_from_sources(sources)
        strategy = "hybrid_" + "_".join(sources) if sources else fallback_strategy
        entry_price = float(item.get("current_price") or 0)
        if entry_price <= 0:
            continue

        # Normalize confidence: composite_score is 0-10, multiply by 10
        confidence = min(max(composite_score * 10, 0), 100)
        explicit_target = item.get("target", {}).get("target_price_for_10pct")
        target_price = (
            float(explicit_target)
            if explicit_target is not None and float(explicit_target) > entry_price
            else round(entry_price * (1 + reward_pct), 2)
        )

        picks.append({
            "ticker": ticker,
            "strategy": strategy,
            "entry_price": entry_price,
            "stop_loss": round(entry_price * (1 - risk_pct), 2),
            "target_price": target_price,
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
        risk_pct, reward_pct, strategy = _risk_profile_from_sources(sources)
        entry_price = float(item.get("current_price") or 0)
        if entry_price <= 0:
            continue
        confidence = min(max(composite_score * 10, 0), 100)
        picks.append({
            "ticker": ticker,
            "strategy": strategy,
            "entry_price": entry_price,
            "stop_loss": round(entry_price * (1 - risk_pct), 2),
            "target_price": round(entry_price * (1 + reward_pct), 2),
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


def _load_latest_from_github() -> dict | None:
    """Fetch latest hybrid analysis from GitHub outputs directory."""
    repo = os.environ.get("ENGINE_OUTPUTS_REPO", "raysyhuang/KooCore-D")
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # List output date folders.
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/contents/outputs",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json()
        date_dirs = sorted(
            [
                i.get("name", "")
                for i in items
                if i.get("type") == "dir" and _DATE_DIR_RE.match(i.get("name", ""))
            ],
            reverse=True,
        )
        if not date_dirs:
            return None

        run_date = date_dirs[0]
        file_resp = requests.get(
            f"https://api.github.com/repos/{repo}/contents/outputs/{run_date}/hybrid_analysis_{run_date}.json",
            headers=headers,
            timeout=10,
        )
        file_resp.raise_for_status()
        payload = file_resp.json()
        encoded = payload.get("content")
        if not encoded:
            return None
        hybrid = json.loads(base64.b64decode(encoded).decode("utf-8"))
        return _map_hybrid_to_payload(hybrid, run_date)
    except Exception as e:
        logger.warning("GitHub fallback failed: %s", e)
        return None


def _to_legacy_picks_payload(engine_payload: dict) -> dict:
    """Convert standardized engine payload to legacy /api/picks format."""
    run_date = engine_payload.get("run_date") or date.today().isoformat()
    day = {"weekly": [], "pro30": [], "movers": []}
    seen = {k: set() for k in day}

    for pick in engine_payload.get("picks", []):
        ticker = str(pick.get("ticker") or "").upper()
        if not ticker:
            continue
        strategy = str(pick.get("strategy") or "").lower()
        meta = pick.get("metadata") or {}
        sources = {str(s).lower() for s in (meta.get("sources") or [])}

        if "movers" in sources or "breakout" in strategy:
            bucket = "movers"
        elif "pro30" in sources or "momentum" in strategy:
            bucket = "pro30"
        else:
            bucket = "weekly"

        if ticker not in seen[bucket]:
            day[bucket].append(ticker)
            seen[bucket].add(ticker)

    return {"picks_data": {run_date: day}}


@router.get("/results")
async def get_engine_results():
    """Return latest engine results in standardized format."""
    global _latest_result

    # First fallback: worker-shared tmp cache populated by /ingest.
    # Always check disk before in-memory cache so requests served by a
    # different worker process can still observe the latest ingested run.
    disk_payload = _load_latest_from_disk()
    if disk_payload:
        _latest_result = disk_payload
        return _latest_result

    if _latest_result:
        return _latest_result

    # Fallback: try to read from filesystem (local dev)
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

    # Last resort: fetch latest committed outputs from GitHub.
    gh_result = _load_latest_from_github()
    if gh_result:
        _latest_result = gh_result
        _save_latest_to_disk(_latest_result)
        return _latest_result

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
    _save_latest_to_disk(_latest_result)

    logger.info(
        "Ingested KooCore-D results for %s: %d picks",
        payload.run_date, len(_latest_result.get("picks", [])),
    )
    return {"status": "ok", "picks_count": len(_latest_result.get("picks", []))}
