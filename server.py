"""
Flask server to serve the HTML dashboard and provide GitHub API endpoints.
"""
import os
import io
import json
import zipfile
from datetime import datetime, timedelta
from flask import Flask, send_from_directory, jsonify, request
import requests
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Configuration
CORE_OWNER = os.getenv("CORE_OWNER", "raysyhuang")
CORE_REPO = os.getenv("CORE_REPO", "KooCore-D")
CORE_ARTIFACT_NAME = os.getenv("CORE_ARTIFACT_NAME", "koocore-outputs")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

# Simple in-memory cache
_cache = {
    'data': None,
    'timestamp': None,
    'ttl': 300  # 5 minutes
}

# In-memory latest standardized engine result for cross-engine collection
_engine_result_cache = {
    'data': None,
    'timestamp': None,
}


def _map_hybrid_to_engine_payload(hybrid: dict, run_date: str, duration: float | None = None) -> dict:
    """Map KooCore hybrid_analysis payload to standardized engine contract."""
    picks = []
    hybrid_top3 = hybrid.get("hybrid_top3", [])
    weighted_picks = hybrid.get("weighted_picks", [])

    for item in hybrid_top3:
        composite_score = item.get("composite_score") or item.get("hybrid_score", 0)
        confidence = max(0.0, min(float(composite_score) * 10.0, 100.0))
        picks.append({
            "ticker": item.get("ticker", ""),
            "strategy": "hybrid",
            "entry_price": float(item.get("current_price") or 0),
            "stop_loss": None,
            "target_price": (item.get("target") or {}).get("target_price_for_10pct"),
            "confidence": round(confidence, 1),
            "holding_period_days": 14,
            "thesis": item.get("verdict") or item.get("confidence"),
            "risk_factors": [],
            "raw_score": composite_score,
            "metadata": {
                "sources": item.get("sources", []),
                "rank": item.get("rank"),
            },
        })

    seen = {p["ticker"] for p in picks}
    for item in weighted_picks:
        ticker = item.get("ticker", "")
        if not ticker or ticker in seen:
            continue
        score = float(item.get("hybrid_score") or 0)
        if score < 3.0:
            break
        confidence = max(0.0, min(score * 10.0, 100.0))
        picks.append({
            "ticker": ticker,
            "strategy": "hybrid_weighted",
            "entry_price": float(item.get("current_price") or 0),
            "stop_loss": None,
            "target_price": None,
            "confidence": round(confidence, 1),
            "holding_period_days": 14,
            "thesis": None,
            "risk_factors": [],
            "raw_score": score,
            "metadata": {"sources": item.get("sources", [])},
        })
        seen.add(ticker)
        if len(picks) >= 10:
            break

    summary = hybrid.get("summary", {})
    screened = (
        int(summary.get("weekly_top5_count", 0))
        + int(summary.get("pro30_candidates_count", 0))
        + int(summary.get("movers_count", 0))
    )

    return {
        "engine_name": "koocore_d",
        "engine_version": "2.0",
        "run_date": run_date,
        "run_timestamp": datetime.utcnow().isoformat(),
        "regime": None,
        "picks": picks,
        "candidates_screened": screened,
        "pipeline_duration_s": duration,
        "status": "success",
    }

def _gh_headers():
    """Get GitHub API headers with authentication."""
    hdr = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        hdr["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdr


def _load_latest_engine_result_from_github():
    """Fetch latest hybrid_analysis artifact and map to engine payload."""
    try:
        url = f"https://api.github.com/repos/{CORE_OWNER}/{CORE_REPO}/actions/artifacts?per_page=100"
        r = requests.get(url, headers=_gh_headers(), timeout=20)
        r.raise_for_status()

        arts = r.json().get("artifacts", [])
        candidates = [a for a in arts if a.get("name") == CORE_ARTIFACT_NAME and not a.get("expired", False)]
        if not candidates:
            return None

        candidates.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        art = candidates[0]

        dl_url = art["archive_download_url"]
        r2 = requests.get(dl_url, headers=_gh_headers(), timeout=60)
        r2.raise_for_status()

        newest_hybrid = None
        newest_name = ""
        with zipfile.ZipFile(io.BytesIO(r2.content), "r") as z:
            for info in z.infolist():
                name = info.filename
                if "hybrid_analysis_" not in name or not name.endswith(".json"):
                    continue
                if name > newest_name:
                    newest_name = name
                    newest_hybrid = json.loads(z.read(name).decode("utf-8"))

        if not newest_hybrid:
            return None

        run_date = (
            newest_hybrid.get("asof_trading_date")
            or newest_hybrid.get("date")
            or datetime.utcnow().strftime("%Y-%m-%d")
        )
        run_date = str(run_date).split("T")[0]
        return _map_hybrid_to_engine_payload(newest_hybrid, run_date)
    except Exception:
        return None


def fetch_latest_github_data():
    """Fetch latest artifact from GitHub Actions and extract picks data."""
    try:
        # Check cache first
        if _cache['data'] and _cache['timestamp']:
            if datetime.now() - _cache['timestamp'] < timedelta(seconds=_cache['ttl']):
                return _cache['data']
        
        # List artifacts
        url = f"https://api.github.com/repos/{CORE_OWNER}/{CORE_REPO}/actions/artifacts?per_page=100"
        r = requests.get(url, headers=_gh_headers(), timeout=20)
        r.raise_for_status()
        
        arts = r.json().get("artifacts", [])
        candidates = [a for a in arts if a.get("name") == CORE_ARTIFACT_NAME and not a.get("expired", False)]
        
        if not candidates:
            return {"error": f"No artifact named '{CORE_ARTIFACT_NAME}' found"}
        
        candidates.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        art = candidates[0]
        
        # Download artifact
        dl_url = art["archive_download_url"]
        r2 = requests.get(dl_url, headers=_gh_headers(), timeout=60)
        r2.raise_for_status()
        
        # Extract ZIP
        picks_data = {}
        meta = {
            "artifact_id": art.get("id"),
            "created_at": art.get("created_at"),
            "size_bytes": art.get("size_in_bytes", 0),
        }
        
        with zipfile.ZipFile(io.BytesIO(r2.content), "r") as z:
            # Look for hybrid_analysis files
            for info in z.infolist():
                if "hybrid_analysis" in info.filename and info.filename.endswith(".json"):
                    try:
                        content = z.read(info.filename)
                        data = json.loads(content.decode("utf-8"))
                        date = data.get("date") or data.get("asof_trading_date")
                        
                        if date and date not in picks_data:
                            picks_data[date] = {
                                "weekly": [],
                                "pro30": [],
                                "movers": []
                            }
                            
                            # Extract tickers from different sources
                            # Weekly/Primary top5
                            primary = data.get("primary_top5", data.get("weekly_top5", []))
                            for item in primary:
                                ticker = item.get("ticker", item) if isinstance(item, dict) else item
                                if ticker:
                                    picks_data[date]["weekly"].append(ticker)
                            
                            # Pro30
                            pro30_tickers = data.get("pro30_tickers", [])
                            picks_data[date]["pro30"].extend(pro30_tickers)
                            
                            # Movers
                            movers_tickers = data.get("movers_tickers", [])
                            picks_data[date]["movers"].extend(movers_tickers)
                    except Exception as e:
                        continue
        
        result = {
            "picks_data": picks_data,
            "meta": meta,
            "last_updated": datetime.now().isoformat()
        }
        
        # Update cache
        _cache['data'] = result
        _cache['timestamp'] = datetime.now()
        
        return result
        
    except Exception as e:
        return {"error": str(e), "message": "Failed to fetch from GitHub API"}

@app.route('/')
def index():
    """Serve the main dashboard HTML file."""
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/picks')
def api_picks():
    """API endpoint to get latest picks data from GitHub."""
    data = fetch_latest_github_data()
    return jsonify(data)

@app.route('/api/status')
def api_status():
    """API endpoint to check connection status."""
    status = {
        "connected": bool(GITHUB_TOKEN),
        "owner": CORE_OWNER,
        "repo": CORE_REPO,
        "artifact": CORE_ARTIFACT_NAME,
        "cache_ttl": _cache['ttl'],
        "last_fetch": _cache['timestamp'].isoformat() if _cache['timestamp'] else None
    }
    return jsonify(status)

@app.route('/api/prices', methods=['POST'])
def api_prices():
    """Fetch real stock prices for performance tracking."""
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not tickers or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Fetch prices using yfinance
        tickers_str = ' '.join(tickers)
        df = yf.download(
            tickers=tickers_str,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=True,
            group_by='column'
        )
        
        if df.empty:
            return jsonify({"error": "No price data available"}), 404
        
        # Extract close prices
        if len(tickers) == 1:
            # Single ticker
            close_data = {tickers[0]: df['Close'].to_dict()}
        else:
            # Multiple tickers
            close_df = df['Close']
            close_data = {}
            for ticker in tickers:
                if ticker in close_df.columns:
                    close_data[ticker] = close_df[ticker].dropna().to_dict()
        
        # Convert timestamps to strings
        result = {}
        for ticker, prices in close_data.items():
            result[ticker] = {
                str(date.date() if hasattr(date, 'date') else date): float(price) 
                for date, price in prices.items()
                if pd.notna(price)
            }
        
        return jsonify({
            "tickers": result,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(result)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/engine/ingest', methods=['POST'])
def api_engine_ingest():
    """Receive standardized ingestion payload from GitHub Actions."""
    expected_key = os.getenv("ENGINE_API_KEY", "").strip()
    if not expected_key:
        return jsonify({"error": "ENGINE_API_KEY not configured"}), 503

    provided_key = request.headers.get("X-Engine-Key", "")
    if provided_key != expected_key:
        return jsonify({"error": "Invalid API key"}), 403

    payload = request.get_json(silent=True) or {}
    hybrid = payload.get("hybrid_analysis")
    run_date = payload.get("run_date")
    duration = payload.get("pipeline_duration_s")

    if not isinstance(hybrid, dict) or not run_date:
        return jsonify({"error": "Invalid payload"}), 400

    _engine_result_cache["data"] = _map_hybrid_to_engine_payload(hybrid, str(run_date), duration)
    _engine_result_cache["timestamp"] = datetime.utcnow().isoformat()

    return jsonify({
        "status": "ok",
        "picks_count": len(_engine_result_cache["data"].get("picks", [])),
    })


@app.route('/api/engine/results', methods=['GET'])
@app.route('/api/engine/results/latest', methods=['GET'])
def api_engine_results():
    """Serve latest standardized engine result for external collection."""
    data = _engine_result_cache.get("data")
    if not data:
        fallback = _load_latest_engine_result_from_github()
        if fallback:
            _engine_result_cache["data"] = fallback
            _engine_result_cache["timestamp"] = datetime.utcnow().isoformat()
            data = fallback
    if not data:
        return jsonify({"error": "No engine results available"}), 404
    return jsonify(data)

@app.route('/<path:path>')
def serve_static(path):
    """Serve any other static files if needed."""
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
