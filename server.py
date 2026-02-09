"""
Flask server to serve the HTML dashboard and provide GitHub API endpoints.
"""
import os
import io
import json
import zipfile
from datetime import datetime, timedelta
from flask import Flask, send_from_directory, jsonify
import requests

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

def _gh_headers():
    """Get GitHub API headers with authentication."""
    hdr = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        hdr["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return hdr

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

@app.route('/<path:path>')
def serve_static(path):
    """Serve any other static files if needed."""
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
