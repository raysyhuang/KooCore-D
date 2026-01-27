"""
FastAPI Dashboard for Momentum Trading System

Provides:
- REST API for accessing scan results
- Real-time WebSocket updates
- Simple web dashboard
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from src.utils.time import utc_now

# Check for FastAPI availability
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")


if FASTAPI_AVAILABLE:
    
    # ----- Pydantic Models -----
    
    class HealthResponse(BaseModel):
        status: str
        timestamp: str
        version: str
    
    class RunSummary(BaseModel):
        date: str
        weekly_top5_count: int
        pro30_candidates_count: int
        movers_count: int
        has_overlaps: bool
    
    class TickerDetail(BaseModel):
        ticker: str
        rank: Optional[int] = None
        composite_score: Optional[float] = None
        technical_score: Optional[float] = None
        catalyst_score: Optional[float] = None
        current_price: Optional[float] = None
        target_price: Optional[float] = None
        confidence: Optional[str] = None
    
    class HybridAnalysis(BaseModel):
        date: str
        summary: dict
        overlaps: dict
        weekly_top5: List[dict]
        pro30_tickers: List[str]
        movers_tickers: List[str]
    
    class PerformanceMetrics(BaseModel):
        total_trades: int
        win_rate: float
        hit_10pct_rate: float
        total_pnl_dollars: float
        avg_pnl_percent: float
    
    
    # ----- Helper Functions -----
    
    def get_outputs_dir() -> Path:
        """Get outputs directory path."""
        return Path("outputs")
    
    def get_available_dates() -> list[str]:
        """Get list of available output dates."""
        outputs_dir = get_outputs_dir()
        if not outputs_dir.exists():
            return []
        
        dates = []
        for p in outputs_dir.iterdir():
            if p.is_dir() and len(p.name) == 10 and p.name[4] == '-':
                dates.append(p.name)
        
        return sorted(dates, reverse=True)
    
    def load_hybrid_analysis(date_str: str) -> Optional[dict]:
        """Load hybrid analysis JSON for a date."""
        path = get_outputs_dir() / date_str / f"hybrid_analysis_{date_str}.json"
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load hybrid analysis: {e}")
            return None
    
    def load_weekly_top5(date_str: str) -> Optional[dict]:
        """Load weekly top5 JSON for a date."""
        path = get_outputs_dir() / date_str / f"weekly_scanner_top5_{date_str}.json"
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load weekly top5: {e}")
            return None
    
    
    # ----- FastAPI App -----
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        logger.info("Starting Momentum Trading System API...")
        yield
        logger.info("Shutting down API...")
    
    
    def create_app(
        title: str = "Momentum Trading System API",
        version: str = "1.0.0",
        cors_origins: list[str] = None,
    ) -> FastAPI:
        """
        Create FastAPI application.
        
        Args:
            title: API title
            version: API version
            cors_origins: List of allowed CORS origins
        
        Returns:
            FastAPI application instance
        """
        app = FastAPI(
            title=title,
            version=version,
            description="REST API for the Momentum Trading System",
            lifespan=lifespan,
        )
        
        # CORS middleware
        if cors_origins is None:
            cors_origins = ["http://localhost:3000", "http://localhost:8000"]
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # ----- Routes -----
        
        @app.get("/", response_class=HTMLResponse)
        async def root():
            """Dashboard home page."""
            dates = get_available_dates()[:10]
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Momentum Trading System</title>
                <style>
                    :root {
                        --bg: #0b1020;
                        --card: rgba(255,255,255,0.06);
                        --text: rgba(255,255,255,0.92);
                        --accent: #7dd3fc;
                    }
                    body {
                        font-family: system-ui, sans-serif;
                        background: var(--bg);
                        color: var(--text);
                        margin: 0;
                        padding: 20px;
                    }
                    .container { max-width: 1200px; margin: 0 auto; }
                    h1 { color: var(--accent); }
                    .card {
                        background: var(--card);
                        border-radius: 12px;
                        padding: 20px;
                        margin: 15px 0;
                    }
                    a { color: var(--accent); text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    .date-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
                    .date-item {
                        background: rgba(125,211,252,0.1);
                        padding: 10px 15px;
                        border-radius: 8px;
                        text-align: center;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 Momentum Trading System</h1>
                    
                    <div class="card">
                        <h2>Recent Scans</h2>
                        <div class="date-list">
            """
            
            for d in dates:
                html += f'<a href="/runs/{d}" class="date-item">{d}</a>'
            
            if not dates:
                html += '<p>No scan results found. Run <code>python main.py all</code> to generate data.</p>'
            
            html += """
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>API Endpoints</h2>
                        <ul>
                            <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                            <li><a href="/health">/health</a> - Health check</li>
                            <li><a href="/runs">/runs</a> - List available runs</li>
                            <li>/runs/{date} - Get run details</li>
                            <li>/runs/{date}/top5 - Weekly Top 5</li>
                            <li>/runs/{date}/pro30 - Pro30 candidates</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return HTMLResponse(content=html)
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=utc_now().isoformat().replace("+00:00", ""),
                version=version,
            )
        
        @app.get("/runs")
        async def list_runs(
            limit: int = Query(default=30, ge=1, le=100),
            offset: int = Query(default=0, ge=0),
        ):
            """List available scan runs."""
            dates = get_available_dates()
            
            paginated = dates[offset:offset + limit]
            
            summaries = []
            for d in paginated:
                hybrid = load_hybrid_analysis(d)
                if hybrid:
                    summary = hybrid.get("summary", {})
                    overlaps = hybrid.get("overlaps", {})
                    has_overlaps = any(len(v) > 0 for v in overlaps.values())
                    
                    summaries.append(RunSummary(
                        date=d,
                        weekly_top5_count=summary.get("weekly_top5_count", 0),
                        pro30_candidates_count=summary.get("pro30_candidates_count", 0),
                        movers_count=summary.get("movers_count", 0),
                        has_overlaps=has_overlaps,
                    ))
            
            return {
                "total": len(dates),
                "limit": limit,
                "offset": offset,
                "runs": summaries,
            }
        
        @app.get("/runs/{date_str}")
        async def get_run(date_str: str):
            """Get full hybrid analysis for a date."""
            hybrid = load_hybrid_analysis(date_str)
            
            if not hybrid:
                raise HTTPException(status_code=404, detail=f"Run not found: {date_str}")
            
            return hybrid
        
        @app.get("/runs/{date_str}/top5", response_model=List[TickerDetail])
        async def get_weekly_top5(date_str: str):
            """Get weekly Top 5 for a date."""
            top5_data = load_weekly_top5(date_str)
            
            if not top5_data:
                # Try from hybrid analysis
                hybrid = load_hybrid_analysis(date_str)
                if hybrid and hybrid.get("weekly_top5"):
                    top5_data = {"top5": hybrid["weekly_top5"]}
            
            if not top5_data:
                raise HTTPException(status_code=404, detail=f"Top 5 not found: {date_str}")
            
            top5_list = top5_data.get("top5", [])
            
            results = []
            for item in top5_list:
                scores = item.get("scores", {})
                target = item.get("target", {})
                
                results.append(TickerDetail(
                    ticker=item.get("ticker", ""),
                    rank=item.get("rank"),
                    composite_score=item.get("composite_score"),
                    technical_score=scores.get("technical"),
                    catalyst_score=scores.get("catalyst"),
                    current_price=item.get("current_price"),
                    target_price=target.get("target_price_for_10pct"),
                    confidence=item.get("confidence"),
                ))
            
            return results
        
        @app.get("/runs/{date_str}/pro30")
        async def get_pro30_candidates(date_str: str):
            """Get Pro30 candidates for a date."""
            hybrid = load_hybrid_analysis(date_str)
            
            if not hybrid:
                raise HTTPException(status_code=404, detail=f"Run not found: {date_str}")
            
            return {
                "date": date_str,
                "pro30_tickers": hybrid.get("pro30_tickers", []),
                "count": len(hybrid.get("pro30_tickers", [])),
            }
        
        @app.get("/runs/{date_str}/overlaps")
        async def get_overlaps(date_str: str):
            """Get overlap analysis for a date."""
            hybrid = load_hybrid_analysis(date_str)
            
            if not hybrid:
                raise HTTPException(status_code=404, detail=f"Run not found: {date_str}")
            
            return {
                "date": date_str,
                "overlaps": hybrid.get("overlaps", {}),
            }
        
        @app.get("/tickers/{ticker}")
        async def get_ticker_history(
            ticker: str,
            limit: int = Query(default=10, ge=1, le=50),
        ):
            """Get historical appearances of a ticker across scans."""
            ticker = ticker.upper()
            dates = get_available_dates()[:50]  # Check last 50 runs
            
            appearances = []
            for d in dates:
                hybrid = load_hybrid_analysis(d)
                if not hybrid:
                    continue
                
                in_weekly = False
                weekly_rank = None
                in_pro30 = ticker in hybrid.get("pro30_tickers", [])
                in_movers = ticker in hybrid.get("movers_tickers", [])
                
                for item in hybrid.get("weekly_top5", []):
                    if item.get("ticker") == ticker:
                        in_weekly = True
                        weekly_rank = item.get("rank")
                        break
                
                if in_weekly or in_pro30 or in_movers:
                    appearances.append({
                        "date": d,
                        "in_weekly_top5": in_weekly,
                        "weekly_rank": weekly_rank,
                        "in_pro30": in_pro30,
                        "in_movers": in_movers,
                    })
                
                if len(appearances) >= limit:
                    break
            
            return {
                "ticker": ticker,
                "appearances": appearances,
                "count": len(appearances),
            }
        
        return app
    
    
    def run_server(
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
    ):
        """
        Run the FastAPI server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            reload: Enable auto-reload for development
        """
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed. Install with: pip install uvicorn")
            return
        
        app = create_app()
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
        )

else:
    # Stub implementations when FastAPI is not available
    
    def create_app(*args, **kwargs):
        raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
    
    def run_server(*args, **kwargs):
        raise ImportError("FastAPI not installed. Install with: pip install uvicorn")
