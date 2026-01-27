"""
Weekly Momentum Scanner Pipeline

Orchestrates the full Weekly Scanner pipeline using core modules.
"""

from __future__ import annotations
import os
from datetime import datetime, timezone
from datetime import date as date_type
from pathlib import Path
from typing import Optional
import pandas as pd
import time

# Core imports
from ..core.config import load_config, get_config_value
from ..core.universe import build_universe
from ..core.yf import download_daily, download_daily_range, download_daily_range_cached, get_ticker_df
from ..core.polygon import download_polygon_batch, fetch_polygon_daily
from ..core.technicals import compute_technicals
from ..core.scoring import compute_technical_score_weekly
from ..core.io import get_run_dir, save_csv, save_json, save_run_metadata
from ..core.helpers import get_ny_date, get_trading_date

# PR1: Data validation and quality ledger
from ..core.asof import validate_ohlcv, enforce_asof
from ..core.quality_ledger import QualityLedger, LedgerRow

# PR2: FeatureSet-based scoring (leak-proof)
from ..core.types import FeatureSet
from ..core.features_compute import compute_features_weekly
from ..core.score_weekly import score_weekly

# PR3: Breakout + momentum normalization + event gate
from ..core.breakout import score_breakout
from ..core.momentum_norm import momentum_atr_adjust
from ..core.event_gate import earnings_proximity_gate

# PR5: Calibration and probability inference
try:
    from ..calibration.snapshot import write_decision_snapshot
    from ..calibration.infer import infer_probability, compute_expected_value, infer_probability_regime
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    write_decision_snapshot = None
    infer_probability = None
    compute_expected_value = None
    infer_probability_regime = None

# PR6: Portfolio construction and sizing
try:
    from ..portfolio.construct import PortfolioConfig, build_trade_plan, write_trade_plan, write_portfolio_summary
    from ..portfolio.sizing import SizingConfig
    from ..portfolio.liquidity import LiquidityConfig
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False
    PortfolioConfig = None
    build_trade_plan = None
    write_trade_plan = None
    write_portfolio_summary = None
    SizingConfig = None
    LiquidityConfig = None

# PR7: Regime classification
try:
    from ..regime.classifier import classify_regime, get_current_regime, Regime
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    classify_regime = None
    get_current_regime = None
    Regime = None

# Governance modules
try:
    from ..governance.artifacts import GovernanceContext, write_governance_record
    from ..governance.performance_monitor import rolling_metrics, check_decay, load_training_metrics, load_recent_trades
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    GovernanceContext = None
    write_governance_record = None
    rolling_metrics = None
    check_decay = None
    load_training_metrics = None
    load_recent_trades = None

# Feature imports
try:
    from ..features.movers.daily_movers import compute_daily_movers_from_universe
    from ..features.movers.mover_filters import filter_movers, build_mover_technicals_df
    from ..features.movers.mover_queue import update_mover_queue, get_eligible_movers, load_mover_queue, save_mover_queue
except ImportError:
    # Fallback if movers not available
    compute_daily_movers_from_universe = None
    filter_movers = None
    update_mover_queue = None
    get_eligible_movers = None
    load_mover_queue = None
    save_mover_queue = None

# Helper imports
from ..core.helpers import fetch_news_for_tickers, get_next_earnings_date, load_manual_headlines, validate_required_columns
from ..core.filters import apply_hard_filters
from ..core.packets import build_weekly_scanner_packet


def run_weekly(
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
    asof_date: Optional[date_type] = None,
    *,
    output_date: Optional[date_type] = None,
    run_dir: Optional[Path] = None,
) -> dict:
    """
    Run the Weekly Momentum Scanner pipeline.
    
    Args:
        config: Optional config dict (if None, loads from config_path)
        config_path: Path to config YAML file (defaults to config/default.yaml)
    
    Returns:
        dict with keys:
          - universe_note: str
          - run_timestamp_utc: str
          - run_dir: Path
          - candidates_csv: Path
          - packets_json: Path
          - metadata_json: Path
    """
    # Load config
    if config is None:
        config = load_config(config_path)
    
    # Separate output folder date from data as-of date:
    # - output_date: where results are written (defaults to NY calendar date)
    # - asof_trading_date: last completed trading day used for downloads/scoring
    if output_date is None:
        output_date = get_ny_date()
    asof_trading_date = get_trading_date(asof_date or output_date)

    root_dir = get_config_value(config, "outputs", "root_dir", default="outputs")
    if run_dir is None:
        run_dir = Path(root_dir) / output_date.strftime("%Y-%m-%d")
        run_dir.mkdir(parents=True, exist_ok=True)

    output_date_str = output_date.strftime("%Y-%m-%d")
    
    # Initialize governance context for audit trail
    governance_flags = []
    calibration_used = False
    calibration_model_version = None
    decay_detected = False
    eligibility_passed = True
    fallback_reason = None
    
    # G5.2: Check for manual override
    disable_calibration = get_config_value(config, "governance", "disable_calibration", default=False)
    if disable_calibration:
        governance_flags.append("calibration_disabled_manual_override")
        fallback_reason = "manual_override"
    
    # PR7: Classify market regime
    current_regime = None
    regime_evidence = {}
    if REGIME_AVAILABLE and get_current_regime:
        try:
            regime_obj = get_current_regime(asof_date=asof_trading_date.strftime("%Y-%m-%d") if asof_trading_date else None)
            current_regime = regime_obj.name
            regime_evidence = regime_obj.evidence or {}
            print(f"  Market regime: {current_regime} (confidence: {regime_obj.confidence:.0%})")
        except Exception as e:
            print(f"  [WARN] Regime classification failed: {e}")
            current_regime = "chop"  # Default to chop
    else:
        current_regime = "chop"
    
    # Build universe
    quarantine_cfg = config.get("data_reliability", {}).get("quarantine", {})
    universe = build_universe(
        mode=get_config_value(config, "universe", "mode", default="SP500+NASDAQ100+R2000"),
        cache_file=get_config_value(config, "universe", "cache_file", default=None),
        cache_max_age_days=get_config_value(config, "universe", "cache_max_age_days", default=7),
        manual_include_file=get_config_value(config, "universe", "manual_include_file", default=None),
        r2000_include_file=get_config_value(config, "universe", "r2000_include_file", default=None),
        manual_include_mode=get_config_value(config, "universe", "manual_include_mode", default="ALWAYS"),
        quarantine_file=quarantine_cfg.get("file", "data/bad_tickers.json"),
        quarantine_enabled=bool(quarantine_cfg.get("enabled", True)),
    )
    
    universe_note = f"Universe: {len(universe)} tickers"
    
    # PR1: Initialize quality ledger for audit trail
    ledger = QualityLedger()
    asof_date_str = asof_trading_date.strftime("%Y-%m-%d") if asof_trading_date else None
    
    # Initialize data provider settings early (needed for movers and main scan)
    runtime = config.get("runtime", {})
    polygon_api_key = os.environ.get("POLYGON_API_KEY") or runtime.get("polygon_api_key")
    use_polygon_primary = bool(runtime.get("polygon_primary", False) and polygon_api_key)
    use_polygon_fallback = bool(runtime.get("polygon_fallback", False) and polygon_api_key)
    polygon_max_workers = int(runtime.get("polygon_max_workers", 8))
    reliability_cfg = config.get("data_reliability", {})
    yf_retry_cfg = reliability_cfg.get("yfinance", {}) if isinstance(reliability_cfg, dict) else {}
    poly_retry_cfg = reliability_cfg.get("polygon", {}) if isinstance(reliability_cfg, dict) else {}
    
    # Handle daily movers if enabled
    mover_source_tags = {}
    if get_config_value(config, "movers", "enabled", default=False) and compute_daily_movers_from_universe:
        movers_config = config.get("movers", {})
        
        try:
            reliability_cfg = config.get("data_reliability", {})
            movers_raw = compute_daily_movers_from_universe(
                universe, 
                top_n=movers_config.get("top_n", 50), 
                asof_date=asof_trading_date,  # Use passed asof_date, not today
                polygon_api_key=polygon_api_key,
                use_polygon_primary=use_polygon_primary,
                polygon_max_workers=polygon_max_workers,
                quarantine_cfg=reliability_cfg.get("quarantine", {}) if isinstance(reliability_cfg, dict) else {},
                yf_retry_cfg=reliability_cfg.get("yfinance", {}) if isinstance(reliability_cfg, dict) else {},
                polygon_retry_cfg=reliability_cfg.get("polygon", {}) if isinstance(reliability_cfg, dict) else {},
            )
            mover_universe = []
            for k in ("gainers", "losers"):
                dfm = movers_raw.get(k)
                if isinstance(dfm, pd.DataFrame) and (not dfm.empty) and "ticker" in dfm.columns:
                    mover_universe += dfm["ticker"].astype(str).tolist()
            tech_df = build_mover_technicals_df(
                mover_universe,
                lookback_days=25,
                auto_adjust=get_config_value(config, "runtime", "yf_auto_adjust", default=False),
                threads=get_config_value(config, "runtime", "threads", default=True),
            )
            movers_filtered = filter_movers(movers_raw, technicals_df=tech_df if not tech_df.empty else None, config=movers_config)
            
            from ..features.movers.mover_queue import (
                load_mover_queue, update_mover_queue, get_eligible_movers, save_mover_queue
            )
            queue_df = load_mover_queue()
            queue_df = update_mover_queue(movers_filtered, datetime.now(timezone.utc), movers_config)
            save_mover_queue(queue_df)
            
            eligible_movers = get_eligible_movers(queue_df, datetime.now(timezone.utc))
            if eligible_movers:
                # Tag movers with source
                mover_source_tags = {t: ["DAILY_MOVER"] for t in eligible_movers}
                universe = sorted(set(universe + eligible_movers))
                universe_note += f" + {len(eligible_movers)} daily movers"
        except Exception as e:
            print(f"[WARN] Daily movers integration failed: {e}")
    
    # Download price data - Check permanent database FIRST, then fallback to API
    lookback_days = int(get_config_value(config, "technicals", "lookback_days", default=300))
    auto_adjust = get_config_value(config, "runtime", "yf_auto_adjust", default=False)
    threads = get_config_value(config, "runtime", "threads", default=True)
    # Note: polygon_api_key, use_polygon_primary, use_polygon_fallback, polygon_max_workers 
    # are already defined above (before movers integration)

    # Calculate date range for data
    end_date = asof_trading_date.strftime("%Y-%m-%d") if asof_date else pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days + 20)).strftime("%Y-%m-%d")

    # ═══════════════════════════════════════════════════════════════════════════════
    # STEP 1: Check permanent price database FIRST (no API calls needed)
    # ═══════════════════════════════════════════════════════════════════════════════
    data: dict[str, pd.DataFrame] = {}
    tickers_need_download = []
    
    try:
        from ..core.price_db import get_price_db
        db = get_price_db()
        
        print(f"\n[1/4] Loading OHLCV from permanent database...")
        for t in universe:
            try:
                raw_df = db.get_prices(t, start_date, end_date)
                if raw_df is not None and not raw_df.empty:
                    # PR1: Validate and enforce as-of
                    clean_df, vstats = validate_ohlcv(raw_df, t)
                    clean_df, astats = enforce_asof(clean_df, asof_date_str, t, strict=False)
                    
                    if not clean_df.empty and len(clean_df) >= 20:
                        data[t] = clean_df
                        ledger.add(LedgerRow(
                            ticker=t,
                            stage="load_prices",
                            provider_used="price_db",
                            rows=len(clean_df),
                            first_date=str(clean_df.index.min().date()) if not clean_df.empty else None,
                            last_date=str(clean_df.index.max().date()) if not clean_df.empty else None,
                            missing_cols=";".join(vstats.get("missing_cols") or []) or None,
                            dropped_bad_rows=vstats.get("dropped_bad_rows"),
                            dropped_future_rows=astats.get("dropped_future_rows"),
                        ))
                    else:
                        tickers_need_download.append(t)
                else:
                    tickers_need_download.append(t)
            except Exception as e:
                ledger.add_exception(t, "load_prices", e, "price_db")
                tickers_need_download.append(t)
        
        if data:
            print(f"  Database: {len(data)} tickers loaded, {len(tickers_need_download)} missing")
    except Exception as e:
        print(f"  [WARN] Price database not available: {e}")
        tickers_need_download = universe
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STEP 2: Download missing tickers (Polygon primary, yfinance fallback)
    # ═══════════════════════════════════════════════════════════════════════════════
    if tickers_need_download:
        polygon_data = {}
        if use_polygon_primary and polygon_api_key:
            print(f"[1/4] Fetching OHLCV from Polygon for {len(tickers_need_download)} tickers...")
            polygon_data = download_polygon_batch(
                tickers=tickers_need_download,
                lookback_days=lookback_days,
                asof_date=asof_date,
                api_key=polygon_api_key,
                max_workers=polygon_max_workers,
                quarantine_cfg=quarantine_cfg,
                retry_cfg=poly_retry_cfg,
            )
            populated = sum(1 for df in polygon_data.values() if df is not None and not df.empty)
            print(f"  Polygon populated {populated}/{len(tickers_need_download)} tickers.")
            
            # Store in permanent database for future use
            try:
                for t, df in polygon_data.items():
                    if df is not None and not df.empty:
                        db.store_prices(t, df, source="polygon")
            except Exception:
                pass  # Ignore storage errors

        yf_targets = [t for t in tickers_need_download if polygon_data.get(t) is None or polygon_data.get(t).empty]
        yf_dict: dict[str, pd.DataFrame] = {}
        if yf_targets:
            print(f"[1/4] Fetching OHLCV from yfinance for {len(yf_targets)} tickers (fallback)...")
            if asof_date:
                # Use cached version for efficiency
                yf_dict, report = download_daily_range_cached(
                    tickers=yf_targets,
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    threads=threads,
                    quarantine_cfg=quarantine_cfg,
                    retry_cfg=yf_retry_cfg,
                )
                cache_hits = report.get("cache_hits", 0)
                if cache_hits > 0:
                    print(f"  Cache: {cache_hits} hits, {report.get('downloaded', 0)} downloaded")
            else:
                # For live runs without asof_date, use standard download
                yf_data, _ = download_daily(
                    tickers=yf_targets,
                    period=f"{lookback_days}d",
                    interval="1d",
                    auto_adjust=auto_adjust,
                    threads=threads,
                    quarantine_cfg=quarantine_cfg,
                    retry_cfg=yf_retry_cfg,
                )
                yf_dict = {t: get_ticker_df(yf_data, t) for t in yf_targets}
            
            # Store in permanent database for future use
            try:
                for t, df in yf_dict.items():
                    if df is not None and not df.empty:
                        db.store_prices(t, df, source="yfinance")
            except Exception:
                pass  # Ignore storage errors

        # Merge downloaded data with validation
        for t in tickers_need_download:
            provider_used = None
            raw_df = pd.DataFrame()
            
            # Try Polygon first
            df_poly = polygon_data.get(t, pd.DataFrame()) if use_polygon_primary else pd.DataFrame()
            if df_poly is not None and not df_poly.empty:
                raw_df = df_poly.dropna(how="any")
                provider_used = "polygon"
            
            # Try yfinance fallback
            if raw_df.empty:
                df_yf = yf_dict.get(t) if yf_dict else pd.DataFrame()
                if df_yf is not None and not df_yf.empty:
                    raw_df = df_yf.dropna(how="any")
                    provider_used = "yfinance"
            
            # Try Polygon fallback
            if raw_df.empty and use_polygon_fallback:
                alt_df = fetch_polygon_daily(
                    ticker=t,
                    lookback_days=lookback_days,
                    asof_date=asof_date,
                    api_key=polygon_api_key,
                    retry_cfg=poly_retry_cfg,
                )
                if not alt_df.empty:
                    raw_df = alt_df.dropna(how="any")
                    provider_used = "polygon_fallback"
            
            # PR1: Validate and enforce as-of for downloaded data
            if not raw_df.empty:
                try:
                    clean_df, vstats = validate_ohlcv(raw_df, t)
                    clean_df, astats = enforce_asof(clean_df, asof_date_str, t, strict=False)
                    
                    if not clean_df.empty:
                        data[t] = clean_df
                        ledger.add(LedgerRow(
                            ticker=t,
                            stage="load_prices",
                            provider_used=provider_used,
                            rows=len(clean_df),
                            first_date=str(clean_df.index.min().date()) if not clean_df.empty else None,
                            last_date=str(clean_df.index.max().date()) if not clean_df.empty else None,
                            missing_cols=";".join(vstats.get("missing_cols") or []) or None,
                            dropped_bad_rows=vstats.get("dropped_bad_rows"),
                            dropped_future_rows=astats.get("dropped_future_rows"),
                        ))
                except Exception as e:
                    ledger.add_exception(t, "load_prices", e, provider_used)
    
    # Screen candidates
    print(f"\n[2/4] Screening {len(universe)} tickers...")
    candidates = []
    dropped: list[dict] = []
    # Convert config dict to format expected by apply_hard_filters
    filter_params = {
        "price_min": get_config_value(config, "liquidity", "price_min", default=2.0),
        "avg_dollar_volume_20d_min": get_config_value(config, "liquidity", "min_avg_dollar_volume_20d", default=50_000_000),
        # Config uses fraction (e.g., 0.15 == 15%)
        "price_up_5d_max_pct": float(get_config_value(config, "liquidity", "max_5d_return", default=0.15)) * 100.0,
    }
    min_tech_score = float(get_config_value(config, "quality_filters_weekly", "min_technical_score", default=0.0) or 0.0)
    
    total = len(universe)
    for idx, ticker in enumerate(universe):
        # Progress indicator
        if (idx + 1) % max(1, total // 20) == 0 or (idx + 1) == total:
            pct = ((idx + 1) / total) * 100
            print(f"  Progress: {idx + 1}/{total} ({pct:.1f}%) | Found: {len(candidates)} candidates", end="\r")
        df = data.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 20:
            dropped.append({"ticker": ticker, "stage": "data", "reason": "empty_or_short_history"})
            ledger.add(LedgerRow(ticker=ticker, stage="screen", reject_reason="empty_or_short_history"))
            continue
        
        # Apply hard filters
        passed, reasons = apply_hard_filters(df, filter_params)
        if not passed:
            dropped.append({"ticker": ticker, "stage": "filters", "reason": "; ".join(reasons)})
            ledger.add(LedgerRow(ticker=ticker, stage="filters", reject_reason="; ".join(reasons)[:200]))
            continue
        
        # PR2: Compute features using FeatureSet (leak-proof)
        features = compute_features_weekly(df, ticker, asof_date_str or "")
        
        # PR2: Score using FeatureSet only (no DataFrame access)
        tech = score_weekly(features)
        
        # PR3: Add breakout scoring
        br = score_breakout(df)
        
        # PR3: Add momentum ATR adjustment
        mom_adj = momentum_atr_adjust(features)
        
        # PR3: Combine scores (base + adjustments, bounded 0-10)
        # Breakout contributes a mild blend (0.5 * breakout_score / 10 * 2.0 = up to +1.0)
        final_tech_score = float(tech.score) + float(mom_adj.score_adj) + (0.5 * float(br.breakout_score) / 10.0 * 2.0)
        final_tech_score = max(0.0, min(10.0, final_tech_score))
        
        # Build combined evidence
        tech_evidence = dict(tech.evidence)
        tech_evidence["breakout"] = {"score": br.breakout_score, **br.evidence}
        tech_evidence["momentum_norm"] = mom_adj.evidence
        tech_evidence["data_gaps"] = tech.data_gaps
        if tech.cap_applied:
            tech_evidence["cap_applied"] = tech.cap_applied
        
        # Create tech_result dict for backward compatibility
        tech_result = {
            "score": round(final_tech_score, 2),
            "evidence": tech_evidence,
            "base_score": tech.score,
            "breakout_score": br.breakout_score,
            "momentum_adj": mom_adj.score_adj,
        }
        
        if tech_result["score"] == 0:
            dropped.append({"ticker": ticker, "stage": "scoring", "reason": "tech_score_zero"})
            ledger.add(LedgerRow(ticker=ticker, stage="scoring", reject_reason="tech_score_zero"))
            continue
        if min_tech_score > 0 and float(tech_result["score"]) < min_tech_score:
            dropped.append({"ticker": ticker, "stage": "scoring", "reason": f"tech_score_below_min_{min_tech_score}"})
            ledger.add(LedgerRow(ticker=ticker, stage="scoring", reject_reason=f"tech_score_below_min_{min_tech_score}"))
            continue
        
        # Get basic metrics
        close = df["Close"]
        volume = df["Volume"]
        last = float(close.iloc[-1])
        adv20 = float((close.tail(20) * volume.tail(20)).mean()) if len(close) >= 20 else 0.0
        
        # Get last trading day timestamp
        try:
            last_date = pd.Timestamp(df.index[-1])
            asof_price_utc = last_date.isoformat() + "Z"
        except (IndexError, ValueError, TypeError):
            asof_price_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Store basic candidate data (company info fetched later for top candidates only)
        # This avoids making thousands of API calls during screening
        candidates.append({
            "ticker": ticker,
            "name": ticker,  # Will be updated later for top candidates
            "exchange": "Unknown",  # Will be updated later for top candidates
            "sector": "Unknown",  # Will be updated later for top candidates
            "technical_score": tech_result["score"],
            "technical_evidence": tech_result["evidence"],
            "breakout_score": tech_result.get("breakout_score", 0.0),  # PR3: Add breakout score
            "momentum_adj": tech_result.get("momentum_adj", 0.0),  # PR3: Add momentum adjustment
            "current_price": last,
            "market_cap_usd": None,  # Will be updated later for top candidates
            "avg_dollar_volume_20d": adv20,
            "asof_price_utc": asof_price_utc,
        })
    
    print()  # New line after progress
    print(f"  Screening complete: {len(candidates)} candidates found")
    
    # Convert to DataFrame and sort by technical score
    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values("technical_score", ascending=False).head(30)
        validate_required_columns(
            candidates_df,
            required_cols=["ticker", "technical_score", "current_price", "asof_price_utc"],
            context="weekly candidates"
        )
    
    # Fetch company info for top candidates only (more efficient than during screening)
    if not candidates_df.empty:
        print(f"\n[3/4] Fetching company info for {len(candidates_df)} top candidates...")
        import yfinance as yf
        tickers_list = candidates_df["ticker"].tolist()
        
        total_info = len(candidates_df)
        for info_idx, (idx, row) in enumerate(candidates_df.iterrows()):
            # Progress for info fetching
            if (info_idx + 1) % 5 == 0 or (info_idx + 1) == total_info:
                pct = ((info_idx + 1) / total_info) * 100
                print(f"  Info: {info_idx + 1}/{total_info} ({pct:.1f}%)", end="\r")
            ticker = row["ticker"]
            # Skip if already has valid info
            if row.get("name") != ticker and row.get("exchange") != "Unknown":
                continue
                
            # Fetch company info with retry
            for attempt in range(2):
                try:
                    if attempt > 0:
                        time.sleep(0.5)
                    tk = yf.Ticker(ticker)
                    info = tk.info
                    if info and isinstance(info, dict) and len(info) > 0:
                        name = info.get("longName", info.get("shortName", ticker))
                        sector = info.get("sector", "Unknown")
                        exchange_raw = info.get("exchange", "Unknown")
                        # Normalize exchange names
                        if exchange_raw and exchange_raw != "Unknown":
                            if "NMS" in exchange_raw or "NASDAQ" in exchange_raw.upper():
                                exchange = "NASDAQ"
                            elif "NYQ" in exchange_raw or "NYSE" in exchange_raw.upper() or "New York" in exchange_raw:
                                exchange = "NYSE"
                            else:
                                exchange = exchange_raw
                        else:
                            exchange = "Unknown"
                        market_cap = info.get("marketCap", None)
                        
                        # Update DataFrame
                        candidates_df.at[idx, "name"] = name
                        candidates_df.at[idx, "sector"] = sector
                        candidates_df.at[idx, "exchange"] = exchange
                        if market_cap:
                            candidates_df.at[idx, "market_cap_usd"] = int(market_cap)
                        break
                except Exception:
                    if attempt == 1:
                        # Final attempt failed, keep defaults
                        pass
                    continue
            # Throttle to avoid rate limiting
            if (info_idx + 1) % 10 == 0:
                time.sleep(0.3)
        
        print()  # New line after info progress
    
    # Save candidates CSV
    candidates_csv = run_dir / f"weekly_scanner_candidates_{output_date_str}.csv"
    if not candidates_df.empty:
        save_csv(candidates_df, candidates_csv)
    
    # Build packets
    print(f"\n[4/4] Building LLM packets...")
    packets = []
    if not candidates_df.empty:
        tickers_list = candidates_df["ticker"].tolist()
        print(f"  Fetching news for {len(tickers_list)} tickers...")
        news_df = fetch_news_for_tickers(
            tickers_list,
            max_items=get_config_value(config, "news", "max_items", default=25),
            throttle_sec=get_config_value(config, "news", "throttle_sec", default=0.15),
        )
        print(f"  Loaded {len(news_df)} news headlines")
        manual_headlines_df = load_manual_headlines("manual_headlines.csv")
        
        # Get API keys for enhanced data sources
        polygon_api_key = os.environ.get("POLYGON_API_KEY")
        fmp_api_key = os.environ.get("FMP_API_KEY")  # Financial Modeling Prep
        adanos_api_key = os.environ.get("ADANOS_API_KEY")  # Adanos (Twitter sentiment)
        
        # Check if options/sentiment fetching is enabled in config
        fetch_options = get_config_value(config, "features", "fetch_options", default=True)
        fetch_sentiment = get_config_value(config, "features", "fetch_sentiment", default=True)
        use_enhanced_sentiment = get_config_value(config, "features", "use_enhanced_sentiment", default=True)
        
        if fetch_options or fetch_sentiment:
            sources_msg = []
            if polygon_api_key:
                sources_msg.append("Polygon")
            if fmp_api_key:
                sources_msg.append("FMP")
            if adanos_api_key:
                sources_msg.append("Adanos")
            sources_str = f" ({', '.join(sources_msg)})" if sources_msg else ""
            print(f"  Fetching options & sentiment data{sources_str} (this may take a moment)...")
        
        total_packets = len(candidates_df)
        for pkt_idx, (_, row) in enumerate(candidates_df.iterrows()):
            if (pkt_idx + 1) % 5 == 0 or (pkt_idx + 1) == total_packets:
                pct = ((pkt_idx + 1) / total_packets) * 100
                print(f"  Packets: {pkt_idx + 1}/{total_packets} ({pct:.1f}%)", end="\r")
            ticker = row["ticker"]
            earnings_date = get_next_earnings_date(ticker)
            source_tags = mover_source_tags.get(ticker, ["BASE_UNIVERSE"])
            
            # Build packet with options and sentiment data (enhanced sources)
            packet = build_weekly_scanner_packet(
                ticker=ticker,
                row=row,
                news_df=news_df,
                earnings_date=earnings_date,
                manual_headlines_df=manual_headlines_df,
                source_tags=source_tags,
                polygon_api_key=polygon_api_key,
                fmp_api_key=fmp_api_key,
                adanos_api_key=adanos_api_key,
                fetch_options=fetch_options,
                fetch_sentiment=fetch_sentiment,
                use_enhanced_sentiment=use_enhanced_sentiment,
            )
            
            # PR3: Add event gate for earnings proximity
            gate = earnings_proximity_gate(
                asof_date=asof_date_str or str(pd.Timestamp.utcnow().date()),
                earnings_date=earnings_date if earnings_date and earnings_date != "Unknown" else None,
                block_days=3,
            )
            packet["event_gate"] = {
                "blocked": gate.blocked,
                "reason": gate.reason,
                **gate.evidence
            }
            
            # Add breakout/momentum fields from candidate row if available
            packet["breakout_score"] = row.get("breakout_score", 0.0)
            packet["momentum_adj"] = row.get("momentum_adj", 0.0)
            
            packets.append(packet)
        
        print()  # New line after packet progress
    
    # PR5+PR7+Governance: Probability inference with decay detection and fallback
    calibration_model_path = get_config_value(config, "calibration", "model_path", default=None)
    calibration_model_dir = get_config_value(config, "calibration", "model_dir", default=None)
    outcomes_dir = get_config_value(config, "calibration", "outcomes_dir", default=None)
    
    # G2: Check for model decay before using calibration
    if CALIBRATION_AVAILABLE and GOVERNANCE_AVAILABLE and not disable_calibration:
        meta_path = None
        if calibration_model_dir and current_regime:
            meta_path = os.path.join(calibration_model_dir, f"calibration_{current_regime}.json")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(calibration_model_dir, "calibration_meta.json")
        elif calibration_model_path:
            meta_path = calibration_model_path.replace(".pkl", "_meta.json")
        
        if meta_path and os.path.exists(meta_path) and outcomes_dir and load_recent_trades and check_decay:
            try:
                training_metrics = load_training_metrics(meta_path)
                recent_trades = load_recent_trades(outcomes_dir, n_recent=20)
                
                if not recent_trades.empty:
                    live_metrics = rolling_metrics(recent_trades, window=20)
                    decay_result = check_decay(live_metrics, training_metrics)
                    
                    if decay_result.get("decay_detected"):
                        decay_detected = True
                        governance_flags.append("calibration_disabled_decay")
                        fallback_reason = f"decay_detected: {', '.join(decay_result.get('triggers', []))}"
                        print(f"  [WARN] Model decay detected, using fallback ranking")
            except Exception as e:
                print(f"  [WARN] Decay check failed: {e}")
    
    # G5.1: Determine if calibration should be used (with fallback)
    should_use_calibration = (
        CALIBRATION_AVAILABLE and
        not disable_calibration and
        not decay_detected and
        (
            (calibration_model_dir and os.path.isdir(calibration_model_dir)) or
            (calibration_model_path and os.path.exists(calibration_model_path))
        )
    )
    
    if should_use_calibration:
        use_regime_inference = calibration_model_dir and os.path.isdir(calibration_model_dir)
        use_single_model = calibration_model_path and os.path.exists(calibration_model_path)
        
        if use_regime_inference:
            # Get model version for governance
            regime_meta_path = os.path.join(calibration_model_dir, f"calibration_{current_regime}.json")
            if os.path.exists(regime_meta_path):
                try:
                    import json
                    with open(regime_meta_path) as f:
                        meta = json.load(f)
                        calibration_model_version = meta.get("model_version")
                except Exception:
                    pass
            
            print(f"  Inferring probabilities (regime-aware: {current_regime})...")
            calibration_used = True
            for packet in packets:
                row_features = {
                    "technical_score": packet.get("technical_score", 0),
                    "breakout_score": packet.get("breakout_score", 0),
                    "momentum_adj": packet.get("momentum_adj", 0),
                    "rsi14": packet.get("rsi14"),
                    "atr_pct": packet.get("atr_pct"),
                    "vol_ratio_3_20": packet.get("vol_ratio_3_20"),
                    "dist_52w_high_pct": packet.get("dist_52w_high_pct"),
                }
                prob = infer_probability_regime(calibration_model_dir, current_regime, row_features) if infer_probability_regime else None
                packet["prob_hit_10"] = round(prob, 4) if prob is not None else None
                
                if prob is not None:
                    ev = compute_expected_value(prob, packet.get("atr_pct"), target_pct=10.0)
                    packet["expected_value"] = round(ev, 4) if ev is not None else None
                else:
                    packet["expected_value"] = None
            print(f"  ✓ Regime-aware probability inference complete")
        
        elif use_single_model:
            # Get model version for governance
            single_meta_path = calibration_model_path.replace(".pkl", "_meta.json")
            if not os.path.exists(single_meta_path):
                single_meta_path = os.path.join(os.path.dirname(calibration_model_path), "calibration_meta.json")
            if os.path.exists(single_meta_path):
                try:
                    import json
                    with open(single_meta_path) as f:
                        meta = json.load(f)
                        calibration_model_version = meta.get("model_version")
                except Exception:
                    pass
            
            print(f"  Inferring probabilities from calibration model...")
            calibration_used = True
            for packet in packets:
                row_features = {
                    "technical_score": packet.get("technical_score", 0),
                    "breakout_score": packet.get("breakout_score", 0),
                    "momentum_adj": packet.get("momentum_adj", 0),
                    "rsi14": packet.get("rsi14"),
                    "atr_pct": packet.get("atr_pct"),
                    "vol_ratio_3_20": packet.get("vol_ratio_3_20"),
                    "dist_52w_high_pct": packet.get("dist_52w_high_pct"),
                }
                prob = infer_probability(calibration_model_path, row_features)
                packet["prob_hit_10"] = round(prob, 4) if prob is not None else None

                if prob is not None:
                    ev = compute_expected_value(prob, packet.get("atr_pct"), target_pct=10.0)
                    packet["expected_value"] = round(ev, 4) if ev is not None else None
                else:
                    packet["expected_value"] = None
            print(f"  ✓ Probability inference complete")
    else:
        # G5.1: Fallback - no calibration, use raw technical_score ranking
        if not fallback_reason:
            if not CALIBRATION_AVAILABLE:
                fallback_reason = "calibration_not_available"
            elif not calibration_model_path and not calibration_model_dir:
                fallback_reason = "no_model_configured"
            else:
                fallback_reason = "model_not_found"
        
        print(f"  Using fallback ranking (reason: {fallback_reason})")
        for packet in packets:
            packet["prob_hit_10"] = None
            packet["expected_value"] = None
    
    # PR5+PR7: Write decision snapshot for future calibration training (with regime)
    if CALIBRATION_AVAILABLE and write_decision_snapshot and not candidates_df.empty:
        snapshot_rows = []
        for _, row in candidates_df.iterrows():
            snapshot_rows.append({
                "run_date": output_date_str,
                "asof_date": asof_date_str or output_date_str,
                "ticker": row["ticker"],
                "regime": current_regime,  # PR7: Include regime for regime-specific training
                "technical_score": row.get("technical_score", 0),
                "breakout_score": row.get("breakout_score", 0),
                "momentum_adj": row.get("momentum_adj", 0),
                "rsi14": row.get("technical_evidence", {}).get("rsi14") if isinstance(row.get("technical_evidence"), dict) else None,
                "atr_pct": row.get("technical_evidence", {}).get("atr_pct") if isinstance(row.get("technical_evidence"), dict) else None,
                "vol_ratio_3_20": row.get("technical_evidence", {}).get("volume_ratio_3d_to_20d") if isinstance(row.get("technical_evidence"), dict) else None,
                "dist_52w_high_pct": row.get("technical_evidence", {}).get("dist_to_52w_high_pct") if isinstance(row.get("technical_evidence"), dict) else None,
                "adv_20": row.get("avg_dollar_volume_20d"),
            })
        
        snapshot_path = run_dir / f"decision_snapshot_{output_date_str}.parquet"
        try:
            write_decision_snapshot(snapshot_rows, str(snapshot_path))
            print(f"  ✓ Decision snapshot: {snapshot_path.name}")
        except Exception as e:
            print(f"  ⚠ Decision snapshot write failed: {e}")
    
    # Save packets JSON
    packets_json = run_dir / f"weekly_scanner_packets_{output_date_str}.json"
    save_json({
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method_version": get_config_value(config, "runtime", "method_version", default="v3.0"),
        "universe_note": universe_note,
        "packets": packets,
    }, packets_json)
    
    # Save dropped ticker log for transparency
    if dropped:
        dropped_df = pd.DataFrame(dropped)
        dropped_csv = run_dir / f"dropped_weekly_{output_date_str}.csv"
        save_csv(dropped_df, dropped_csv)
    
    # Save metadata
    metadata_json = save_run_metadata(
        run_dir=run_dir,
        method_version=get_config_value(config, "runtime", "method_version", default="v3.0"),
        config=config,
        universe_size=len(universe),
        candidates_count=len(candidates_df),
        output_date=output_date_str,
        asof_trading_date=asof_trading_date.strftime("%Y-%m-%d"),
    )
    
    # PR1: Write quality ledger
    ledger_path = run_dir / f"quality_ledger_{output_date_str}.csv"
    try:
        ledger.write_csv(str(ledger_path))
        print(f"  ✓ Quality ledger: {ledger_path.name} ({len(ledger)} rows)")
    except Exception as e:
        print(f"  ⚠ Quality ledger write failed: {e}")
    
    # PR6: Build trade plan with position sizing
    trade_plan = []
    if PORTFOLIO_AVAILABLE and build_trade_plan and packets:
        try:
            portfolio_usd = float(get_config_value(config, "portfolio", "portfolio_usd", default=100000))
            max_positions = int(get_config_value(config, "portfolio", "max_positions", default=5))
            sizing_method = get_config_value(config, "portfolio", "sizing_method", default="kelly")
            kelly_shrink = float(get_config_value(config, "portfolio", "kelly_shrink", default=0.25))
            stop_pct = float(get_config_value(config, "portfolio", "stop_pct", default=5.0))
            
            portfolio_cfg = PortfolioConfig(
                portfolio_usd=portfolio_usd,
                max_positions=max_positions,
                min_prob=0.0,
                sizing=SizingConfig(
                    method=sizing_method,
                    kelly_shrink=kelly_shrink,
                    kelly_cap=float(get_config_value(config, "portfolio", "kelly_cap", default=0.05)),
                    max_weight=float(get_config_value(config, "portfolio", "max_weight", default=0.10)),
                    portfolio_gross=float(get_config_value(config, "portfolio", "portfolio_gross", default=1.0)),
                ),
                liquidity=LiquidityConfig(
                    max_adv_pct=float(get_config_value(config, "portfolio", "max_adv_pct", default=1.0)),
                    min_adv_usd=float(get_config_value(config, "portfolio", "min_adv_usd", default=2_000_000)),
                ),
            )
            
            trade_plan = build_trade_plan(
                candidates=packets,
                cfg=portfolio_cfg,
                target_pct=10.0,
                stop_pct=stop_pct,
                regime=current_regime,  # PR7: Pass regime for regime-aware sizing
            )
            
            # Write trade plan artifacts
            if trade_plan:
                write_trade_plan(trade_plan, str(run_dir), output_date_str)
                write_portfolio_summary(trade_plan, portfolio_cfg, str(run_dir), output_date_str, regime=current_regime)
                print(f"  ✓ Trade plan: {len(trade_plan)} positions, gross weight: {sum(p['weight'] for p in trade_plan):.1%}")
        except Exception as e:
            print(f"  ⚠ Portfolio construction failed: {e}")
    
    # Generate HTML report
    try:
        from ..core.report import generate_html_report
        html_file = generate_html_report(run_dir, output_date_str)
        print(f"  ✓ HTML report: {html_file}")
    except Exception as e:
        print(f"  ⚠ HTML report generation failed: {e}")
        html_file = None
    
    # G6: Write governance.json for audit trail
    if GOVERNANCE_AVAILABLE and write_governance_record:
        try:
            from ..governance.artifacts import GovernanceRecord
            governance_record = GovernanceRecord(
                calibration_used=calibration_used,
                model_version=calibration_model_version,
                regime=current_regime,
                eligibility_passed=eligibility_passed,
                decay_detected=decay_detected,
                fallback_reason=fallback_reason,
                governance_flags=governance_flags,
            )
            gov_path = write_governance_record(str(run_dir), output_date_str, governance_record)
            print(f"  ✓ Governance record: {os.path.basename(gov_path)}")
        except Exception as e:
            print(f"  ⚠ Governance record write failed: {e}")
    
    return {
        "universe_note": universe_note,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": run_dir,
        "candidates_csv": candidates_csv,
        "packets_json": packets_json,
        "metadata_json": metadata_json,
        "html_report": str(html_file) if html_file else None,
        "output_date": output_date_str,
        "asof_trading_date": asof_trading_date.strftime("%Y-%m-%d"),
    }

