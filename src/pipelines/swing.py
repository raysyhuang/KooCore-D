"""
Swing Strategy Pipeline (5-10 trading day horizon).

Regime-aware signals:
- bull: trend continuation breakouts
- chop: pullback to trend
- stress: skip or tighten thresholds
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from datetime import date as date_type
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.config import load_config, get_config_value
from ..core.universe import build_universe
from ..core.yf import download_daily, download_daily_range_cached, get_ticker_df
from ..core.polygon import download_polygon_batch, fetch_polygon_daily
from ..core.asof import validate_ohlcv, enforce_asof
from ..core.quality_ledger import QualityLedger, LedgerRow
from ..core.features_compute import compute_features_weekly
from ..core.score_weekly import score_weekly
from ..core.breakout import score_breakout
from ..core.momentum_norm import momentum_atr_adjust
from ..core.event_gate import earnings_proximity_gate
from ..core.filters import apply_hard_filters
from ..core.helpers import (
    fetch_news_for_tickers,
    get_ny_date,
    get_trading_date,
    get_next_earnings_date,
    load_manual_headlines,
    validate_required_columns,
)
from ..core.io import save_csv, save_json, save_run_metadata
from ..utils.time import utc_now_iso_z
from ..core.packets import build_weekly_scanner_packet
from ..core.signal_history import (
    load_signal_history,
    save_signal_history,
    update_signal_history,
    check_persistence,
)

# Regime (optional)
try:
    from ..regime.classifier import get_current_regime
    REGIME_AVAILABLE = True
except ImportError:
    get_current_regime = None
    REGIME_AVAILABLE = False

# Calibration (optional)
try:
    from ..calibration.infer import infer_probability_regime, infer_probability, compute_expected_value
    CALIBRATION_AVAILABLE = True
except ImportError:
    infer_probability_regime = None
    infer_probability = None
    compute_expected_value = None
    CALIBRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_regime(asof_date: Optional[str]) -> str:
    if REGIME_AVAILABLE and get_current_regime:
        try:
            return get_current_regime(asof_date=asof_date).name
        except Exception:
            return "chop"
    return "chop"


def _compute_ma(series: pd.Series, window: int) -> Optional[float]:
    if series is None or len(series) < window:
        return None
    return float(series.rolling(window).mean().iloc[-1])


def _swing_signal_for_regime(
    *,
    regime: str,
    last_close: float,
    rsi14: Optional[float],
    vol_ratio: Optional[float],
    dist_52w: Optional[float],
    ma20: Optional[float],
    ma50: Optional[float],
    ma200: Optional[float],
    breakout_score: float,
    cfg: dict,
) -> tuple[bool, str, str]:
    """Return (passes, signal_type, reason)."""
    if regime == "stress":
        stress_cfg = cfg.get("stress", {})
        if not stress_cfg.get("allow", False):
            return False, "", "stress_blocked"
        min_breakout = float(stress_cfg.get("min_breakout_score", 8.0))
        min_rsi = float(stress_cfg.get("min_rsi", 58.0))
        if breakout_score >= min_breakout and (rsi14 is not None and rsi14 >= min_rsi):
            return True, "breakout", "stress_breakout"
        return False, "", "stress_threshold_not_met"

    if regime == "bull":
        bull = cfg.get("risk_on", {})
        min_breakout = float(bull.get("min_breakout_score", 6.0))
        min_rsi = float(bull.get("min_rsi", 55.0))
        max_rsi = float(bull.get("max_rsi", 75.0))
        min_vol = float(bull.get("min_vol_ratio", 1.3))
        max_dist = float(bull.get("max_dist_52w_high_pct", 8.0))
        if breakout_score >= min_breakout:
            if rsi14 is not None and (min_rsi <= rsi14 <= max_rsi):
                if vol_ratio is not None and vol_ratio >= min_vol:
                    if dist_52w is None or dist_52w <= max_dist:
                        return True, "breakout", "bull_breakout"
        return False, "", "bull_threshold_not_met"

    # Default: chop (pullback)
    chop = cfg.get("chop", {})
    pullback_band = float(chop.get("pullback_band_pct", 2.5))
    min_rsi = float(chop.get("min_rsi", 40.0))
    max_rsi = float(chop.get("max_rsi", 55.0))
    max_vol = float(chop.get("max_vol_ratio", 1.2))
    max_dist = float(chop.get("max_dist_52w_high_pct", 15.0))
    require_ma200 = bool(chop.get("require_ma200", True))
    if ma20 and ma50 and last_close > ma50:
        if not require_ma200 or (ma200 and last_close > ma200):
            pullback_pct = abs(last_close - ma20) / ma20 * 100.0
            if pullback_pct <= pullback_band:
                if rsi14 is not None and (min_rsi <= rsi14 <= max_rsi):
                    if vol_ratio is None or vol_ratio <= max_vol:
                        if dist_52w is None or dist_52w <= max_dist:
                            return True, "pullback", "chop_pullback"
    return False, "", "chop_threshold_not_met"


def run_swing(
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
    asof_date: Optional[date_type] = None,
    *,
    output_date: Optional[date_type] = None,
    run_dir: Optional[Path] = None,
) -> dict:
    """Run swing strategy pipeline."""
    if config is None:
        config = load_config(config_path)

    if output_date is None:
        output_date = get_ny_date()
    asof_trading_date = get_trading_date(asof_date or output_date)
    output_date_str = output_date.strftime("%Y-%m-%d")
    asof_date_str = asof_trading_date.strftime("%Y-%m-%d")

    root_dir = get_config_value(config, "outputs", "root_dir", default="outputs")
    if run_dir is None:
        run_dir = Path(root_dir) / output_date_str
        run_dir.mkdir(parents=True, exist_ok=True)

    swing_cfg = config.get("swing_strategy", {})
    if not swing_cfg.get("enabled", True):
        return {
            "run_dir": run_dir,
            "packets_json": None,
            "candidates_csv": None,
            "metadata_json": None,
        }

    # Determine regime
    current_regime = _get_regime(asof_date_str)
    logger.info(f"  Swing regime: {current_regime}")

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

    # Date range for data
    lookback_days = int(get_config_value(config, "technicals", "lookback_days", default=300))
    end_date = asof_trading_date.strftime("%Y-%m-%d")
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days + 20)).strftime("%Y-%m-%d")

    # Config for data providers
    runtime = config.get("runtime", {})
    polygon_api_key = os.environ.get("POLYGON_API_KEY") or runtime.get("polygon_api_key")
    use_polygon_primary = bool(runtime.get("polygon_primary", False) and polygon_api_key)
    use_polygon_fallback = bool(runtime.get("polygon_fallback", False) and polygon_api_key)
    polygon_max_workers = int(runtime.get("polygon_max_workers", 8))
    auto_adjust = bool(runtime.get("yf_auto_adjust", False))
    threads = bool(runtime.get("threads", True))

    # Quality ledger
    ledger = QualityLedger()

    # Load price data with DB/Polygon/YF fallback
    data: dict[str, pd.DataFrame] = {}
    tickers_need_download = []
    try:
        from ..core.price_db import get_price_db
        db = get_price_db()
        for t in universe:
            try:
                raw_df = db.get_prices(t, start_date, end_date)
                if raw_df is not None and not raw_df.empty:
                    clean_df, vstats = validate_ohlcv(raw_df, t)
                    clean_df, astats = enforce_asof(clean_df, asof_date_str, t, strict=False)
                    if not clean_df.empty and len(clean_df) >= 20:
                        data[t] = clean_df
                        ledger.add(LedgerRow(
                            ticker=t,
                            stage="load_prices",
                            provider_used="price_db",
                            rows=len(clean_df),
                            first_date=str(clean_df.index.min().date()),
                            last_date=str(clean_df.index.max().date()),
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
    except Exception:
        tickers_need_download = universe

    if tickers_need_download:
        polygon_data: dict[str, pd.DataFrame] = {}
        if use_polygon_primary and polygon_api_key:
            polygon_data = download_polygon_batch(
                tickers=tickers_need_download,
                lookback_days=lookback_days,
                asof_date=asof_date_str,
                api_key=polygon_api_key,
                max_workers=polygon_max_workers,
                quarantine_cfg=quarantine_cfg,
            )

        yf_targets = [t for t in tickers_need_download if polygon_data.get(t) is None or polygon_data.get(t).empty]
        yf_dict: dict[str, pd.DataFrame] = {}
        if yf_targets:
            if asof_date_str:
                yf_dict, _ = download_daily_range_cached(
                    tickers=yf_targets,
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    threads=threads,
                    quarantine_cfg=quarantine_cfg,
                )
            else:
                yf_data, _ = download_daily(
                    tickers=yf_targets,
                    period=f"{lookback_days}d",
                    interval="1d",
                    auto_adjust=auto_adjust,
                    threads=threads,
                    quarantine_cfg=quarantine_cfg,
                )
                yf_dict = {t: get_ticker_df(yf_data, t) for t in yf_targets}

        for t in tickers_need_download:
            provider_used = None
            raw_df = pd.DataFrame()
            df_poly = polygon_data.get(t, pd.DataFrame()) if use_polygon_primary else pd.DataFrame()
            if df_poly is not None and not df_poly.empty:
                raw_df = df_poly.dropna(how="any")
                provider_used = "polygon"
            if raw_df.empty:
                df_yf = yf_dict.get(t) if yf_dict else pd.DataFrame()
                if df_yf is not None and not df_yf.empty:
                    raw_df = df_yf.dropna(how="any")
                    provider_used = "yfinance"
            if raw_df.empty and use_polygon_fallback:
                alt_df = fetch_polygon_daily(
                    ticker=t,
                    lookback_days=lookback_days,
                    asof_date=asof_date_str,
                    api_key=polygon_api_key,
                )
                if not alt_df.empty:
                    raw_df = alt_df.dropna(how="any")
                    provider_used = "polygon_fallback"
            if not raw_df.empty:
                try:
                    clean_df, vstats = validate_ohlcv(raw_df, t)
                    clean_df, astats = enforce_asof(clean_df, asof_date_str, t, strict=False)
                    if not clean_df.empty and len(clean_df) >= 20:
                        data[t] = clean_df
                        ledger.add(LedgerRow(
                            ticker=t,
                            stage="load_prices",
                            provider_used=provider_used,
                            rows=len(clean_df),
                            first_date=str(clean_df.index.min().date()),
                            last_date=str(clean_df.index.max().date()),
                            missing_cols=";".join(vstats.get("missing_cols") or []) or None,
                            dropped_bad_rows=vstats.get("dropped_bad_rows"),
                            dropped_future_rows=astats.get("dropped_future_rows"),
                        ))
                except Exception as e:
                    ledger.add_exception(t, "validate", e, provider_used)

    # Screening
    candidates: list[dict] = []
    dropped: list[dict] = []
    filter_params = {
        "price_min": get_config_value(config, "liquidity", "price_min", default=5.0),
        "avg_dollar_volume_20d_min": get_config_value(config, "liquidity", "min_avg_dollar_volume_20d", default=50_000_000),
        "price_up_5d_max_pct": float(get_config_value(config, "liquidity", "max_5d_return", default=0.15)) * 100.0,
    }
    min_final_score = float(swing_cfg.get("min_final_score", 5.8))

    total = len(universe)
    for idx, ticker in enumerate(universe):
        if (idx + 1) % max(1, total // 20) == 0 or (idx + 1) == total:
            pct = ((idx + 1) / total) * 100
            logger.info(f"  Swing scan: {idx + 1}/{total} ({pct:.1f}%) | Found: {len(candidates)}")
        df = data.get(ticker, pd.DataFrame())
        if df.empty or len(df) < 20:
            dropped.append({"ticker": ticker, "stage": "data", "reason": "empty_or_short_history"})
            ledger.add(LedgerRow(ticker=ticker, stage="screen", reject_reason="empty_or_short_history"))
            continue
        passed, reasons = apply_hard_filters(df, filter_params)
        if not passed:
            dropped.append({"ticker": ticker, "stage": "filters", "reason": "; ".join(reasons)})
            ledger.add(LedgerRow(ticker=ticker, stage="filters", reject_reason="; ".join(reasons)[:200]))
            continue

        features = compute_features_weekly(df, ticker, asof_date_str)
        tech = score_weekly(features)
        br = score_breakout(df)
        mom_adj = momentum_atr_adjust(features)

        final_tech_score = float(tech.score) + float(mom_adj.score_adj) + (0.4 * float(br.breakout_score) / 10.0 * 2.0)
        final_tech_score = max(0.0, min(10.0, final_tech_score))
        if final_tech_score < min_final_score:
            dropped.append({"ticker": ticker, "stage": "scoring", "reason": "score_below_min"})
            continue

        close = df["Close"]
        volume = df["Volume"]
        last_close = float(close.iloc[-1])
        adv20 = float((close.tail(20) * volume.tail(20)).mean()) if len(close) >= 20 else 0.0
        ma20 = _compute_ma(close, 20)
        ma50 = _compute_ma(close, 50)
        ma200 = _compute_ma(close, 200)
        rsi14 = features.rsi14
        vol_ratio = features.vol_ratio_3_20
        dist_52w = features.dist_52w_high_pct

        passed_signal, signal_type, signal_reason = _swing_signal_for_regime(
            regime=current_regime,
            last_close=last_close,
            rsi14=rsi14,
            vol_ratio=vol_ratio,
            dist_52w=dist_52w,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200,
            breakout_score=float(br.breakout_score),
            cfg=swing_cfg,
        )
        if not passed_signal:
            dropped.append({"ticker": ticker, "stage": "signal", "reason": signal_reason})
            continue

        signal_bonus = 0.5 if signal_type == "breakout" else 0.3
        swing_score = max(0.0, min(10.0, final_tech_score + signal_bonus))

        try:
            last_date = pd.Timestamp(df.index[-1])
            asof_price_utc = last_date.isoformat() + "Z"
        except Exception:
            asof_price_utc = utc_now_iso_z()

        tech_evidence = dict(tech.evidence)
        tech_evidence["breakout"] = {"score": br.breakout_score, **br.evidence}
        tech_evidence["momentum_norm"] = mom_adj.evidence
        tech_evidence["data_gaps"] = tech.data_gaps
        tech_evidence["ma20"] = round(ma20, 2) if ma20 else None
        tech_evidence["ma50"] = round(ma50, 2) if ma50 else None
        tech_evidence["ma200"] = round(ma200, 2) if ma200 else None

        candidates.append({
            "ticker": ticker,
            "name": ticker,
            "exchange": "Unknown",
            "sector": "Unknown",
            "technical_score": round(final_tech_score, 2),
            "swing_score": round(swing_score, 2),
            "swing_signal": signal_type,
            "swing_regime": current_regime,
            "signal_reason": signal_reason,
            "technical_evidence": tech_evidence,
            "breakout_score": br.breakout_score,
            "momentum_adj": mom_adj.score_adj,
            "current_price": last_close,
            "market_cap_usd": None,
            "avg_dollar_volume_20d": adv20,
            "asof_price_utc": asof_price_utc,
            "rsi14": rsi14,
            "atr_pct": features.atr_pct,
            "vol_ratio_3_20": vol_ratio,
            "dist_52w_high_pct": dist_52w,
        })

    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values("swing_score", ascending=False)
        validate_required_columns(
            candidates_df,
            required_cols=["ticker", "technical_score", "current_price", "asof_price_utc"],
            context="swing candidates",
        )

    # Signal persistence gating
    persistence_cfg = swing_cfg.get("persistence", {})
    persistence_enabled = bool(persistence_cfg.get("enabled", True))
    if persistence_enabled and not candidates_df.empty:
        raw_candidates = candidates_df.copy()
        history_path = persistence_cfg.get("file", "data/signal_history.json")
        history = load_signal_history(history_path)
        lookback_days = int(persistence_cfg.get("lookback_days", 3))
        min_hits = int(persistence_cfg.get("min_hits", 2))
        min_improve = float(persistence_cfg.get("min_score_improve", 0.2))
        allow_new = bool(persistence_cfg.get("allow_new", False))
        min_score_new = persistence_cfg.get("min_score_new", None)

        persistence_pass = []
        for _, row in candidates_df.iterrows():
            result = check_persistence(
                history,
                row["ticker"],
                output_date_str,
                lookback_days=lookback_days,
                min_hits=min_hits,
                min_score_improve=min_improve,
                current_score=row.get("swing_score"),
                allow_new=allow_new,
                min_score_new=min_score_new,
            )
            persistence_pass.append(result)
        candidates_df["persistence_pass"] = [r["passed"] for r in persistence_pass]
        candidates_df["persistence_hits"] = [r["hits"] for r in persistence_pass]
        candidates_df["persistence_improved"] = [r["improved"] for r in persistence_pass]
        candidates_df = candidates_df[candidates_df["persistence_pass"] == True]

        # Update history using pre-gate signals for continuity
        raw_candidates = raw_candidates.to_dict(orient="records")
        history = update_signal_history(
            history,
            output_date_str,
            raw_candidates,
            max_days=int(persistence_cfg.get("max_days", 30)),
            source="swing",
        )
        save_signal_history(history, history_path)

    # Probability / EV gating
    min_prob = swing_cfg.get("min_prob_hit")
    min_ev = swing_cfg.get("min_expected_value")
    if CALIBRATION_AVAILABLE and (min_prob is not None or min_ev is not None) and not candidates_df.empty:
        calibration_model_path = get_config_value(config, "calibration", "model_path", default=None)
        calibration_model_dir = get_config_value(config, "calibration", "model_dir", default=None)
        for idx, row in candidates_df.iterrows():
            row_features = {
                "technical_score": row.get("technical_score", 0),
                "breakout_score": row.get("breakout_score", 0),
                "momentum_adj": row.get("momentum_adj", 0),
                "rsi14": row.get("rsi14"),
                "atr_pct": row.get("atr_pct"),
                "vol_ratio_3_20": row.get("vol_ratio_3_20"),
                "dist_52w_high_pct": row.get("dist_52w_high_pct"),
            }
            prob = None
            if calibration_model_dir and infer_probability_regime:
                prob = infer_probability_regime(calibration_model_dir, current_regime, row_features)
            elif calibration_model_path and infer_probability:
                prob = infer_probability(calibration_model_path, row_features)
            ev = compute_expected_value(prob, row.get("atr_pct"), target_pct=10.0) if prob is not None else None
            candidates_df.at[idx, "prob_hit_10"] = round(prob, 4) if prob is not None else None
            candidates_df.at[idx, "expected_value"] = round(ev, 4) if ev is not None else None

        if min_prob is not None:
            candidates_df = candidates_df[candidates_df["prob_hit_10"].fillna(0) >= float(min_prob)]
        if min_ev is not None:
            candidates_df = candidates_df[candidates_df["expected_value"].fillna(0) >= float(min_ev)]

    # Trim to top N
    top_n = int(swing_cfg.get("top_n", 25))
    if not candidates_df.empty:
        candidates_df = candidates_df.head(top_n)

    # Fetch company info for top candidates
    if not candidates_df.empty:
        import yfinance as yf
        for idx, row in candidates_df.iterrows():
            ticker = row["ticker"]
            if row.get("name") != ticker and row.get("exchange") != "Unknown":
                continue
            for attempt in range(2):
                try:
                    if attempt > 0:
                        import time
                        time.sleep(0.5)
                    info = yf.Ticker(ticker).info
                    if info and isinstance(info, dict):
                        name = info.get("longName", info.get("shortName", ticker))
                        sector = info.get("sector", "Unknown")
                        exchange_raw = info.get("exchange", "Unknown")
                        exchange = exchange_raw
                        if exchange_raw and exchange_raw != "Unknown":
                            if "NMS" in exchange_raw or "NASDAQ" in exchange_raw.upper():
                                exchange = "NASDAQ"
                            elif "NYQ" in exchange_raw or "NYSE" in exchange_raw.upper() or "New York" in exchange_raw:
                                exchange = "NYSE"
                        candidates_df.at[idx, "name"] = name
                        candidates_df.at[idx, "sector"] = sector
                        candidates_df.at[idx, "exchange"] = exchange
                        market_cap = info.get("marketCap", None)
                        if market_cap:
                            candidates_df.at[idx, "market_cap_usd"] = int(market_cap)
                        break
                except Exception:
                    continue

    # Save candidates
    candidates_csv = run_dir / f"swing_candidates_{output_date_str}.csv"
    if not candidates_df.empty:
        save_csv(candidates_df, candidates_csv)

    # Build packets for LLM ranking
    packets = []
    if not candidates_df.empty:
        tickers_list = candidates_df["ticker"].tolist()
        news_df = fetch_news_for_tickers(
            tickers=tickers_list,
            max_items=get_config_value(config, "news", "max_items", default=25),
            throttle_sec=get_config_value(config, "news", "throttle_sec", default=0.15),
        )
        manual_headlines = load_manual_headlines()
        for _, row in candidates_df.iterrows():
            ticker = row["ticker"]
            earnings_date = get_next_earnings_date(ticker)
            gate = earnings_proximity_gate(
                asof_date_str,
                earnings_date,
                block_days=int(swing_cfg.get("earnings_block_days", 3)),
            )
            candidates_df.loc[candidates_df["ticker"] == ticker, "event_blocked"] = gate.blocked
            candidates_df.loc[candidates_df["ticker"] == ticker, "event_reason"] = gate.reason
            if gate.blocked and bool(swing_cfg.get("block_near_earnings", True)):
                continue
            source_tags = ["SWING", f"SWING_{row.get('swing_signal', '').upper()}"]
            packet = build_weekly_scanner_packet(
                ticker=ticker,
                row=row,
                news_df=news_df,
                earnings_date=earnings_date,
                manual_headlines_df=manual_headlines,
                source_tags=source_tags,
                polygon_api_key=polygon_api_key,
                fetch_options=bool(get_config_value(config, "features", "fetch_options", default=True)),
                fetch_sentiment=bool(get_config_value(config, "features", "fetch_sentiment", default=True)),
                use_enhanced_sentiment=bool(get_config_value(config, "features", "use_enhanced_sentiment", default=True)),
            )
            packets.append(packet)

    packets_json = run_dir / f"swing_packets_{output_date_str}.json"
    save_json({
        "run_timestamp_utc": utc_now_iso_z(),
        "method_version": get_config_value(config, "runtime", "method_version", default="v3.2"),
        "regime": current_regime,
        "universe_note": f"Universe: {len(universe)} tickers",
        "packets": packets,
    }, packets_json)

    # Save metadata
    metadata_json = save_run_metadata(
        run_dir=run_dir,
        method_version=get_config_value(config, "runtime", "method_version", default="v3.2"),
        config=config,
        universe_size=len(universe),
        candidates_count=len(candidates_df),
        output_date=output_date_str,
        asof_trading_date=asof_trading_date.strftime("%Y-%m-%d"),
    )

    # Save dropped log
    if dropped:
        dropped_df = pd.DataFrame(dropped)
        dropped_csv = run_dir / f"dropped_swing_{output_date_str}.csv"
        save_csv(dropped_df, dropped_csv)

    # Write quality ledger
    ledger_path = run_dir / f"quality_ledger_swing_{output_date_str}.csv"
    try:
        ledger.write_csv(str(ledger_path))
    except Exception:
        pass

    return {
        "run_dir": run_dir,
        "packets_json": packets_json,
        "candidates_csv": candidates_csv,
        "metadata_json": metadata_json,
        "regime": current_regime,
        "candidates_count": int(len(candidates_df)) if not candidates_df.empty else 0,
    }
