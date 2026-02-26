"""
30-Day Momentum Screener Pipeline

Orchestrates the full 30-Day Momentum Screener pipeline using core modules.
"""

from __future__ import annotations
import os
from datetime import datetime
from datetime import date as date_type
from pathlib import Path
from typing import Optional
import pandas as pd

# Core imports
from ..core.config import load_config, get_config_value
from ..core.universe import build_universe
from ..core.io import get_run_dir, save_csv, save_json, save_run_metadata
from ..core.helpers import fetch_news_for_tickers, load_manual_headlines, get_trading_date, get_ny_date, validate_required_columns
from ..core.regime import check_regime
from ..core.universe import load_tickers_from_file
from ..utils.time import utc_now

# Pro30-specific imports
from .pro30_screening import (
    build_attention_pool,
    screen_universe_30d,
    build_pro30_llm_packet,
    is_market_open,
)

# Movers imports
try:
    from ..features.movers.daily_movers import compute_daily_movers_from_universe
    from ..features.movers.mover_filters import filter_movers
    from ..features.movers.mover_queue import (
        update_mover_queue, get_eligible_movers, load_mover_queue, save_mover_queue
    )
except ImportError:
    compute_daily_movers_from_universe = None
    filter_movers = None
    update_mover_queue = None
    get_eligible_movers = None
    load_mover_queue = None
    save_mover_queue = None


def _convert_config_to_legacy_params(config: dict) -> dict:
    """Convert YAML config format to legacy PARAMS dict format."""
    p = {}
    
    # Universe
    p["universe_mode"] = get_config_value(config, "universe", "mode", default="SP500+NASDAQ100+R2000")
    p["universe_cache_file"] = get_config_value(config, "universe", "cache_file", default="universe_cache.csv")
    p["universe_cache_max_age_days"] = get_config_value(config, "universe", "cache_max_age_days", default=7)
    p["manual_include_file"] = get_config_value(config, "universe", "manual_include_file", default="tickers/manual_include_tickers.txt")
    p["r2000_include_file"] = get_config_value(config, "universe", "r2000_include_file", default="tickers/r2000.txt")
    p["manual_include_mode"] = get_config_value(config, "universe", "manual_include_mode", default="ALWAYS")

    # Quarantine (data reliability)
    dr = config.get("data_reliability", {})
    qcfg = dr.get("quarantine", {}) if isinstance(dr, dict) else {}
    p["quarantine_file"] = qcfg.get("file", "data/bad_tickers.json")
    p["quarantine_enabled"] = bool(qcfg.get("enabled", True))
    p["quarantine_cfg"] = qcfg
    p["yfinance_retry_cfg"] = dr.get("yfinance", {}) if isinstance(dr, dict) else {}
    p["polygon_retry_cfg"] = dr.get("polygon", {}) if isinstance(dr, dict) else {}
    
    # Attention Pool
    ap = config.get("attention_pool", {})
    p["attention_rvol_min"] = ap.get("rvol_min", 1.8)
    p["attention_atr_pct_min"] = ap.get("atr_pct_min", 3.5)
    p["attention_min_abs_day_move_pct"] = ap.get("min_abs_day_move_pct", 3.0)
    p["attention_lookback_days"] = ap.get("lookback_days", 120)
    p["attention_chunk_size"] = ap.get("chunk_size", 200)
    p["enable_intraday_attention"] = ap.get("enable_intraday", False)
    # runtime override to force intraday
    p["allow_partial_day_attention"] = get_config_value(config, "runtime", "allow_partial_day_attention", default=False)
    p["intraday_interval"] = ap.get("intraday_interval", "5m")
    p["intraday_lookback_days"] = ap.get("intraday_lookback_days", 5)
    p["market_open_buffer_min"] = ap.get("market_open_buffer_min", 20)
    p["intraday_rvol_min"] = ap.get("intraday_rvol_min", 2.0)
    
    # Quality filters
    qf = config.get("quality_filters_30d", {})
    p["price_min"] = get_config_value(config, "technicals", "price_min", default=7.0)
    p["avg_vol_min"] = 1_000_000  # share volume
    p["avg_dollar_vol_min"] = get_config_value(config, "liquidity", "min_avg_dollar_volume_20d", default=20_000_000)
    p["rvol_min"] = qf.get("rvol_min", 2.0)
    p["atr_pct_min"] = qf.get("atr_pct_min", 4.0)
    p["near_high_max_pct"] = qf.get("near_high_max_pct", 8.0)
    p["rsi_reversal_max"] = qf.get("rsi_reversal_max", 35.0)
    p["breakout_rsi_min"] = qf.get("breakout_rsi_min", 55.0)
    p["reversal_dist_to_high_min_pct"] = qf.get("reversal_dist_to_high_min_pct", 15.0)
    p["reversal_rsi_max"] = qf.get("reversal_rsi_max", 32.0)
    p["min_score"] = qf.get("min_score", 0.0)
    
    # Output controls
    out30 = config.get("outputs_30d", {})
    p["top_n_breakout"] = out30.get("top_n_breakout", 15)
    p["top_n_reversal"] = out30.get("top_n_reversal", 15)
    p["top_n_total"] = out30.get("top_n_total", 25)
    
    # Lookbacks
    p["lookback_days"] = get_config_value(config, "technicals", "lookback_days", default=300)
    
    # Regime gate
    rg = config.get("regime_gate", {})
    p["enable_regime_gate"] = rg.get("enabled", True)
    p["spy_symbol"] = rg.get("spy_symbol", "SPY")
    p["vix_symbol"] = rg.get("vix_symbol", "^VIX")
    p["spy_ma_days"] = rg.get("spy_ma_days", 20)
    p["vix_max"] = rg.get("vix_max", 25.0)
    p["regime_action"] = rg.get("action", "WARN")

    # Regime overrides (bear market scoring adjustments)
    ro = config.get("regime_overrides", {}).get("bear", {})
    p["regime_bear_reversal_bonus"] = ro.get("reversal_setup_bonus", 6.0)
    p["regime_bear_breakout_bonus"] = ro.get("breakout_setup_bonus", 2.0)
    
    # News
    p["news_max_items"] = get_config_value(config, "news", "max_items", default=25)
    p["packet_headlines"] = get_config_value(config, "news", "packet_headlines", default=12)
    p["throttle_sec"] = get_config_value(config, "news", "throttle_sec", default=0.15)
    
    # Daily Movers
    movers = config.get("movers", {})
    p["enable_daily_movers"] = movers.get("enabled", False)
    p["daily_movers_top_n"] = movers.get("top_n", 50)
    p["daily_movers_config"] = {
        "gainers_pct_range": movers.get("gainers_pct_range", [7.0, 15.0]),
        "losers_pct_range": movers.get("losers_pct_range", [-15.0, -7.0]),
        "gainers_volume_spike": movers.get("gainers_volume_spike", 2.0),
        "losers_volume_spike": movers.get("losers_volume_spike", 1.8),
        "close_position_min": movers.get("close_position_min", 0.75),
        "price_min": movers.get("price_min", 2.0),
        "adv_20d_min": movers.get("adv_20d_min", 50000000),
        "cooling_days_required": movers.get("cooling_days_required", 1),
        "max_age_days": movers.get("max_age_days", 5),
    }
    
    # Data provider (Massive/Polygon primary + fallbacks)
    runtime = config.get("runtime", {})
    api_key = os.environ.get("POLYGON_API_KEY") or runtime.get("polygon_api_key")
    p["polygon_api_key"] = api_key
    p["enable_polygon_primary"] = bool(runtime.get("polygon_primary", False) and api_key)
    p["enable_polygon_fallback"] = bool(runtime.get("polygon_fallback", False) and api_key)
    p["polygon_max_workers"] = int(runtime.get("polygon_max_workers", 8))
    p["attention_use_polygon"] = bool(runtime.get("polygon_primary", False) and api_key)
    p["enable_polygon_intraday"] = bool(runtime.get("polygon_intraday", False) and api_key)
    p["polygon_intraday_interval"] = runtime.get("polygon_intraday_interval", 5)
    p["polygon_intraday_lookback_days"] = runtime.get("polygon_intraday_lookback_days", 5)
    p["block_recent_splits"] = runtime.get("block_recent_splits", True)
    p["split_block_days"] = runtime.get("split_block_days", 7)
    p["split_lookback_days"] = runtime.get("split_lookback_days", 120)
    
    return p


def _get_eligible_daily_movers(params: dict, universe_tickers: list[str]) -> tuple[list[str], dict]:
    """
    Get eligible daily movers from the queue.
    
    Returns:
        tuple: (eligible_tickers_list, mover_source_tags_dict)
    """
    if not params.get("enable_daily_movers", False):
        return [], {}
    
    if not compute_daily_movers_from_universe:
        return [], {}
    
    print("\n[DAILY_MOVERS] Processing daily movers idea funnel...")
    
    # Get daily movers
    top_n = params.get("daily_movers_top_n", 50)
    from ..core.helpers import get_ny_date, get_trading_date
    import os
    from datetime import date as date_type
    
    # Use passed asof_date if available, otherwise fall back to last trading day
    asof_date_str = params.get("asof_date")  # "YYYY-MM-DD" string or None
    if asof_date_str:
        asof_date_obj = date_type.fromisoformat(asof_date_str)
    else:
        asof_date_obj = get_trading_date(get_ny_date())
    
    polygon_api_key = os.environ.get("POLYGON_API_KEY")
    movers = compute_daily_movers_from_universe(
        tickers=universe_tickers, 
        top_n=top_n,
        asof_date=asof_date_obj,
        polygon_api_key=polygon_api_key,
        use_polygon_primary=bool(params.get("enable_polygon_primary") and polygon_api_key),
        polygon_max_workers=params.get("polygon_max_workers", 8),
    )
    
    if movers["gainers"].empty and movers["losers"].empty:
        print("[DAILY_MOVERS] No movers found.")
        return [], {}
    
    print(f"[DAILY_MOVERS] Found {len(movers['gainers'])} gainers, {len(movers['losers'])} losers (raw)")
    
    # Apply filters
    movers_config = params.get("daily_movers_config", {})
    # Make movers credibility real: pass adv/avgvol so volume spike + $ADV20 checks can be enforced
    try:
        from ..features.movers.mover_filters import build_mover_technicals_df
        mover_universe = []
        for k in ("gainers", "losers"):
            dfm = movers.get(k)
            if isinstance(dfm, pd.DataFrame) and (not dfm.empty) and "ticker" in dfm.columns:
                mover_universe += dfm["ticker"].astype(str).tolist()
        tech_df = build_mover_technicals_df(
            mover_universe,
            lookback_days=25,
            auto_adjust=False,
            threads=True,
        )
    except Exception:
        tech_df = None
    candidates = filter_movers(movers, technicals_df=tech_df if tech_df is not None and not tech_df.empty else None, config=movers_config)
    
    passed = candidates[candidates["filter_pass"] == True]
    print(f"[DAILY_MOVERS] {len(passed)} candidates passed filters")
    
    # Update queue
    asof_utc = utc_now()
    queue_df = load_mover_queue()
    queue_df = update_mover_queue(passed, asof_utc, movers_config)
    save_mover_queue(queue_df)
    
    # Get eligible (cooling period passed)
    eligible = get_eligible_movers(queue_df, asof_utc)
    print(f"[DAILY_MOVERS] {len(eligible)} tickers eligible (cooling period passed)")
    
    # Build source tags dict
    source_tags = {}
    if not queue_df.empty:
        eligible_df = queue_df[queue_df["status"] == "ELIGIBLE"]
        for _, row in eligible_df.iterrows():
            ticker = row["ticker"]
            mover_type = row["mover_type"]
            if mover_type == "GAINER":
                source_tags[ticker] = ["DAILY_MOVER_GAINER"]
            elif mover_type == "LOSER":
                source_tags[ticker] = ["DAILY_MOVER_LOSER"]
    
    return eligible, source_tags


def _run_pro30_pipeline(params: dict, asof_date: date_type | None = None) -> dict:
    """
    Run the full Pro30 screener pipeline.
    
    Args:
        params: Dict with all pipeline parameters
    
    Returns:
        Dict with regime_info, tickers_to_screen, candidates, breakout_df, reversal_df, news_df, packets
    """
    # Regime gate
    regime_info = {"ok": True, "message": "Regime gate disabled."}
    if params.get("enable_regime_gate", True):
        asof_str = get_trading_date(asof_date).strftime("%Y-%m-%d") if asof_date else None
        regime_info = check_regime(params, asof_date=asof_str)
        print(f"[REGIME] {regime_info.get('message','')}")
        if not regime_info.get("ok", True):
            action = params.get("regime_action", "WARN").upper()
            if action == "BLOCK":
                print("[REGIME] BLOCK enabled: returning no candidates.")
                return {
                    "regime_info": regime_info,
                    "tickers_to_screen": [],
                    "candidates": pd.DataFrame(),
                    "breakout_df": pd.DataFrame(),
                    "reversal_df": pd.DataFrame(),
                    "news_df": pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"]),
                    "manual_headlines_df": pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"]),
                    "packets": [],
                }
            print("[REGIME] WARN: continuing, but consider smaller size / fewer trades.")

    # Build dynamic attention pool from universe
    print("\n[1/4] Building dynamic attention pool from universe...")
    broad = build_universe(
        mode=params["universe_mode"],
        cache_file=params.get("universe_cache_file"),
        cache_max_age_days=params.get("universe_cache_max_age_days", 7),
        manual_include_file=params.get("manual_include_file"),
        r2000_include_file=params.get("r2000_include_file"),
        manual_include_mode=params.get("manual_include_mode", "ALWAYS"),
        quarantine_file=params.get("quarantine_file", "data/bad_tickers.json"),
        quarantine_enabled=bool(params.get("quarantine_enabled", True)),
    )
    
    if not broad:
        print("Failed to fetch universe. Cannot proceed without dynamic universe.")
        return {
            "regime_info": regime_info,
            "tickers_to_screen": [],
            "candidates": pd.DataFrame(),
            "breakout_df": pd.DataFrame(),
            "reversal_df": pd.DataFrame(),
            "news_df": pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"]),
            "manual_headlines_df": pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"]),
            "packets": [],
        }
    
    print(f"Fetched {len(broad)} universe tickers.")
    # Historical replay: keep attention pool "as-of" by disabling intraday mode and using date-bounded downloads
    if asof_date:
        params = dict(params)
        params["enable_intraday_attention"] = False
        params["asof_date"] = get_trading_date(asof_date).strftime("%Y-%m-%d")

    attention = build_attention_pool(broad, params)
    print(f"\nAttention pool: {len(attention)} tickers showing market attention today.")

    # Load manual picks
    manual_picks = []
    manual_file = params.get("manual_include_file", "")
    r2000_file = params.get("r2000_include_file", "")
    if manual_file:
        manual_picks.extend(load_tickers_from_file(manual_file))
    if r2000_file:
        manual_picks.extend(load_tickers_from_file(r2000_file))
    manual_picks = sorted(set(manual_picks))
    
    if manual_file and Path(manual_file).exists():
        manual_tickers = load_tickers_from_file(manual_file)
        if manual_tickers:
            print(f"Manual include file ({manual_file}): {len(manual_tickers)} tickers")
    if r2000_file and Path(r2000_file).exists():
        r2000_tickers = load_tickers_from_file(r2000_file)
        if r2000_tickers:
            print(f"R2000 file ({r2000_file}): {len(r2000_tickers)} tickers")
    
    # Get eligible daily movers
    eligible_movers, mover_source_tags = _get_eligible_daily_movers(params, broad)
    if eligible_movers:
        print(f"Eligible daily movers: {len(eligible_movers)} tickers")

    mode = str(params.get("manual_include_mode", "ALWAYS")).upper()
    if mode == "ONLY_IF_IN_UNIVERSE":
        broad_set = set(broad)
        manual_picks = [t for t in manual_picks if t in broad_set]
        print(f"Manual picks after universe filter: {len(manual_picks)} tickers")

    tickers_to_screen = sorted(set(attention + manual_picks + eligible_movers))

    print(f"Total tickers to screen (attention + manual): {len(tickers_to_screen)}")
    
    if not tickers_to_screen:
        print("No tickers to screen. Try lowering attention_rvol_min or attention_atr_pct_min.")
        return {
            "regime_info": regime_info,
            "tickers_to_screen": [],
            "candidates": pd.DataFrame(),
            "breakout_df": pd.DataFrame(),
            "reversal_df": pd.DataFrame(),
            "news_df": pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"]),
            "manual_headlines_df": pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"]),
            "packets": [],
        }
    
    print(f"Sample (first 10 of {len(tickers_to_screen)}): {', '.join(tickers_to_screen[:10])}")

    print(f"\n[2/4] Applying quality filters to {len(tickers_to_screen)} tickers...")
    # Determine regime for scoring adjustments
    _regime_for_scoring = None
    if not regime_info.get("spy_above_ma", True):
        _regime_for_scoring = "stress"
    screened = screen_universe_30d(tickers_to_screen, params, regime=_regime_for_scoring)
    breakout_df = screened["breakout_df"]
    reversal_df = screened["reversal_df"]
    candidates = screened["combined_df"]
    dropped = screened.get("dropped", [])

    # Optional backtest-calibrated min-score gate
    min_score = float(params.get("min_score", 0.0) or 0.0)
    if min_score > 0:
        for name, df in [("breakout_df", breakout_df), ("reversal_df", reversal_df), ("candidates", candidates)]:
            if df is not None and not df.empty and "Score" in df.columns:
                df2 = df[df["Score"].astype(float) >= min_score].copy()
                if name == "breakout_df":
                    breakout_df = df2
                elif name == "reversal_df":
                    reversal_df = df2
                else:
                    candidates = df2

        # Re-sort and re-apply top-N limits (keeps behavior stable after filtering)
        if breakout_df is not None and not breakout_df.empty and "Score" in breakout_df.columns:
            breakout_df = breakout_df.sort_values("Score", ascending=False).reset_index(drop=True)
            breakout_df = breakout_df.head(int(params.get("top_n_breakout", 15)))
        if reversal_df is not None and not reversal_df.empty and "Score" in reversal_df.columns:
            reversal_df = reversal_df.sort_values("Score", ascending=False).reset_index(drop=True)
            reversal_df = reversal_df.head(int(params.get("top_n_reversal", 15)))
        if (breakout_df is not None and not breakout_df.empty) or (reversal_df is not None and not reversal_df.empty):
            candidates = pd.concat([breakout_df, reversal_df], ignore_index=True)
            if not candidates.empty and "Score" in candidates.columns:
                candidates = candidates.sort_values("Score", ascending=False).reset_index(drop=True)
                candidates = candidates.head(int(params.get("top_n_total", 25)))

    # Pull recent news headlines
    news_df = pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"])
    if not candidates.empty:
        tickers = candidates["Ticker"].tolist()
        news_df = fetch_news_for_tickers(tickers, max_items=params["news_max_items"], throttle_sec=params["throttle_sec"])

        # Clean news_df
        if not news_df.empty:
            news_df = news_df.dropna(subset=["title"])
            news_df = news_df[news_df["title"].astype(str).str.strip().ne("")]

    # Load manual headlines
    manual_headlines_df = load_manual_headlines("manual_headlines.csv")
    if not manual_headlines_df.empty:
        manual_count = len(manual_headlines_df)
        print(f"Loaded {manual_count} manual headline(s) from manual_headlines.csv")

    # Build LLM packets
    packets: list[str] = []
    if not candidates.empty:
        for _, row in candidates.iterrows():
            t = row["Ticker"]
            tags = mover_source_tags.get(t, ["BASE_UNIVERSE"])
            
            packets.append(build_pro30_llm_packet(
                ticker=t,
                metrics_row=row,
                news_df=news_df,
                max_headlines=params["packet_headlines"],
                regime_info=regime_info,
                manual_headlines_df=manual_headlines_df,
                source_tags=tags,
            ))

    return {
        "regime_info": regime_info,
        "tickers_to_screen": tickers_to_screen,
        "candidates": candidates,
        "breakout_df": breakout_df,
        "reversal_df": reversal_df,
        "news_df": news_df,
        "manual_headlines_df": manual_headlines_df,
        "packets": packets,
        "dropped": dropped,
    }


def run_pro30(
    config: Optional[dict] = None,
    config_path: Optional[str] = None,
    asof_date: Optional[date_type] = None,
    *,
    output_date: Optional[date_type] = None,
    run_dir: Optional[Path] = None,
) -> dict:
    """
    Run the 30-Day Momentum Screener pipeline.
    
    Args:
        config: Optional config dict (if None, loads from config_path)
        config_path: Path to config YAML file (defaults to config/default.yaml)
    
    Returns:
        dict with keys:
          - run_dir: Path
          - candidates_csv: Path
          - breakout_csv: Path
          - reversal_csv: Path
          - packets_txt: Path
          - metadata_json: Path
    """
    # Load config
    if config is None:
        config = load_config(config_path)
    
    # Convert config to params format
    p = _convert_config_to_legacy_params(config)
    
    # Run pipeline (as-of date controls data)
    result = _run_pro30_pipeline(p, asof_date=asof_date)

    # Separate output folder date from data as-of trading date
    if output_date is None:
        output_date = get_ny_date()
    asof_trading_date = get_trading_date(asof_date or output_date)

    root_dir = get_config_value(config, "outputs", "root_dir", default="outputs")
    if run_dir is None:
        run_dir = Path(root_dir) / output_date.strftime("%Y-%m-%d")
        run_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs using output_date in filenames (so notebooks/manual checks look under the run folder)
    date_str = output_date.strftime("%Y-%m-%d")
    
    candidates = result.get("candidates", pd.DataFrame())
    breakout_df = result.get("breakout_df", pd.DataFrame())
    reversal_df = result.get("reversal_df", pd.DataFrame())
    packets = result.get("packets", [])
    dropped = result.get("dropped", [])

    # Quality gates – block run if critical fields are missing
    for name, df in (("candidates", candidates), ("breakout", breakout_df), ("reversal", reversal_df)):
        _validate_output_quality(df, context=f"pro30 {name}")
    
    candidates_csv = None
    breakout_csv = None
    reversal_csv = None
    packets_txt = None
    
    if not candidates.empty:
        validate_required_columns(
            candidates,
            required_cols=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Setup", "Score"],
            context="pro30 candidates"
        )
        candidates_csv = run_dir / f"30d_momentum_candidates_{date_str}.csv"
        save_csv(candidates, candidates_csv)
    
    if not breakout_df.empty:
        validate_required_columns(
            breakout_df,
            required_cols=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Ret5d%", "Setup", "Score"],
            context="pro30 breakout"
        )
        breakout_csv = run_dir / f"30d_breakout_candidates_{date_str}.csv"
        save_csv(breakout_df, breakout_csv)
    
    if not reversal_df.empty:
        validate_required_columns(
            reversal_df,
            required_cols=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Ret5d%", "Setup", "Score"],
            context="pro30 reversal"
        )
        reversal_csv = run_dir / f"30d_reversal_candidates_{date_str}.csv"
        save_csv(reversal_df, reversal_csv)
    
    if packets:
        packets_txt = run_dir / f"llm_packets_{date_str}.txt"
        with open(packets_txt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(packets))
    
    # Save news dump
    news_df = result.get("news_df", pd.DataFrame())
    if not news_df.empty:
        news_csv = run_dir / f"news_dump_{date_str}.csv"
        save_csv(news_df, news_csv)

    if dropped:
        dropped_df = pd.DataFrame(dropped)
        dropped_csv = run_dir / f"dropped_pro30_{date_str}.csv"
        save_csv(dropped_df, dropped_csv)
    
    # Save metadata
    metadata_json = save_run_metadata(
        run_dir=run_dir,
        method_version=get_config_value(config, "runtime", "method_version", default="v3.0"),
        config=config,
        regime_info=result.get("regime_info", {}),
        tickers_screened=len(result.get("tickers_to_screen", [])),
        candidates_count=len(candidates),
        breakout_count=len(breakout_df),
        reversal_count=len(reversal_df),
        dropped_count=len(dropped),
        output_date=date_str,
        asof_trading_date=asof_trading_date.strftime("%Y-%m-%d"),
    )
    
    return {
        "run_dir": run_dir,
        "candidates_csv": candidates_csv,
        "breakout_csv": breakout_csv,
        "reversal_csv": reversal_csv,
        "packets_txt": packets_txt,
        "metadata_json": metadata_json,
        "regime_info": result.get("regime_info", {}),
    }


def _validate_output_quality(df: pd.DataFrame, context: str):
    """Raise if critical fields are missing or invalid; blocks the run."""
    if df is None or df.empty:
        return
    critical = ["Ticker", "Last", "$ADV20", "RVOL", "ATR%", "Ret20d%", "Ret5d%"]
    missing = {c: int(df[c].isna().sum()) for c in critical if c in df.columns}
    missing = {k: v for k, v in missing.items() if v > 0}
    if missing:
        raise ValueError(f"{context}: missing critical fields {missing}")
    if "Ticker" in df.columns:
        dupes = int(df["Ticker"].duplicated().sum())
        if dupes:
            raise ValueError(f"{context}: duplicate tickers detected ({dupes})")
    if "Last" in df.columns:
        bad_price = int((df["Last"] <= 0).sum())
        if bad_price:
            raise ValueError(f"{context}: non-positive prices detected ({bad_price})")

