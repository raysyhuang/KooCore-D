"""Complete scan command handler (all systems + hybrid analysis)."""

from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from src.core.config import load_config
from src.core.helpers import get_ny_date, get_trading_date
from src.core.io import get_run_dir, save_json
from src.core.report import generate_html_report
from src.pipelines.weekly import run_weekly
from src.pipelines.pro30 import run_pro30
from src.pipelines.swing import run_swing
from src.core.universe import build_universe
from src.features.movers.daily_movers import compute_daily_movers_from_universe
from src.features.movers.mover_filters import filter_movers
from src.features.movers.mover_queue import (
    update_mover_queue, get_eligible_movers, load_mover_queue, save_mover_queue
)
from src.core.llm import rank_weekly_candidates, rank_with_debate

# Check if debate is available
try:
    from src.core.debate import DEBATE_AVAILABLE
except ImportError:
    DEBATE_AVAILABLE = False

# New modules (v3.1+)
try:
    from src.core.alerts import AlertConfig, send_overlap_alert, send_run_summary_alert
    from src.core.logging_utils import log_operation, ProgressLogger
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _open_browser(file_path: Path) -> None:
    """Open file in default browser (cross-platform)."""
    import webbrowser
    try:
        url = file_path.as_uri()
        webbrowser.open(url)
        logger.info("Opened report in browser")
    except Exception as e:
        logger.warning(f"Failed to open browser: {e}")
        logger.info(f"Manually open: {file_path}")


def cmd_all(args) -> int:
    """Run all screeners and produce hybrid analysis."""
    logger.info("=" * 60)
    logger.info("COMPLETE SCAN - All Systems")
    logger.info("=" * 60)
    
    # Output folder date = last trading day (not calendar date)
    # This ensures we don't create folders for weekends/holidays
    # - output_date_str: last trading day (or --date override)
    # - asof_date: same as output_date (last completed trading day)
    from datetime import datetime as dt
    if args.date:
        # User specified a date explicitly
        output_date = dt.strptime(args.date, "%Y-%m-%d").date()
        output_date_str = args.date
    else:
        # Use last trading day, not calendar date
        output_date = get_trading_date()
        output_date_str = output_date.strftime("%Y-%m-%d")
    asof_date = get_trading_date(output_date)
    config = load_config(args.config)
    
    if args.no_movers:
        config["movers"]["enabled"] = False
    if getattr(args, "legacy_pro30", False):
        config = dict(config)
        config.setdefault("liquidity", {})
        config["liquidity"]["min_avg_dollar_volume_20d"] = 20_000_000
        config.setdefault("quality_filters_30d", {})
        config["quality_filters_30d"]["min_score"] = 0.0
        config.setdefault("movers", {})
        config["movers"]["enabled"] = False
    if getattr(args, "intraday_attention", False):
        config.setdefault("attention_pool", {})
        config["attention_pool"]["enable_intraday"] = True
    if getattr(args, "allow_partial_day", False):
        config.setdefault("runtime", {})
        config["runtime"]["allow_partial_day_attention"] = True
    
    results = {}
    
    # Step 1: Daily Movers
    logger.info("\n[1/6] Daily Movers Discovery...")
    try:
        import os
        ucfg = config.get("universe", {})
        quarantine_cfg = config.get("data_reliability", {}).get("quarantine", {})
        universe = build_universe(
            mode=ucfg.get("mode", "SP500+NASDAQ100+R2000"),
            cache_file=ucfg.get("cache_file"),
            cache_max_age_days=ucfg.get("cache_max_age_days", 7),
            manual_include_file=ucfg.get("manual_include_file"),
            r2000_include_file=ucfg.get("r2000_include_file"),
            manual_include_mode=ucfg.get("manual_include_mode", "ALWAYS"),
            quarantine_file=quarantine_cfg.get("file", "data/bad_tickers.json"),
            quarantine_enabled=bool(quarantine_cfg.get("enabled", True)),
        )
        movers_config = config.get("movers", {})
        runtime_config = config.get("runtime", {})
        polygon_api_key = os.environ.get("POLYGON_API_KEY")
        reliability_cfg = config.get("data_reliability", {})
        movers_raw = compute_daily_movers_from_universe(
            universe, 
            top_n=movers_config.get("top_n", 50), 
            asof_date=get_trading_date(asof_date),
            polygon_api_key=polygon_api_key,
            use_polygon_primary=bool(runtime_config.get("polygon_primary", False) and polygon_api_key),
            polygon_max_workers=runtime_config.get("polygon_max_workers", 8),
            quarantine_cfg=reliability_cfg.get("quarantine", {}) if isinstance(reliability_cfg, dict) else {},
            yf_retry_cfg=reliability_cfg.get("yfinance", {}) if isinstance(reliability_cfg, dict) else {},
            polygon_retry_cfg=reliability_cfg.get("polygon", {}) if isinstance(reliability_cfg, dict) else {},
        )
        # Make movers credibility real: pass adv/avgvol so volume spike + $ADV20 checks can be enforced
        try:
            from src.features.movers.mover_filters import build_mover_technicals_df
            mover_universe = []
            for k in ("gainers", "losers"):
                dfm = movers_raw.get(k)
                if isinstance(dfm, pd.DataFrame) and (not dfm.empty) and "ticker" in dfm.columns:
                    mover_universe += dfm["ticker"].astype(str).tolist()
            tech_df = build_mover_technicals_df(
                mover_universe,
                lookback_days=25,
                auto_adjust=bool(config.get("runtime", {}).get("yf_auto_adjust", False)),
                threads=bool(config.get("runtime", {}).get("threads", True)),
            )
        except Exception:
            tech_df = None
        movers_filtered = filter_movers(movers_raw, technicals_df=tech_df if tech_df is not None and not tech_df.empty else None, config=movers_config)
        queue_df = load_mover_queue()
        queue_df = update_mover_queue(movers_filtered, datetime.utcnow(), movers_config)
        save_mover_queue(queue_df)
        eligible_movers = get_eligible_movers(queue_df, datetime.utcnow())
        results["movers"] = {"count": len(eligible_movers), "tickers": eligible_movers}
        logger.info(f"  ✓ Found {len(eligible_movers)} eligible movers")
    except Exception as e:
        logger.error(f"  ✗ Movers failed: {e}", exc_info=True)
        results["movers"] = {"count": 0, "tickers": []}
    
    # Root output dir (configurable)
    outputs_root = Path(config.get("outputs", {}).get("root_dir", "outputs"))
    output_dir = outputs_root / output_date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Swing Strategy (Primary)
    logger.info("\n[2/7] Swing Strategy (Primary)...")
    try:
        swing_result = run_swing(
            config=config,
            asof_date=asof_date,
            output_date=output_date,
            run_dir=output_dir,
        )
        results["swing"] = swing_result
        logger.info("  ✓ Swing strategy complete")
    except Exception as e:
        logger.error(f"  ✗ Swing strategy failed: {e}", exc_info=True)
        results["swing"] = None
    
    # Step 3: Weekly Scanner (Secondary)
    logger.info("\n[3/7] Weekly Scanner (Secondary)...")
    try:
        weekly_result = run_weekly(
            config=config,
            asof_date=asof_date,
            output_date=output_date,
            run_dir=output_dir,
        )
        results["weekly"] = weekly_result
        logger.info("  ✓ Weekly scanner complete")
    except Exception as e:
        logger.error(f"  ✗ Weekly scanner failed: {e}", exc_info=True)
        results["weekly"] = None
    
    # Step 4: 30-Day Screener (Secondary)
    logger.info("\n[4/7] 30-Day Screener (Secondary)...")
    try:
        pro30_result = run_pro30(
            config=config,
            asof_date=asof_date,
            output_date=output_date,
            run_dir=output_dir,
        )
        results["pro30"] = pro30_result
        logger.info("  ✓ 30-Day screener complete")
    except Exception as e:
        logger.error(f"  ✗ 30-Day screener failed: {e}", exc_info=True)
        results["pro30"] = None
    
    # Step 5: LLM Ranking & Hybrid Analysis (with optional Bull/Bear Debate)
    logger.info("\n[5/7] LLM Ranking & Hybrid Analysis...")
    debate_analysis = {}  # Store debate results for later use
    try:
        # Load primary packets (configurable; fallback to other source)
        use_swing_primary = bool(config.get("swing_strategy", {}).get("use_as_primary", True))
        primary_source = "swing" if use_swing_primary else "weekly"
        packets_file = None
        if primary_source == "swing":
            if results.get("swing") and results["swing"].get("packets_json"):
                packets_file = results["swing"]["packets_json"]
            elif results.get("weekly") and results["weekly"].get("packets_json"):
                primary_source = "weekly"
                packets_file = results["weekly"]["packets_json"]
        else:
            if results.get("weekly") and results["weekly"].get("packets_json"):
                packets_file = results["weekly"]["packets_json"]
            elif results.get("swing") and results["swing"].get("packets_json"):
                primary_source = "swing"
                packets_file = results["swing"]["packets_json"]
        
        if packets_file:
            with open(packets_file, "r") as f:
                packets_data = json.load(f)
            packets = packets_data.get("packets", [])
            if not packets and primary_source == "swing" and results.get("weekly") and results["weekly"].get("packets_json"):
                primary_source = "weekly"
                with open(results["weekly"]["packets_json"], "r") as f:
                    packets_data = json.load(f)
                packets = packets_data.get("packets", [])
            
            if packets:
                # Rank candidates
                model = args.model or "gpt-5.2"
                
                # Check if debate mode is enabled (use debate for GPT-5.2 by default)
                # --no-debate flag disables it
                no_debate = getattr(args, "no_debate", False)
                use_debate = (not no_debate) and DEBATE_AVAILABLE
                debate_rounds = getattr(args, "debate_rounds", 1)
                
                if use_debate and model in ["gpt-5.2", "gpt-4o", "gpt-4-turbo"]:
                    logger.info("  Using advanced ranking with Bull/Bear debate...")
                    llm_result = rank_with_debate(
                        packets=packets,
                        provider=args.provider,
                        model=model,
                        api_key=args.api_key,
                        debate_rounds=debate_rounds,
                        debate_top_n=10,
                        use_memory=True,
                    )
                    debate_analysis = llm_result.get("debate_analysis", {})
                else:
                    llm_result = rank_weekly_candidates(
                        packets=packets,
                        provider=args.provider,
                        model=model,
                        api_key=args.api_key,
                    )
                all_top5 = llm_result.get("top5", [])
                
                # Apply rank-based filtering (based on backtest: Rank 1=38%, Rank 2,4=19%)
                weekly_filters = config.get("quality_filters_weekly", {})
                top_ranks_only = weekly_filters.get("top_ranks_only", 5)  # Default: all 5
                if top_ranks_only < 5 and all_top5:
                    filtered_top5 = [item for item in all_top5 if item.get("rank", 99) <= top_ranks_only]
                    logger.info(f"  📊 Rank filter: Keeping ranks 1-{top_ranks_only} ({len(filtered_top5)} of {len(all_top5)} picks)")
                    results["llm_primary_top5"] = filtered_top5
                    llm_result["top5"] = filtered_top5
                    llm_result["rank_filter_applied"] = f"Top {top_ranks_only} only"
                else:
                    results["llm_primary_top5"] = all_top5
                
                # Save LLM results
                llm_result["primary_label"] = "Swing" if primary_source == "swing" else "Weekly"
                if primary_source == "swing":
                    top5_file = output_dir / f"swing_top5_{output_date_str}.json"
                else:
                    top5_file = output_dir / f"weekly_scanner_top5_{output_date_str}.json"
                save_json(llm_result, top5_file)
                
                # Save debate analysis separately if available
                if debate_analysis:
                    debate_file = output_dir / f"debate_analysis_{output_date_str}.json"
                    save_json({
                        "date": output_date_str,
                        "model": model,
                        "debate_rounds": debate_rounds,
                        "analysis": debate_analysis,
                    }, debate_file)
                    logger.info(f"  ✓ Debate analysis saved to {debate_file.name}")
                
                results["llm_primary_source"] = primary_source
                results["llm_primary_top5_file"] = str(top5_file)
                logger.info(f"  ✓ LLM ranking complete (source: {primary_source})")
            else:
                logger.warning("  ⚠ No packets found for LLM ranking")
                results["llm_primary_top5"] = []
        else:
            logger.warning("  ⚠ Primary packets not available")
            results["llm_primary_top5"] = []
    except Exception as e:
        logger.error(f"  ✗ LLM ranking failed: {e}", exc_info=True)
        results["llm_primary_top5"] = []
    
    # Generate Hybrid Analysis
    logger.info("\n" + "=" * 60)
    logger.info("HYBRID ANALYSIS - Cross-Referenced Results")
    logger.info("=" * 60)
    
    # Load 30-day candidates
    pro30_tickers = set()
    if results.get("pro30") and results["pro30"].get("candidates_csv"):
        try:
            pro30_df = pd.read_csv(results["pro30"]["candidates_csv"])
            if not pro30_df.empty and "Ticker" in pro30_df.columns:
                pro30_tickers = set(pro30_df["Ticker"].tolist())
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
            logger.debug(f"Pro30 CSV read skipped: {e}")
    
    # Get primary top 5 tickers (Swing preferred)
    primary_top5_tickers = set()
    for item in results.get("llm_primary_top5", []):
        if isinstance(item, dict) and "ticker" in item:
            primary_top5_tickers.add(item["ticker"])
    
    # Get movers
    movers_tickers = set(results.get("movers", {}).get("tickers", []))
    
    # Find overlaps
    overlap_primary_pro30 = primary_top5_tickers.intersection(pro30_tickers)
    overlap_primary_movers = primary_top5_tickers.intersection(movers_tickers)
    overlap_pro30_movers = pro30_tickers.intersection(movers_tickers)
    overlap_all_three = primary_top5_tickers.intersection(pro30_tickers).intersection(movers_tickers)
    
    # Print summary
    primary_label = "Swing" if results.get("llm_primary_source") == "swing" else "Weekly"
    primary_regime = None
    if results.get("swing") and results["swing"].get("regime"):
        primary_regime = results["swing"]["regime"]
    logger.info("\n📊 Results Summary:")
    logger.info(f"  {primary_label} Top 5: {len(primary_top5_tickers)} tickers")
    logger.info(f"  30-Day Candidates: {len(pro30_tickers)} tickers")
    logger.info(f"  Daily Movers: {len(movers_tickers)} tickers")
    
    logger.info("\n🎯 Overlap Analysis (Higher Conviction):")
    if overlap_all_three:
        logger.info(f"  ⭐ ALL THREE (Highest Conviction): {len(overlap_all_three)} tickers")
        for t in sorted(overlap_all_three):
            logger.info(f"    - {t}")
    
    if overlap_primary_pro30:
        logger.info(f"  🔥 {primary_label} + 30-Day: {len(overlap_primary_pro30)} tickers")
        for t in sorted(overlap_primary_pro30 - overlap_all_three):
            logger.info(f"    - {t}")
    
    if overlap_primary_movers:
        logger.info(f"  📈 {primary_label} + Movers: {len(overlap_primary_movers)} tickers")
        for t in sorted(overlap_primary_movers - overlap_all_three):
            logger.info(f"    - {t}")
    
    if overlap_pro30_movers:
        logger.info(f"  💎 30-Day + Movers: {len(overlap_pro30_movers)} tickers")
        for t in sorted(overlap_pro30_movers - overlap_all_three):
            logger.info(f"    - {t}")
    
    # Save hybrid results
    # output_dir already created above (output_date_str folder)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # PRO30 WEIGHTING: Based on backtest, Pro30 has 33.3% hit rate vs Weekly's 27.6%
    # Give Pro30 picks 2x weight in final scoring
    # ═══════════════════════════════════════════════════════════════════════════════
    weighted_picks = []
    all_tickers = primary_top5_tickers | pro30_tickers | movers_tickers
    
    for ticker in all_tickers:
        hybrid_score = 0.0
        sources = []
        
        # Primary Top 5 contribution (Swing preferred)
        if ticker in primary_top5_tickers:
            primary_item = next((x for x in results.get("llm_primary_top5", []) if x.get("ticker") == ticker), None)
            hw = config.get("hybrid_weighting", {})
            if primary_label == "Swing":
                base_weight = float(hw.get("swing_weight", 1.2))
                rank1_bonus = float(hw.get("swing_rank1_bonus", 0.6))
                rank2_bonus = float(hw.get("swing_rank2_bonus", 0.3))
            else:
                base_weight = float(hw.get("weekly_weight", 1.0))
                rank1_bonus = float(hw.get("weekly_rank1_bonus", 0.5))
                rank2_bonus = float(hw.get("weekly_rank2_bonus", 0.2))
            rank = primary_item.get("rank", 5) if primary_item else 5
            primary_score = base_weight
            if rank == 1:
                primary_score += rank1_bonus
            elif rank == 2:
                primary_score += rank2_bonus
            hybrid_score += primary_score
            sources.append(f"{primary_label}({rank if primary_item else '?'})")
        
        # Pro30 contribution (weight: 2.0 based on higher hit rate)
        if ticker in pro30_tickers:
            hybrid_score += 2.0
            sources.append("Pro30")
        
        # Movers contribution (weight: 0.5 - currently 0% hit rate in backtest)
        if ticker in movers_tickers:
            hybrid_score += 0.5
            sources.append("Movers")
        
        # Overlap bonuses
        hw = config.get("hybrid_weighting", {})
        if ticker in overlap_all_three:
            hybrid_score += float(hw.get("all_three_overlap_bonus", 3.0))
        elif ticker in overlap_primary_pro30:
            hybrid_score += float(hw.get("weekly_pro30_overlap_bonus", 1.5))
        
        weighted_picks.append({
            "ticker": ticker,
            "hybrid_score": hybrid_score,
            "sources": sources,
            "in_all_three": ticker in overlap_all_three,
            "in_primary_pro30": ticker in overlap_primary_pro30,
        })
    
    # Sort by hybrid score
    weighted_picks.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Log top weighted picks
    if weighted_picks:
        logger.info(f"\n📊 Weighted Rankings (Pro30 + {primary_label} + Movers):")
        for i, pick in enumerate(weighted_picks[:10], 1):
            sources_str = ", ".join(pick["sources"])
            logger.info(f"  {i}. {pick['ticker']}: {pick['hybrid_score']:.1f} pts [{sources_str}]")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HYBRID TOP 3: Best picks across ALL models (weighted by hit rate)
    # Pro30 has ~50% hit rate, Weekly ~25%, Movers ~0%
    # ═══════════════════════════════════════════════════════════════════════════════
    hybrid_top3 = []
    for pick in weighted_picks[:3]:
        ticker = pick["ticker"]
        hybrid_entry = {
            "ticker": ticker,
            "hybrid_score": pick["hybrid_score"],
            "sources": pick["sources"],
            "rank": weighted_picks.index(pick) + 1,
        }
        
        # Try to get detailed info from weekly or pro30 data
        primary_item = next((x for x in results.get("llm_primary_top5", []) if x.get("ticker") == ticker), None)
        if primary_item:
            # Copy relevant fields from primary packet
            hybrid_entry.update({
                "name": primary_item.get("name", ""),
                "sector": primary_item.get("sector", ""),
                "current_price": primary_item.get("current_price", 0),
                "composite_score": primary_item.get("composite_score", 0),
                "confidence": primary_item.get("confidence", "SPECULATIVE"),
                "primary_catalyst": primary_item.get("primary_catalyst", {}),
                "scores": primary_item.get("scores", {}),
                "evidence": primary_item.get("evidence", {}),
                "target": primary_item.get("target", {}),
                "risk_factors": primary_item.get("risk_factors", []),
                "data_gaps": primary_item.get("data_gaps", []),
            })
        else:
            # For Pro30/Movers only picks, get basic info from packets if available
            hybrid_entry.update({
                "name": "",
                "sector": "",
                "current_price": 0,
                "composite_score": pick["hybrid_score"],  # Use hybrid score
                "confidence": "MEDIUM" if "Pro30" in pick["sources"] else "SPECULATIVE",
            })
        
        hybrid_top3.append(hybrid_entry)
    
    # Store hybrid top 3 in results for downstream use
    results["hybrid_top3"] = hybrid_top3
    
    logger.info("\n🎯 HYBRID TOP 3 (Best Across All Models):")
    for item in hybrid_top3:
        sources_str = ", ".join(item.get("sources", []))
        name = item.get("name", "")[:25] or "(Pro30/Movers)"
        logger.info(f"  {item['rank']}. {item['ticker']} ({name}) — Hybrid: {item['hybrid_score']:.1f} pts [{sources_str}]")
    
    # Generate comprehensive HTML report
    try:
        html_file = generate_html_report(output_dir, output_date_str)
        html_path = html_file.resolve() if html_file else None
    except Exception as e:
        logger.warning(f"\n⚠ HTML report generation failed: {e}", exc_info=True)
        html_file = None
        html_path = None
    
    hybrid_file = output_dir / f"hybrid_analysis_{output_date_str}.json"
    hybrid_data = {
        "date": output_date_str,
        "asof_trading_date": asof_date.strftime("%Y-%m-%d") if asof_date else None,
        "summary": {
            "primary_top5_count": len(primary_top5_tickers),
            "weekly_top5_count": len(primary_top5_tickers),
            "pro30_candidates_count": len(pro30_tickers),
            "movers_count": len(movers_tickers),
            "hybrid_top3_count": len(hybrid_top3),
        },
        "overlaps": {
            "all_three": sorted(list(overlap_all_three)),
            "primary_pro30": sorted(list(overlap_primary_pro30 - overlap_all_three)),
            "primary_movers": sorted(list(overlap_primary_movers - overlap_all_three)),
            "pro30_movers": sorted(list(overlap_pro30_movers - overlap_all_three)),
            # Backward-compatible keys
            "weekly_pro30": sorted(list(overlap_primary_pro30 - overlap_all_three)),
            "weekly_movers": sorted(list(overlap_primary_movers - overlap_all_three)),
        },
        "hybrid_top3": hybrid_top3,  # NEW: Best 3 picks across all models
        "weighted_picks": weighted_picks[:20],  # Top 20 by hybrid score
        "weighting_note": f"Primary={primary_label}, Pro30, Movers, overlaps (configurable)",
        "primary_label": primary_label,
        "primary_top5": results.get("llm_primary_top5", []),
        "pro30_tickers": sorted(list(pro30_tickers)),
        "movers_tickers": sorted(list(movers_tickers)),
    }
    
    save_json(hybrid_data, hybrid_file)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONVICTION RANKER (Self-Improving Model)
    # ═══════════════════════════════════════════════════════════════════════════════
    conviction_result = None
    try:
        from src.pipelines.conviction_ranker import rank_candidates, format_conviction_picks
        from src.core.adaptive_scorer import get_adaptive_scorer
        
        logger.info("\n[5.5/7] Conviction Ranker (Adaptive Model)...")
        
        scorer = get_adaptive_scorer()
        model_info = scorer.get_model_info()
        
        # Run conviction ranking
        conviction_result = rank_candidates(
            weekly_picks=results.get("llm_primary_top5", []),
            pro30_picks=list(pro30_tickers),
            movers_picks=list(movers_tickers),
            scorer=scorer,
            max_picks=3,
            min_confidence="MEDIUM",
        )
        
        # Display top conviction picks
        if conviction_result.get("top_picks"):
            formatted = format_conviction_picks(conviction_result)
            logger.info("\n" + formatted)
            
            # Store conviction result in results
            results["conviction_picks"] = conviction_result
            
            # Save conviction results
            conviction_file = output_dir / f"conviction_picks_{output_date_str}.json"
            save_json(conviction_result, conviction_file)
            logger.info(f"  ✓ Conviction picks saved to {conviction_file}")
        else:
            logger.info("  ℹ No picks meet conviction threshold today")
        
        # Check if model needs retraining
        if scorer.should_retrain():
            logger.info("  ℹ Model has new outcomes - consider running 'python main.py learn'")
        
    except ImportError as e:
        logger.debug(f"Conviction ranker not available: {e}")
    except Exception as e:
        logger.warning(f"  ⚠ Conviction ranking failed: {e}")
    
    # Open browser if requested
    if hasattr(args, 'open') and args.open and html_path:
        _open_browser(html_path)
    
    # Append to model history log
    try:
        _append_model_history(
            date_str=output_date_str,
            weekly_top5=primary_top5_tickers,
            pro30_tickers=pro30_tickers,
            movers_tickers=movers_tickers,
            overlaps=hybrid_data["overlaps"],
            weekly_top5_data=hybrid_data.get("primary_top5", []),
        )
    except Exception as e:
        logger.debug(f"Could not append to model history: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # POST-SCAN ANALYSIS (Auto-integrated)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Step 6: Quick Model Validation (run BEFORE alerts to include in notification)
    logger.info("\n[6/7] Quick Model Validation...")
    model_health_data = None
    try:
        from src.commands.validate import run_full_backtest, generate_scorecard
        from datetime import timedelta
        
        cutoff = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        val_df = run_full_backtest(
            start_date=cutoff,
            holding_periods=[5, 7],
            hit_thresholds=[5.0, 7.0, 10.0],
        )
        
        if not val_df.empty:
            scorecard = generate_scorecard(val_df, primary_period=7, primary_threshold=7)
            kpi = scorecard.get("primary_kpi", {})
            health = scorecard.get("model_health", "Unknown")
            
            logger.info(f"  ✓ Model Health: {health}")
            logger.info(f"    Hit Rate (+7%): {(kpi.get('hit_rate') or 0) * 100:.1f}%")
            logger.info(f"    Win Rate: {(kpi.get('win_rate') or 0) * 100:.1f}%")
            
            # Strategy ranking
            strategy_data = []
            for s in scorecard.get("strategy_ranking", [])[:3]:
                hr = (s.get('hit_rate') or 0) * 100
                logger.info(f"    {s['strategy']}: {hr:.1f}% hit rate (n={s['n']})")
                strategy_data.append({
                    "name": s.get('strategy'),
                    "hit_rate": s.get('hit_rate') or 0,
                    "n": s.get('n', 0)
                })
            
            # Build model health data for alerts
            model_health_data = {
                "status": health,
                "hit_rate": kpi.get('hit_rate'),
                "win_rate": kpi.get('win_rate'),
                "strategies": strategy_data
            }
        else:
            logger.info("  ⚠ Insufficient historical data for validation")
    except Exception as e:
        logger.warning(f"  ⚠ Validation skipped: {e}")
    
    # Step 7: Confluence Analysis
    logger.info("\n[7/7] Confluence Analysis (Multi-Signal Alignment)...")
    try:
        from src.pipelines.confluence import run_confluence_scan, save_confluence_results
        
        confluence_picks = run_confluence_scan(
            date=output_date_str,
            min_signals=2,
            include_options=False,  # Skip for speed
            include_sector=False,   # Skip for speed
        )
        
        if confluence_picks:
            logger.info(f"  ✓ Found {len(confluence_picks)} high-conviction picks (2+ signals)")
            for c in confluence_picks[:5]:
                sources = ", ".join(s.source for s in c.signals)
                logger.info(f"    🎯 {c.ticker}: {c.confluence_score}/10 [{sources}]")
            
            save_confluence_results(confluence_picks, date=output_date_str)
        else:
            logger.info("  ℹ No confluence picks today (no multi-signal alignment)")
    except Exception as e:
        logger.warning(f"  ⚠ Confluence scan skipped: {e}")
    
    # Auto-track positions from this scan
    # Primary: Hybrid Top 3 (best across all models by weighted scoring)
    position_alerts_summary = None
    try:
        from src.features.positions.tracker import PositionTracker, send_position_alerts
        
        tracker = PositionTracker()
        
        # Use Hybrid Top 3 as primary picks to track (best performers by hit rate)
        # Tag them with source "hybrid_top3" for tracking
        hybrid_top3_data = results.get("hybrid_top3", [])
        for pick in hybrid_top3_data:
            pick["source_type"] = "hybrid_top3"  # Mark as hybrid pick
        
        # Also track weekly picks separately (for backward compatibility)
        weekly_picks_data = results.get("llm_primary_top5", [])
        
        # Don't separately track Pro30/Movers - they're already in hybrid_top3 if ranked high enough
        # This prevents duplicate tracking
        pro30_list = []  # Skip - covered by hybrid_top3
        movers_list = []  # Skip - covered by hybrid_top3
        
        # Add conviction scores to picks if available
        if conviction_result and conviction_result.get("all_candidates"):
            conviction_map = {
                c["ticker"]: c for c in conviction_result["all_candidates"]
            }
            for pick in hybrid_top3_data:
                ticker = pick.get("ticker")
                if ticker in conviction_map:
                    pick["conviction_score"] = conviction_map[ticker].get("conviction_score")
                    pick["confidence"] = conviction_map[ticker].get("confidence")
            for pick in weekly_picks_data:
                ticker = pick.get("ticker")
                if ticker in conviction_map:
                    pick["conviction_score"] = conviction_map[ticker].get("conviction_score")
                    pick["confidence"] = conviction_map[ticker].get("confidence")
        
        # Track Hybrid Top 3 as the primary picks
        added = tracker.add_positions_from_scan(
            scan_date=output_date_str,
            weekly_picks=hybrid_top3_data,  # Use Hybrid Top 3 as primary
            pro30_picks=pro30_list,
            movers_picks=movers_list,
            config=config,
        )
        
        if added > 0:
            logger.info(f"  📊 Position tracker: Added {added} new positions")
        
        # Monitor existing positions for drawdown alerts
        alerts = tracker.monitor_positions()
        if alerts:
            logger.info(f"  ⚠️ Position alerts: {len(alerts)} alerts generated")
            for alert in alerts[:3]:
                logger.info(f"    {alert['message']}")
            position_alerts_summary = {
                "count": len(alerts),
                "sample": [a.get("message", "") for a in alerts[:5]],
                "high": sum(1 for a in alerts if a.get("severity") == "high"),
                "warning": sum(1 for a in alerts if a.get("severity") == "warning"),
                "info": sum(1 for a in alerts if a.get("severity") == "info"),
            }
            
            # Send alerts if enabled
            if ALERTS_AVAILABLE:
                try:
                    alerts_cfg = config.get("alerts", {})
                    single_message_only = bool(alerts_cfg.get("single_message_only", True))
                    if alerts_cfg.get("enabled") and alerts_cfg.get("position_alerts_enabled", True) and not single_message_only:
                        send_position_alerts(alerts, config)
                    elif not alerts_cfg.get("position_alerts_enabled", True):
                        logger.debug("Position alerts disabled in config (position_alerts_enabled: false)")
                    elif single_message_only:
                        logger.debug("Position alerts suppressed (single_message_only: true)")
                except Exception as e:
                    logger.debug(f"Position alert sending failed: {e}")
        
        tracker.save()
    except Exception as e:
        logger.debug(f"Position tracking skipped: {e}")
    
    # Build primary summary data for CLI/alerts
    primary_label = "Swing" if results.get("llm_primary_source") == "swing" else "Weekly"
    primary_top5 = results.get("llm_primary_top5", [])
    primary_candidates_count = 0
    if primary_label == "Swing" and results.get("swing") and results["swing"].get("candidates_csv"):
        if results["swing"].get("candidates_count") is not None:
            primary_candidates_count = int(results["swing"].get("candidates_count") or 0)
        else:
            try:
                swing_df = pd.read_csv(results["swing"]["candidates_csv"])
                primary_candidates_count = int(len(swing_df))
            except Exception:
                primary_candidates_count = 0
    elif primary_label == "Weekly" and results.get("weekly") and results["weekly"].get("candidates_csv"):
        try:
            weekly_df = pd.read_csv(results["weekly"]["candidates_csv"])
            primary_candidates_count = int(len(weekly_df))
        except Exception:
            primary_candidates_count = 0

    # Send single consolidated alert after all steps (if enabled)
    if ALERTS_AVAILABLE:
        try:
            alerts_raw = config.get("alerts", {})
            alert_kwargs = {
                "enabled": alerts_raw.get("enabled", False),
                "channels": alerts_raw.get("channels", []),
                "slack_webhook": alerts_raw.get("slack_webhook"),
                "discord_webhook": alerts_raw.get("discord_webhook"),
                "alert_log_path": alerts_raw.get("alert_log_path", "outputs/alerts.log"),
            }
            email_cfg = alerts_raw.get("email", {})
            if email_cfg:
                alert_kwargs["smtp_host"] = email_cfg.get("smtp_host", "smtp.gmail.com")
                alert_kwargs["smtp_port"] = email_cfg.get("smtp_port", 587)
                alert_kwargs["from_address"] = email_cfg.get("from_address")
                alert_kwargs["to_addresses"] = email_cfg.get("to_addresses", [])
            triggers_cfg = alerts_raw.get("triggers", {})
            if triggers_cfg:
                alert_kwargs["trigger_all_three_overlap"] = triggers_cfg.get("all_three_overlap", True)
                alert_kwargs["trigger_weekly_pro30_overlap"] = triggers_cfg.get("weekly_pro30_overlap", True)
                alert_kwargs["trigger_high_composite_score"] = triggers_cfg.get("high_composite_score", 7.0)
            
            alert_config = AlertConfig(**alert_kwargs)
            if alert_config.enabled:
                send_run_summary_alert(
                    date_str=output_date_str,
                    weekly_count=len(primary_top5_tickers),
                    pro30_count=len(pro30_tickers),
                    movers_count=len(movers_tickers),
                    overlaps=hybrid_data["overlaps"],
                    config=alert_config,
                    weekly_tickers=sorted(list(primary_top5_tickers)),
                    pro30_tickers=sorted(list(pro30_tickers)),
                    movers_tickers=sorted(list(movers_tickers)),
                    model_health=model_health_data,
                    weekly_top5_data=primary_top5,
                    hybrid_top3=results.get("hybrid_top3", []),
                    primary_label=primary_label,
                    primary_candidates_count=primary_candidates_count,
                    position_alerts=position_alerts_summary,
                    regime=primary_regime,
                )
                logger.info("Alerts sent successfully")
        except Exception as e:
            logger.warning(f"Failed to send alerts: {e}")

    # CLI Summary (comprehensive)
    summary_lines = [
        f"Run Date: {output_date_str} | As-of: {asof_date.strftime('%Y-%m-%d') if asof_date else 'N/A'}",
        f"Primary Strategy: {primary_label} | Candidates: {primary_candidates_count} | Top5: {len(primary_top5_tickers)}",
        f"Secondary Pools: Pro30={len(pro30_tickers)} | Movers={len(movers_tickers)}",
        f"Overlaps: AllThree={len(overlap_all_three)} | {primary_label}+Pro30={len(overlap_primary_pro30)} | {primary_label}+Movers={len(overlap_primary_movers)}",
    ]
    if primary_top5:
        summary_lines.append("Primary Top Picks:")
        for item in primary_top5[:5]:
            ticker = item.get("ticker", "?")
            score = item.get("composite_score")
            if score is None:
                score = item.get("swing_score", 0) or 0
            try:
                score = float(score)
            except Exception:
                score = 0.0
            verdict = item.get("verdict") or item.get("confidence", "")
            summary_lines.append(f"  - {ticker}: score={score:.2f} {verdict}".rstrip())
    if model_health_data:
        summary_lines.append(f"Model Health: {model_health_data.get('status', 'Unknown')}")
        hit_rate = model_health_data.get("hit_rate")
        win_rate = model_health_data.get("win_rate")
        if hit_rate is not None and win_rate is not None:
            summary_lines.append(f"  Hit={hit_rate * 100:.1f}% | Win={win_rate * 100:.1f}%")
    if position_alerts_summary:
        summary_lines.append(f"Position Alerts: {position_alerts_summary.get('count', 0)} (high={position_alerts_summary.get('high', 0)}, warning={position_alerts_summary.get('warning', 0)})")
    if html_path:
        summary_lines.append(f"Report: {html_path}")

    logger.info("\n" + "=" * 60)
    logger.info("RUN SUMMARY")
    logger.info("=" * 60)
    for line in summary_lines:
        logger.info(line)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    return 0


def _append_model_history(
    date_str: str,
    weekly_top5: list,
    pro30_tickers: list,
    movers_tickers: list,
    overlaps: dict,
    weekly_top5_data: list,
) -> None:
    """Append run summary to model_history.md for AI analysis."""
    history_path = Path("outputs/model_history.md")
    
    # Build the entry
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    lines = [
        f"\n### {date_str} (run: {timestamp})",
        f"",
        f"**Picks:**",
        f"- Primary Top 5: {', '.join(weekly_top5) if weekly_top5 else '(none)'}",
        f"- Pro30: {', '.join(pro30_tickers[:5]) if pro30_tickers else '(none)'}" + (f" (+{len(pro30_tickers)-5} more)" if len(pro30_tickers) > 5 else ""),
        f"- Movers: {', '.join(movers_tickers) if movers_tickers else '(none)'}",
        f"",
        f"**Overlaps:**",
        f"- All Three: {', '.join(overlaps.get('all_three', [])) or '(none)'}",
        f"- Primary+Pro30: {', '.join(overlaps.get('primary_pro30', overlaps.get('weekly_pro30', []))) or '(none)'}",
        f"",
    ]
    
    # Add top pick details
    if weekly_top5_data:
        lines.append("**Top Pick Details:**")
        for item in weekly_top5_data[:3]:
            ticker = item.get("ticker", "?")
            score = item.get("composite_score", 0)
            catalyst = item.get("primary_catalyst", {}).get("title", "N/A")[:60]
            confidence = item.get("confidence", "?")
            lines.append(f"- {ticker}: score={score:.2f}, confidence={confidence}, catalyst=\"{catalyst}\"")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Append to file
    with open(history_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))



