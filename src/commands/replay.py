"""Historical replay command handler (generates outputs for past dates)."""

from __future__ import annotations

import logging
from datetime import datetime as dt

from src.core.config import load_config
from src.core.helpers import get_trading_date, iter_weekdays
from src.pipelines.weekly import run_weekly
from src.pipelines.pro30 import run_pro30
from src.core.io import save_json
from src.core.report import generate_html_report
from src.core.llm import rank_weekly_candidates

logger = logging.getLogger(__name__)


def cmd_replay(args) -> int:
    """
    Replay historical dates and write outputs/YYYY-MM-DD/ for each trading weekday.

    Important limitation:
    - News/catalyst fetching via yfinance is not point-in-time; it may reflect *current* headlines,
      not what was known on the historical date. Prices/technicals are as-of the historical date.
    
    By default, pit_mode=True is set which disables non-point-in-time enrichment:
    - News fetching is disabled (would return current headlines, not historical)
    - Sentiment fetching is disabled
    - Options data fetching is disabled
    
    This ensures replay results are based only on historical price data.
    """
    start = dt.strptime(args.start, "%Y-%m-%d").date()
    end = dt.strptime(args.end, "%Y-%m-%d").date()

    config = load_config(args.config)
    if args.no_movers:
        config["movers"]["enabled"] = False
    
    # PR1: Enable point-in-time mode by default for replay
    # This disables non-PIT enrichment (news, sentiment, options)
    pit_mode = getattr(args, 'pit_mode', True)  # Default True for replay
    if pit_mode:
        logger.info("PIT mode enabled: disabling non-point-in-time enrichment (news/sentiment/options)")
        # Disable features that are not point-in-time
        if "features" not in config:
            config["features"] = {}
        config["features"]["fetch_options"] = False
        config["features"]["fetch_sentiment"] = False
        config["features"]["fetch_news"] = False
        config["features"]["pit_mode"] = True

    provider = args.provider
    model = args.model
    api_key = args.api_key

    dates = iter_weekdays(start, end)
    if not dates:
        logger.error("No dates to replay.")
        return 1

    logger.info("=" * 60)
    logger.info(f"REPLAY {args.start} → {args.end} (weekdays only): {len(dates)} day(s)")
    logger.info("=" * 60)

    for d in dates:
        td = get_trading_date(d)
        date_str = td.strftime("%Y-%m-%d")
        logger.info("\n" + "-" * 60)
        logger.info(f"Replay date: {date_str}")

        weekly_result = run_weekly(config=config, asof_date=td)
        pro30_result = run_pro30(config=config, asof_date=td)

        # LLM Top 5 (optional)
        if args.llm:
            try:
                packets_file = weekly_result.get("packets_json")
                if packets_file:
                    import json
                    with open(packets_file, "r", encoding="utf-8") as f:
                        packets_data = json.load(f)
                    packets = packets_data.get("packets", []) or []
                    if packets:
                        llm_result = rank_weekly_candidates(
                            packets=packets,
                            provider=provider,
                            model=model,
                            api_key=api_key,
                        )
                        run_dir = weekly_result["run_dir"]
                        top5_file = run_dir / f"weekly_scanner_top5_{date_str}.json"
                        save_json(llm_result, top5_file)
                        logger.info("  ✓ LLM ranking complete")
                    else:
                        logger.warning("  ⚠ No packets found for LLM ranking")
            except Exception as e:
                logger.error(f"  ✗ LLM ranking failed: {e}", exc_info=True)

        # HTML report (also emits summary/top5/run_card)
        try:
            run_dir = weekly_result["run_dir"]
            generate_html_report(run_dir, date_str)
            logger.info("  ✓ HTML + artifacts generated")
        except Exception as e:
            logger.warning(f"  ⚠ Report generation failed: {e}")

        if args.max_days and args.max_days > 0:
            args.max_days -= 1
            if args.max_days == 0:
                logger.info("Reached --max-days limit; stopping.")
                break

    return 0

