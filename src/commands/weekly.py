"""Weekly scanner command handler."""

from __future__ import annotations
import logging
from pathlib import Path
from src.pipelines.weekly import run_weekly
from src.core.config import load_config

logger = logging.getLogger(__name__)


def cmd_weekly(args) -> int:
    """Run Weekly Momentum Scanner."""
    logger.info("=" * 60)
    logger.info("Weekly Momentum Scanner")
    logger.info("=" * 60)
    
    config = load_config(args.config)
    if args.no_movers:
        config["movers"]["enabled"] = False

    asof_date = None
    if getattr(args, "date", None):
        from datetime import datetime as dt
        from src.core.helpers import get_trading_date
        asof_date = get_trading_date(dt.strptime(args.date, "%Y-%m-%d").date())
    
    try:
        result = run_weekly(config=config, asof_date=asof_date)
        run_dir = Path(result['run_dir'])
        date_str = run_dir.name
        
        logger.info("\n" + "=" * 60)
        logger.info("RUN COMPLETE — Summary")
        logger.info("=" * 60)
        logger.info(f"\n📁 Output Directory: {run_dir.resolve()}")
        logger.info(f"  - Candidates: {result.get('candidates_csv', 'N/A')}")
        logger.info(f"  - Packets: {result.get('packets_json', 'N/A')}")
        
        # Check for HTML report
        html_file = run_dir / f"report_{date_str}.html"
        if html_file.exists():
            logger.info(f"\n📄 HTML Report: {html_file.resolve()}")
        
        logger.info(f"\nNext: python main.py llm --date {date_str}")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1

