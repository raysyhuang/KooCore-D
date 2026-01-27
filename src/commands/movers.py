"""Daily movers command handler."""

from __future__ import annotations
import logging
from datetime import datetime
from src.core.config import load_config
from src.core.universe import build_universe
from src.features.movers.daily_movers import compute_daily_movers_from_universe
from src.features.movers.mover_filters import filter_movers
from src.features.movers.mover_queue import (
    update_mover_queue, get_eligible_movers, load_mover_queue, save_mover_queue
)
from src.core.helpers import get_ny_date
from src.utils.time import utc_now

logger = logging.getLogger(__name__)


def cmd_movers(args) -> int:
    """Run Daily Movers discovery only."""
    import os
    logger.info("=" * 60)
    logger.info("Daily Movers Discovery")
    logger.info("=" * 60)
    
    config = load_config(args.config)
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
    logger.info(f"Universe: {len(universe)} tickers")
    
    movers_config = config.get("movers", {})
    runtime_config = config.get("runtime", {})
    polygon_api_key = os.environ.get("POLYGON_API_KEY")
    from src.core.helpers import get_trading_date
    movers_raw = compute_daily_movers_from_universe(
        universe, 
        top_n=movers_config.get("top_n", 50), 
        asof_date=get_trading_date(get_ny_date()),  # Use last trading day, not today
        polygon_api_key=polygon_api_key,
        use_polygon_primary=bool(runtime_config.get("polygon_primary", False) and polygon_api_key),
        polygon_max_workers=runtime_config.get("polygon_max_workers", 8),
    )
    movers_filtered = filter_movers(movers_raw, technicals_df=None, config=movers_config)
    
    queue_df = load_mover_queue()
    queue_df = update_mover_queue(movers_filtered, utc_now(), movers_config)
    save_mover_queue(queue_df)
    
    eligible = get_eligible_movers(queue_df, utc_now())
    logger.info(f"\n✓ Eligible movers: {len(eligible)} tickers")
    if eligible:
        for t in eligible[:20]:
            logger.info(f"  - {t}")
    
    return 0

