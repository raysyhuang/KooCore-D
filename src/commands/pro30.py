"""30-day screener command handler."""

from __future__ import annotations
import logging
from pathlib import Path
from src.pipelines.pro30 import run_pro30
from src.core.config import load_config
from src.core.helpers import get_trading_date

logger = logging.getLogger(__name__)


def _apply_legacy_preset(config: dict) -> dict:
    """
    Apply legacy thresholds to approximate old Pro30 behavior.
    """
    cfg = dict(config)
    cfg.setdefault("liquidity", {})
    cfg["liquidity"]["min_avg_dollar_volume_20d"] = 20_000_000
    cfg.setdefault("quality_filters_30d", {})
    cfg["quality_filters_30d"]["min_score"] = 0.0
    cfg.setdefault("movers", {})
    cfg["movers"]["enabled"] = False
    return cfg


def cmd_pro30(args) -> int:
    """Run 30-Day Momentum Screener."""
    logger.info("=" * 60)
    logger.info("30-Day Momentum Screener")
    logger.info("=" * 60)
    
    config = load_config(args.config)
    if args.no_movers:
        config["movers"]["enabled"] = False
    if getattr(args, "legacy", False):
        config = _apply_legacy_preset(config)
    if getattr(args, "intraday_attention", False):
        config.setdefault("attention_pool", {})
        config["attention_pool"]["enable_intraday"] = True
    if getattr(args, "allow_partial_day", False):
        config.setdefault("runtime", {})
        config["runtime"]["allow_partial_day_attention"] = True

    asof_date = get_trading_date()
    if getattr(args, "date", None):
        from datetime import datetime as dt
        asof_date = get_trading_date(dt.strptime(args.date, "%Y-%m-%d").date())
    
    try:
        result = run_pro30(config=config, asof_date=asof_date)
        run_dir = Path(result['run_dir'])
        date_str = run_dir.name
        
        logger.info("\n" + "=" * 60)
        logger.info("RUN COMPLETE — Summary")
        logger.info("=" * 60)
        logger.info(f"\n📁 Output Directory: {run_dir.resolve()}")
        
        # Show candidate counts
        if result.get('candidates_csv'):
            try:
                import pandas as pd
                df = pd.read_csv(result['candidates_csv'])
                logger.info(f"  - Candidates: {len(df)} tickers")
            except Exception:
                pass
        
        if result.get('breakout_csv'):
            try:
                import pandas as pd
                df = pd.read_csv(result['breakout_csv'])
                logger.info(f"  - Breakouts: {len(df)} tickers")
            except Exception:
                pass
        
        if result.get('reversal_csv'):
            try:
                import pandas as pd
                df = pd.read_csv(result['reversal_csv'])
                logger.info(f"  - Reversals: {len(df)} tickers")
            except Exception:
                pass
        
        # Check for HTML report
        html_file = run_dir / f"report_{date_str}.html"
        if html_file.exists():
            logger.info(f"\n📄 HTML Report: {html_file.resolve()}")
        
        # Regime summary
        regime_info = result.get("regime_info") or {}
        if regime_info:
            logger.info("\nMarket regime snapshot:")
            logger.info(f"  - OK: {regime_info.get('ok', True)}")
            if regime_info.get("message"):
                logger.info(f"  - Message: {regime_info.get('message')}")
            if "spy_ma" in regime_info or "vix" in regime_info:
                logger.info(f"  - Details: {regime_info}")
        
        return 0
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1

