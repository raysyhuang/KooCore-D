"""
Phase 6 adaptive model updates overlay.

Thin wrapper that runs Phase 6 analysis during the main pipeline
when enabled. Follows the phase3/4/5 overlay pattern.
"""

from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def run_phase6_analysis(outcomes_data, context, cfg) -> Dict[str, Any]:
    """
    Run Phase 6 recommendation analysis.

    Returns empty dict if phase6 is disabled.
    Otherwise generates and saves recommendations.
    """
    if not _get(cfg, "enabled", False):
        return {}

    try:
        from src.learning.phase6_engine import Phase6RecommendationEngine

        engine = Phase6RecommendationEngine(cfg if isinstance(cfg, dict) else {})
        rec = engine.generate_recommendations(context)
        engine.save_recommendations(rec)

        logger.info(
            f"  Phase 6: {rec['total_outcomes']} outcomes analyzed, "
            f"{len(rec['source_recommendations'])} source recs, "
            f"{len(rec['suppression_rules'])} suppression rules"
        )

        return rec

    except Exception as e:
        logger.warning(f"Phase 6 analysis failed: {e}")
        return {}
