"""
Phase 6 Recommendation Engine

Analyzes outcome data and generates weight-update recommendations
for human review. Never auto-applies changes.

Key features:
- Bayesian smoothing for small-sample hit rates
- Per-source weight recommendations via log-ratio bonus
- Filter recommendations for data quality issues (NaN entries, stop breaches)
- Suppression rules for catastrophically bad sources
- Confidence labeling (LOW/MEDIUM/HIGH) based on sample size
"""

from __future__ import annotations
import json
import math
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Phase6RecommendationEngine:
    """
    Generates weight-update recommendations from outcome data.

    Reads closed-trade outcomes, applies Bayesian smoothing, and produces
    a JSON recommendation file for human review.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.min_observations = cfg.get("min_observations", 20)
        self.prior_weight = cfg.get("bayesian_prior_weight", 0.3)
        self.prior_hit_rate = cfg.get("bayesian_prior_hit_rate", 0.25)

        rec_cfg = cfg.get("recommendations", {})
        self.output_path = Path(rec_cfg.get("output_path", "outputs/phase6/recommendations.json"))
        self.backup_dir = Path(rec_cfg.get("backup_dir", "outputs/phase6/backups"))
        self.source_bonus_min = rec_cfg.get("source_bonus_min", -1.5)
        self.source_bonus_max = rec_cfg.get("source_bonus_max", 3.0)
        self.min_source_samples = rec_cfg.get("min_source_samples", 5)

    # ------------------------------------------------------------------
    # Bayesian smoothing
    # ------------------------------------------------------------------

    def _bayesian_smooth(self, observed_rate: float, n: int) -> float:
        """
        Pull small samples toward prior (25% baseline).

        Formula: smoothed = (n * observed + k * prior) / (n + k)
        where k = prior_weight * min_observations.
        """
        k = self.prior_weight * self.min_observations  # e.g. 0.3 * 20 = 6
        return (n * observed_rate + k * self.prior_hit_rate) / (n + k)

    # ------------------------------------------------------------------
    # Source bonus from hit rate
    # ------------------------------------------------------------------

    def _source_bonus_from_hit_rate(self, smoothed: float, overall: float) -> float:
        """
        Compute a log-ratio bonus comparing smoothed source rate to overall.

        bonus = log2(smoothed / overall) clamped to [source_bonus_min, source_bonus_max].
        If overall is zero, returns 0.
        """
        if overall <= 0 or smoothed <= 0:
            return 0.0
        ratio = smoothed / overall
        bonus = math.log2(ratio)
        return max(self.source_bonus_min, min(self.source_bonus_max, bonus))

    # ------------------------------------------------------------------
    # Confidence label
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_label(n: int) -> str:
        """Confidence based on sample size."""
        if n >= 30:
            return "HIGH"
        if n >= 15:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Per-source recommendations
    # ------------------------------------------------------------------

    def _compute_source_recommendations(
        self, df: pd.DataFrame, current_weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Compute per-source weight recommendations."""
        if df.empty:
            return []

        # Overall stats
        total = len(df)
        overall_hits = int(df["hit_7pct"].sum()) if "hit_7pct" in df.columns else 0
        overall_rate = overall_hits / total if total > 0 else 0.0

        recommendations = []

        for source, group in df.groupby("source"):
            n = len(group)
            if n < self.min_source_samples:
                continue

            hits = int(group["hit_7pct"].sum()) if "hit_7pct" in group.columns else 0
            observed_rate = hits / n if n > 0 else 0.0
            smoothed_rate = self._bayesian_smooth(observed_rate, n)
            bonus = self._source_bonus_from_hit_rate(smoothed_rate, overall_rate)
            confidence = self._confidence_label(n)

            avg_return = float(group["final_return_pct"].mean()) if "final_return_pct" in group.columns else 0.0
            max_dd = float(group["max_drawdown_pct"].min()) if "max_drawdown_pct" in group.columns else 0.0

            current = current_weights.get(source, 0.0)

            recommendations.append({
                "source": source,
                "n": n,
                "observed_hit_rate": round(observed_rate, 4),
                "smoothed_hit_rate": round(smoothed_rate, 4),
                "suggested_bonus": round(bonus, 3),
                "current_bonus": current,
                "delta": round(bonus - current, 3),
                "avg_return_pct": round(avg_return, 2),
                "worst_drawdown_pct": round(max_dd, 2),
                "confidence": confidence,
            })

        return sorted(recommendations, key=lambda r: r["suggested_bonus"], reverse=True)

    # ------------------------------------------------------------------
    # Filter recommendations (data-quality issues)
    # ------------------------------------------------------------------

    def _compute_filter_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Flag data quality issues: NaN entries, stop-loss breaches, etc."""
        filters = []

        # NaN entry prices (ghost positions)
        if "entry_price" in df.columns:
            nan_entries = df["entry_price"].isna() | (df["entry_price"] <= 0)
            nan_count = int(nan_entries.sum())
            if nan_count > 0:
                tickers = df.loc[nan_entries, "ticker"].tolist()
                filters.append({
                    "type": "nan_entry_price",
                    "severity": "HIGH",
                    "count": nan_count,
                    "tickers": tickers[:10],
                    "recommendation": "Fix position tracker to validate entry_price before creating Position",
                })

        # Stop-loss breaches (exits worse than configured stop)
        if "exit_reason" in df.columns and "max_drawdown_pct" in df.columns:
            stopped = df[df["exit_reason"] == "stopped"]
            if len(stopped) > 0:
                avg_dd = float(stopped["max_drawdown_pct"].mean())
                breach_count = int((stopped["max_drawdown_pct"] < -7.0).sum())
                if breach_count > 0:
                    filters.append({
                        "type": "stop_loss_breach",
                        "severity": "MEDIUM",
                        "count": breach_count,
                        "avg_drawdown_at_stop": round(avg_dd, 2),
                        "recommendation": "Review stop-loss monitoring frequency; "
                                          "avg stopped at {:.1f}% vs -7% target".format(avg_dd),
                    })

        # Positions held too long without exit
        if "days_held" in df.columns:
            long_holds = df[df["days_held"] > 14]
            if len(long_holds) > 3:
                filters.append({
                    "type": "excessive_holding_period",
                    "severity": "LOW",
                    "count": len(long_holds),
                    "avg_days_held": round(float(long_holds["days_held"].mean()), 1),
                    "recommendation": "Review exit discipline; some positions held well beyond target",
                })

        return filters

    # ------------------------------------------------------------------
    # Suppression rules
    # ------------------------------------------------------------------

    def _compute_suppression_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Flag sources with dangerously low smoothed hit rates for potential suppression."""
        suppressions = []

        if df.empty:
            return suppressions

        for source, group in df.groupby("source"):
            n = len(group)
            if n < self.min_source_samples:
                continue

            hits = int(group["hit_7pct"].sum()) if "hit_7pct" in group.columns else 0
            observed_rate = hits / n if n > 0 else 0.0
            smoothed_rate = self._bayesian_smooth(observed_rate, n)

            if smoothed_rate < 0.15:
                avg_return = float(group["final_return_pct"].mean()) if "final_return_pct" in group.columns else 0.0
                suppressions.append({
                    "source": source,
                    "n": n,
                    "observed_hit_rate": round(observed_rate, 4),
                    "smoothed_hit_rate": round(smoothed_rate, 4),
                    "avg_return_pct": round(avg_return, 2),
                    "confidence": self._confidence_label(n),
                    "action": "SUPPRESS",
                    "reason": f"{source} smoothed hit rate {smoothed_rate:.1%} < 15% threshold",
                })

        return suppressions

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_recommendations(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point. Reads from outcome_db, computes recommendations,
        and returns the full recommendation dict.
        """
        from src.core.outcome_db import get_outcome_db
        from src.core.adaptive_scorer import get_adaptive_scorer

        db = get_outcome_db()
        df = db.get_training_data()

        total_outcomes = len(df)
        logger.info(f"Phase 6: Loaded {total_outcomes} outcomes for analysis")

        if total_outcomes < self.min_observations:
            logger.warning(
                f"Phase 6: Only {total_outcomes} outcomes "
                f"(need {self.min_observations}). Generating low-confidence recommendations."
            )

        # Get current model weights
        scorer = get_adaptive_scorer()
        current_source_bonus = dict(scorer.weights.source_bonus)
        model_info = scorer.get_model_info()

        # Compute all recommendations
        source_recs = self._compute_source_recommendations(df, current_source_bonus)
        filter_recs = self._compute_filter_recommendations(df)
        suppression_rules = self._compute_suppression_rules(df)

        # Overall stats
        overall_hits = int(df["hit_7pct"].sum()) if "hit_7pct" in df.columns and not df.empty else 0
        overall_rate = overall_hits / total_outcomes if total_outcomes > 0 else 0.0

        rec = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "phase": 6,
            "total_outcomes": total_outcomes,
            "overall_hit_rate": round(overall_rate, 4),
            "bayesian_config": {
                "prior_weight": self.prior_weight,
                "prior_hit_rate": self.prior_hit_rate,
                "k": self.prior_weight * self.min_observations,
            },
            "current_model": {
                "version": model_info.get("version", 1),
                "source_bonus": current_source_bonus,
            },
            "source_recommendations": source_recs,
            "filter_recommendations": filter_recs,
            "suppression_rules": suppression_rules,
            "context": context or {},
        }

        return rec

    # ------------------------------------------------------------------
    # Save recommendations
    # ------------------------------------------------------------------

    def save_recommendations(self, rec: Dict[str, Any]) -> Path:
        """
        Save recommendations to JSON with backup of previous version.

        Returns the path where recommendations were saved.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup existing file
        if self.output_path.exists():
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"recommendations_{ts}.json"
            shutil.copy2(self.output_path, backup_path)
            logger.info(f"Phase 6: Backed up previous recommendations to {backup_path}")

        with open(self.output_path, "w") as f:
            json.dump(rec, f, indent=2)

        logger.info(f"Phase 6: Saved recommendations to {self.output_path}")
        return self.output_path
