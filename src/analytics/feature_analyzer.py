"""
Feature Analyzer Module

Analyzes outcome data to discover which features predict +7% winners.
Computes feature importance, optimal weights, and generates insights.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Feature columns used for analysis
NUMERIC_FEATURES = [
    "technical_score",
    "rsi14",
    "volume_ratio_3d_to_20d",
    "dist_to_52w_high_pct",
    "realized_vol_5d_ann_pct",
    "composite_score",
    "overlap_count",
    "rank",
]

BOOLEAN_FEATURES = [
    "above_ma10",
    "above_ma20",
    "above_ma50",
]

CATEGORICAL_FEATURES = [
    "source",
    "sector",
]

# Target thresholds
TARGET_HIT_7PCT = "hit_7pct"
TARGET_HIT_10PCT = "hit_10pct"


def analyze_feature_importance(
    outcomes_df: pd.DataFrame,
    target: str = TARGET_HIT_7PCT,
) -> Dict[str, float]:
    """
    Compute correlation of each feature with the target outcome.
    
    Uses point-biserial correlation for numeric features with binary target.
    
    Args:
        outcomes_df: DataFrame with outcome records
        target: Target column (default: "hit_7pct")
    
    Returns:
        Dict mapping feature name to correlation coefficient
        Positive values indicate feature is associated with wins
    """
    if outcomes_df.empty or target not in outcomes_df.columns:
        return {}
    
    # Ensure target is numeric
    target_series = outcomes_df[target].astype(float)
    
    correlations = {}
    
    # Numeric features
    for col in NUMERIC_FEATURES:
        if col in outcomes_df.columns:
            valid_mask = outcomes_df[col].notna() & target_series.notna()
            if valid_mask.sum() >= 10:  # Need at least 10 observations
                try:
                    corr = outcomes_df.loc[valid_mask, col].astype(float).corr(
                        target_series[valid_mask]
                    )
                    if not np.isnan(corr):
                        correlations[col] = round(corr, 4)
                except Exception as e:
                    logger.debug(f"Could not compute correlation for {col}: {e}")
    
    # Boolean features (convert to 0/1)
    for col in BOOLEAN_FEATURES:
        if col in outcomes_df.columns:
            valid_mask = outcomes_df[col].notna() & target_series.notna()
            if valid_mask.sum() >= 10:
                try:
                    bool_series = outcomes_df.loc[valid_mask, col].astype(float)
                    corr = bool_series.corr(target_series[valid_mask])
                    if not np.isnan(corr):
                        correlations[col] = round(corr, 4)
                except Exception as e:
                    logger.debug(f"Could not compute correlation for {col}: {e}")
    
    # Sort by absolute correlation (most predictive first)
    correlations = dict(
        sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    
    return correlations


def compute_source_hit_rates(outcomes_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute hit rates by source (weekly_top5, pro30, movers).
    
    Returns dict with per-source statistics.
    """
    if outcomes_df.empty or "source" not in outcomes_df.columns:
        return {}
    
    source_stats = {}
    
    for source in outcomes_df["source"].unique():
        mask = outcomes_df["source"] == source
        source_df = outcomes_df[mask]
        n = len(source_df)
        
        if n > 0:
            hit_7 = source_df["hit_7pct"].sum() if "hit_7pct" in source_df else 0
            hit_10 = source_df["hit_10pct"].sum() if "hit_10pct" in source_df else 0
            avg_return = source_df["max_return_pct"].mean() if "max_return_pct" in source_df else 0
            
            source_stats[source] = {
                "count": n,
                "hit_7pct_rate": hit_7 / n,
                "hit_10pct_rate": hit_10 / n,
                "avg_max_return": avg_return or 0,
            }
    
    return source_stats


def compute_overlap_bonus(outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how overlap count correlates with success.
    
    Overlap = ticker flagged by multiple scanners (weekly + pro30 + movers).
    """
    if outcomes_df.empty or "overlap_count" not in outcomes_df.columns:
        return {}
    
    results = {
        "by_overlap_count": {},
        "correlation_with_hit7": None,
        "recommended_bonus": 0.0,
    }
    
    for count in sorted(outcomes_df["overlap_count"].unique()):
        mask = outcomes_df["overlap_count"] == count
        n = mask.sum()
        if n > 0:
            hit_rate = outcomes_df.loc[mask, "hit_7pct"].mean() if "hit_7pct" in outcomes_df else 0
            results["by_overlap_count"][int(count)] = {
                "count": int(n),
                "hit_7pct_rate": float(hit_rate) if not np.isnan(hit_rate) else 0,
            }
    
    # Compute correlation
    valid = outcomes_df["overlap_count"].notna() & outcomes_df["hit_7pct"].notna()
    if valid.sum() >= 10:
        corr = outcomes_df.loc[valid, "overlap_count"].corr(
            outcomes_df.loc[valid, "hit_7pct"].astype(float)
        )
        results["correlation_with_hit7"] = float(corr) if not np.isnan(corr) else None
        
        # Recommend bonus based on correlation
        if corr and corr > 0.1:
            results["recommended_bonus"] = min(2.0, corr * 5)
    
    return results


def compute_sector_performance(outcomes_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Analyze hit rates by sector."""
    if outcomes_df.empty or "sector" not in outcomes_df.columns:
        return {}
    
    sector_stats = {}
    
    for sector in outcomes_df["sector"].dropna().unique():
        mask = outcomes_df["sector"] == sector
        n = mask.sum()
        
        if n >= 3:  # Need at least 3 observations per sector
            hit_rate = outcomes_df.loc[mask, "hit_7pct"].mean() if "hit_7pct" in outcomes_df else 0
            avg_return = outcomes_df.loc[mask, "max_return_pct"].mean() if "max_return_pct" in outcomes_df else 0
            
            sector_stats[sector] = {
                "count": int(n),
                "hit_7pct_rate": float(hit_rate) if not np.isnan(hit_rate) else 0,
                "avg_max_return": float(avg_return) if not np.isnan(avg_return) else 0,
            }
    
    # Sort by hit rate
    sector_stats = dict(
        sorted(sector_stats.items(), key=lambda x: x[1]["hit_7pct_rate"], reverse=True)
    )
    
    return sector_stats


def compute_optimal_weights(
    outcomes_df: pd.DataFrame,
    min_observations: int = 30,
) -> Dict[str, Any]:
    """
    Find weights that maximize historical hit rate.
    
    Uses a simple linear optimization approach (no scipy dependency required).
    For production, consider using scipy.optimize.
    
    Args:
        outcomes_df: DataFrame with outcome records
        min_observations: Minimum observations required
    
    Returns:
        Dict with optimized weights
    """
    if len(outcomes_df) < min_observations:
        logger.warning(f"Insufficient data for weight optimization: {len(outcomes_df)} < {min_observations}")
        return {"status": "insufficient_data", "observations": len(outcomes_df)}
    
    # Get feature correlations
    correlations = analyze_feature_importance(outcomes_df)
    
    # Get source stats
    source_stats = compute_source_hit_rates(outcomes_df)
    
    # Compute source bonuses based on hit rate differential
    overall_hit_rate = outcomes_df["hit_7pct"].mean() if "hit_7pct" in outcomes_df else 0
    source_bonus = {}
    for source, stats in source_stats.items():
        diff = stats["hit_7pct_rate"] - overall_hit_rate
        # Scale to reasonable bonus range (-1 to +1)
        source_bonus[source] = round(min(1.0, max(-1.0, diff * 3)), 2)
    
    # Get overlap analysis
    overlap_analysis = compute_overlap_bonus(outcomes_df)
    
    # Get sector analysis
    sector_stats = compute_sector_performance(outcomes_df)
    sector_bonus = {}
    for sector, stats in sector_stats.items():
        diff = stats["hit_7pct_rate"] - overall_hit_rate
        if abs(diff) > 0.05:  # Only apply bonus if significant difference
            sector_bonus[sector] = round(min(0.5, max(-0.5, diff * 2)), 2)
    
    # Compute RSI penalty (high RSI = overbought = lower hit rate)
    rsi_penalty = 0.0
    if "rsi14" in correlations:
        # Negative correlation means high RSI = lower hit rate
        rsi_corr = correlations.get("rsi14", 0)
        if rsi_corr < -0.1:
            rsi_penalty = round(rsi_corr * 2, 2)  # Scale to penalty
    
    # Volume spike bonus
    volume_bonus = 0.0
    if "volume_ratio_3d_to_20d" in correlations:
        vol_corr = correlations.get("volume_ratio_3d_to_20d", 0)
        if vol_corr > 0.1:
            volume_bonus = round(vol_corr * 2, 2)
    
    return {
        "status": "success",
        "observations": len(outcomes_df),
        "overall_hit_rate": round(overall_hit_rate, 4) if not np.isnan(overall_hit_rate) else 0,
        "weights": {
            "overlap_bonus": overlap_analysis.get("recommended_bonus", 0),
            "source_bonus": source_bonus,
            "sector_bonus": sector_bonus,
            "high_rsi_penalty": rsi_penalty,
            "volume_spike_bonus": volume_bonus,
        },
        "feature_correlations": correlations,
        "source_stats": source_stats,
        "sector_stats": sector_stats,
        "overlap_analysis": overlap_analysis,
    }


def generate_insights_report(outcomes_df: pd.DataFrame) -> str:
    """
    Generate a human-readable analysis of what's working.
    
    Returns formatted string with insights.
    """
    if outcomes_df.empty:
        return "No outcome data available for analysis."
    
    lines = []
    lines.append("=" * 60)
    lines.append("FEATURE IMPORTANCE ANALYSIS")
    lines.append("=" * 60)
    
    n = len(outcomes_df)
    hit_7 = outcomes_df["hit_7pct"].sum() if "hit_7pct" in outcomes_df else 0
    hit_rate = hit_7 / n if n > 0 else 0
    
    lines.append(f"\nTotal Observations: {n}")
    lines.append(f"Overall Hit Rate (+7%): {hit_rate*100:.1f}% ({hit_7}/{n})")
    
    # Source performance
    lines.append("\n--- BY SOURCE ---")
    source_stats = compute_source_hit_rates(outcomes_df)
    for source, stats in sorted(source_stats.items(), key=lambda x: x[1]["hit_7pct_rate"], reverse=True):
        lines.append(f"  {source}: {stats['hit_7pct_rate']*100:.1f}% hit rate (n={stats['count']})")
    
    # Overlap analysis
    lines.append("\n--- OVERLAP BONUS ---")
    overlap = compute_overlap_bonus(outcomes_df)
    for count, stats in overlap.get("by_overlap_count", {}).items():
        lines.append(f"  Overlap {count}: {stats['hit_7pct_rate']*100:.1f}% hit rate (n={stats['count']})")
    
    if overlap.get("correlation_with_hit7"):
        lines.append(f"  Correlation with success: {overlap['correlation_with_hit7']:.3f}")
        lines.append(f"  Recommended overlap bonus: +{overlap.get('recommended_bonus', 0):.2f}")
    
    # Feature correlations
    lines.append("\n--- FEATURE IMPORTANCE ---")
    correlations = analyze_feature_importance(outcomes_df)
    for feature, corr in list(correlations.items())[:10]:
        direction = "+" if corr > 0 else ""
        lines.append(f"  {feature}: {direction}{corr:.3f}")
    
    # Sector performance
    lines.append("\n--- TOP SECTORS ---")
    sector_stats = compute_sector_performance(outcomes_df)
    for sector, stats in list(sector_stats.items())[:5]:
        lines.append(f"  {sector}: {stats['hit_7pct_rate']*100:.1f}% hit rate (n={stats['count']})")
    
    # Recommendations
    lines.append("\n--- RECOMMENDATIONS ---")
    
    # Best source
    if source_stats:
        best_source = max(source_stats.items(), key=lambda x: x[1]["hit_7pct_rate"])
        worst_source = min(source_stats.items(), key=lambda x: x[1]["hit_7pct_rate"])
        if best_source[1]["hit_7pct_rate"] > worst_source[1]["hit_7pct_rate"] + 0.1:
            lines.append(f"  * Prioritize {best_source[0]} picks ({best_source[1]['hit_7pct_rate']*100:.0f}% hit rate)")
            lines.append(f"  * De-weight {worst_source[0]} picks ({worst_source[1]['hit_7pct_rate']*100:.0f}% hit rate)")
    
    # Overlap bonus
    if overlap.get("recommended_bonus", 0) > 0:
        lines.append(f"  * Apply +{overlap['recommended_bonus']:.1f} bonus per additional scanner overlap")
    
    # High correlation features
    for feature, corr in list(correlations.items())[:3]:
        if abs(corr) > 0.15:
            if corr > 0:
                lines.append(f"  * Higher {feature} correlates with wins (+{corr:.2f})")
            else:
                lines.append(f"  * Higher {feature} correlates with losses ({corr:.2f})")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


class FeatureAnalyzer:
    """
    Main class for analyzing features and computing optimal weights.
    
    Usage:
        analyzer = FeatureAnalyzer()
        analyzer.load_outcomes()
        report = analyzer.generate_report()
        weights = analyzer.compute_weights()
    """
    
    def __init__(self, db_path: str = None):
        self.outcomes_df = pd.DataFrame()
        self.db_path = db_path
    
    def load_outcomes(
        self,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Load outcome data from the database.
        
        Returns number of records loaded.
        """
        try:
            from src.core.outcome_db import get_outcome_db
            
            db = get_outcome_db(self.db_path) if self.db_path else get_outcome_db()
            self.outcomes_df = db.get_training_data(
                min_date=min_date,
                max_date=max_date,
                sources=sources,
            )
            
            logger.info(f"Loaded {len(self.outcomes_df)} outcome records for analysis")
            return len(self.outcomes_df)
        except Exception as e:
            logger.error(f"Failed to load outcomes: {e}")
            return 0
    
    def analyze(self) -> Dict[str, Any]:
        """Run full feature analysis and return results."""
        return compute_optimal_weights(self.outcomes_df)
    
    def get_correlations(self) -> Dict[str, float]:
        """Get feature correlations with hit_7pct."""
        return analyze_feature_importance(self.outcomes_df)
    
    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance by source."""
        return compute_source_hit_rates(self.outcomes_df)
    
    def get_sector_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance by sector."""
        return compute_sector_performance(self.outcomes_df)
    
    def get_overlap_analysis(self) -> Dict[str, Any]:
        """Get overlap bonus analysis."""
        return compute_overlap_bonus(self.outcomes_df)
    
    def generate_report(self) -> str:
        """Generate human-readable insights report."""
        return generate_insights_report(self.outcomes_df)
    
    def compute_weights(self, min_observations: int = 30) -> Dict[str, Any]:
        """Compute optimal weights for the adaptive scorer."""
        return compute_optimal_weights(self.outcomes_df, min_observations)
