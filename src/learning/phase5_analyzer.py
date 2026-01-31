"""
Phase-5 Analyzer

Comprehensive analysis of Phase-5 learning data.

Key analyses:
1. Source attribution (which signals work)
2. Rank decay curves (where signal strength dies)
3. Regime sensitivity (what flips in chop)
4. Overlap value analysis
5. Feature importance
6. Suppression zone identification

Design principle: Answer questions, not build models (yet).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class Phase5Analyzer:
    """
    Analyzes Phase-5 learning data to produce actionable insights.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        from src.learning.phase5_store import get_phase5_store
        self.store = get_phase5_store(base_path)
        self._df = None  # Cached merged DataFrame
    
    @property
    def df(self):
        """Get cached merged DataFrame."""
        if self._df is None:
            self._df = self.store.load_merged()
        return self._df
    
    def reload(self):
        """Reload data from storage."""
        self._df = None
        return self.df
    
    def get_summary_stats(self) -> dict:
        """Get high-level summary statistics."""
        df = self.df
        
        if df.empty:
            return {"error": "No data available"}
        
        total_rows = len(df)
        resolved = df["outcome_7d"].notna().sum()
        
        stats = {
            "total_rows": total_rows,
            "resolved_rows": resolved,
            "pending_rows": total_rows - resolved,
            "resolution_rate": resolved / max(total_rows, 1),
            "date_range": {
                "first": df["scan_date"].min(),
                "last": df["scan_date"].max(),
            },
        }
        
        # Outcome breakdown (for resolved only)
        if resolved > 0:
            resolved_df = df[df["outcome_7d"].notna()]
            
            hits = (resolved_df["outcome_7d"] == "hit").sum()
            misses = (resolved_df["outcome_7d"] == "miss").sum()
            neutral = (resolved_df["outcome_7d"] == "neutral").sum()
            
            stats["outcomes"] = {
                "hit": hits,
                "miss": misses,
                "neutral": neutral,
                "hit_rate": hits / resolved,
            }
            
            stats["returns"] = {
                "mean": resolved_df["return_7d"].mean(),
                "median": resolved_df["return_7d"].median(),
                "std": resolved_df["return_7d"].std(),
                "max": resolved_df["return_7d"].max(),
                "min": resolved_df["return_7d"].min(),
            }
        
        return stats
    
    def get_source_attribution(self) -> dict:
        """
        Analyze which signal sources produce best results.
        
        Key question: "Which source mattered?"
        """
        df = self.df
        resolved = df[df["outcome_7d"].notna()].copy()
        
        if resolved.empty:
            return {"error": "No resolved outcomes"}
        
        results = {}
        
        # By signal source
        sources = [
            ("in_swing_top5", "Swing"),
            ("in_weekly_top5", "Weekly"),
            ("in_pro30", "Pro30"),
            ("in_movers", "Movers"),
        ]
        
        for col, name in sources:
            if col not in resolved.columns:
                continue
            
            subset = resolved[resolved[col] == True]
            if len(subset) == 0:
                continue
            
            hits = (subset["outcome_7d"] == "hit").sum()
            n = len(subset)
            
            results[name] = {
                "count": n,
                "hits": hits,
                "hit_rate": hits / n,
                "avg_return": subset["return_7d"].mean(),
                "avg_drawdown": subset["max_drawdown_7d"].mean() if "max_drawdown_7d" in subset.columns else None,
            }
        
        # Source-only (exclusive attribution)
        for col, name in sources:
            if col not in resolved.columns:
                continue
            
            # Only this source, no others
            other_cols = [c for c, _ in sources if c != col and c in resolved.columns]
            mask = resolved[col] == True
            for other in other_cols:
                mask = mask & (resolved[other] == False)
            
            subset = resolved[mask]
            if len(subset) == 0:
                continue
            
            hits = (subset["outcome_7d"] == "hit").sum()
            n = len(subset)
            
            results[f"{name}_only"] = {
                "count": n,
                "hits": hits,
                "hit_rate": hits / n if n > 0 else 0,
                "avg_return": subset["return_7d"].mean() if n > 0 else 0,
            }
        
        return results
    
    def get_rank_decay_curves(self) -> dict:
        """
        Analyze how signal strength decays with rank.
        
        Key question: "Where does signal die?"
        """
        df = self.df
        resolved = df[df["outcome_7d"].notna()].copy()
        
        if resolved.empty:
            return {"error": "No resolved outcomes"}
        
        results = {}
        
        # Pro30 rank decay
        if "pro30_rank" in resolved.columns:
            pro30_decay = {}
            for rank_bucket in ["r1", "r2_3", "r4_5", "r6_10", "r11_20", "r21+"]:
                if rank_bucket == "r1":
                    mask = resolved["pro30_rank"] == 1
                elif rank_bucket == "r2_3":
                    mask = resolved["pro30_rank"].between(2, 3)
                elif rank_bucket == "r4_5":
                    mask = resolved["pro30_rank"].between(4, 5)
                elif rank_bucket == "r6_10":
                    mask = resolved["pro30_rank"].between(6, 10)
                elif rank_bucket == "r11_20":
                    mask = resolved["pro30_rank"].between(11, 20)
                else:
                    mask = resolved["pro30_rank"] > 20
                
                subset = resolved[mask & resolved["pro30_rank"].notna()]
                if len(subset) == 0:
                    continue
                
                hits = (subset["outcome_7d"] == "hit").sum()
                pro30_decay[rank_bucket] = {
                    "count": len(subset),
                    "hit_rate": hits / len(subset),
                    "avg_return": subset["return_7d"].mean(),
                }
            
            results["pro30_rank_decay"] = pro30_decay
        
        # Primary (Swing/Weekly) rank decay
        for col, name in [("swing_rank", "swing"), ("weekly_rank", "weekly")]:
            if col not in resolved.columns:
                continue
            
            decay = {}
            for rank in [1, 2, 3, 4, 5]:
                mask = resolved[col] == rank
                subset = resolved[mask]
                if len(subset) == 0:
                    continue
                
                hits = (subset["outcome_7d"] == "hit").sum()
                decay[f"r{rank}"] = {
                    "count": len(subset),
                    "hit_rate": hits / len(subset),
                    "avg_return": subset["return_7d"].mean(),
                }
            
            results[f"{name}_rank_decay"] = decay
        
        return results
    
    def get_regime_analysis(self) -> dict:
        """
        Analyze how signals perform in different regimes.
        
        Key question: "What flips in chop?"
        """
        df = self.df
        resolved = df[df["outcome_7d"].notna()].copy()
        
        if resolved.empty or "regime" not in resolved.columns:
            return {"error": "No resolved outcomes or missing regime"}
        
        results = {}
        
        for regime in ["bull", "chop", "stress"]:
            subset = resolved[resolved["regime"] == regime]
            if len(subset) == 0:
                continue
            
            hits = (subset["outcome_7d"] == "hit").sum()
            
            regime_stats = {
                "count": len(subset),
                "hit_rate": hits / len(subset),
                "avg_return": subset["return_7d"].mean(),
                "avg_drawdown": subset["max_drawdown_7d"].mean() if "max_drawdown_7d" in subset.columns else None,
            }
            
            # Per-source in this regime
            for col, name in [("in_pro30", "pro30"), ("in_swing_top5", "swing"), ("in_weekly_top5", "weekly")]:
                if col not in subset.columns:
                    continue
                
                source_subset = subset[subset[col] == True]
                if len(source_subset) > 0:
                    source_hits = (source_subset["outcome_7d"] == "hit").sum()
                    regime_stats[f"{name}_hit_rate"] = source_hits / len(source_subset)
                    regime_stats[f"{name}_count"] = len(source_subset)
            
            results[regime] = regime_stats
        
        return results
    
    def get_overlap_analysis(self) -> dict:
        """
        Analyze value of signal overlap/confluence.
        
        Key question: "Is overlap bonus too high?"
        """
        df = self.df
        resolved = df[df["outcome_7d"].notna()].copy()
        
        if resolved.empty:
            return {"error": "No resolved outcomes"}
        
        results = {}
        
        # By overlap flags
        if "overlap_all_three" in resolved.columns:
            all_three = resolved[resolved["overlap_all_three"] == True]
            if len(all_three) > 0:
                hits = (all_three["outcome_7d"] == "hit").sum()
                results["overlap_all_three"] = {
                    "count": len(all_three),
                    "hit_rate": hits / len(all_three),
                    "avg_return": all_three["return_7d"].mean(),
                }
        
        if "overlap_primary_pro30" in resolved.columns:
            primary_pro30 = resolved[
                (resolved["overlap_primary_pro30"] == True) & 
                (resolved.get("overlap_all_three", False) != True)
            ]
            if len(primary_pro30) > 0:
                hits = (primary_pro30["outcome_7d"] == "hit").sum()
                results["overlap_primary_pro30"] = {
                    "count": len(primary_pro30),
                    "hit_rate": hits / len(primary_pro30),
                    "avg_return": primary_pro30["return_7d"].mean(),
                }
        
        # By confluence score (if available)
        if "confluence_score" in resolved.columns:
            for score in [1, 2, 3]:
                subset = resolved[resolved["confluence_score"] == score]
                if len(subset) > 0:
                    hits = (subset["outcome_7d"] == "hit").sum()
                    results[f"confluence_{score}"] = {
                        "count": len(subset),
                        "hit_rate": hits / len(subset),
                        "avg_return": subset["return_7d"].mean(),
                    }
        
        return results
    
    def get_technical_bucket_analysis(self) -> dict:
        """Analyze performance by technical buckets."""
        df = self.df
        resolved = df[df["outcome_7d"].notna()].copy()
        
        if resolved.empty:
            return {"error": "No resolved outcomes"}
        
        results = {}
        
        buckets = [
            "trend_state",
            "rsi_bucket", 
            "volatility_bucket",
            "liquidity_bucket",
            "distance_52w_bucket",
        ]
        
        for bucket_col in buckets:
            if bucket_col not in resolved.columns:
                continue
            
            bucket_results = {}
            for value in resolved[bucket_col].dropna().unique():
                subset = resolved[resolved[bucket_col] == value]
                if len(subset) == 0:
                    continue
                
                hits = (subset["outcome_7d"] == "hit").sum()
                bucket_results[value] = {
                    "count": len(subset),
                    "hit_rate": hits / len(subset),
                    "avg_return": subset["return_7d"].mean(),
                }
            
            results[bucket_col] = bucket_results
        
        return results
    
    def generate_scorecard(self) -> dict:
        """
        Generate comprehensive scorecard.
        
        This is what Phase-5 produces every run.
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary_stats(),
            "source_attribution": self.get_source_attribution(),
            "rank_decay": self.get_rank_decay_curves(),
            "regime_analysis": self.get_regime_analysis(),
            "overlap_analysis": self.get_overlap_analysis(),
            "technical_buckets": self.get_technical_bucket_analysis(),
        }
    
    def save_scorecard(self, output_dir: Optional[Path] = None) -> Path:
        """Save scorecard to JSON file."""
        scorecard = self.generate_scorecard()
        
        if output_dir is None:
            output_dir = self.store.metrics_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = output_dir / f"phase5_scorecard_{date_str}.json"
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2, default=str)
        
        logger.info(f"Saved scorecard to {path}")
        return path
    
    def get_weight_recommendations(self) -> dict:
        """
        Generate weight recommendations based on analysis.
        
        This is Phase-6 territory - produce suggestions, don't apply.
        """
        source_attr = self.get_source_attribution()
        regime_analysis = self.get_regime_analysis()
        overlap_analysis = self.get_overlap_analysis()
        
        recommendations = {
            "current_analysis_date": datetime.now().isoformat(),
            "source_weights": {},
            "regime_adjustments": {},
            "overlap_adjustments": {},
            "suppression_rules": [],
        }
        
        # Source weight suggestions (baseline 1.0)
        baseline_hit_rate = 0.25  # Expected random hit rate
        
        for source, stats in source_attr.items():
            if "_only" in source or stats.get("count", 0) < 10:
                continue
            
            hit_rate = stats.get("hit_rate", 0)
            
            # Simple linear scaling: hit_rate 2x baseline = weight 2.0
            suggested_weight = hit_rate / baseline_hit_rate
            # Clamp to reasonable range
            suggested_weight = max(0.5, min(3.0, suggested_weight))
            
            recommendations["source_weights"][source] = {
                "observed_hit_rate": hit_rate,
                "sample_size": stats["count"],
                "suggested_weight": round(suggested_weight, 2),
            }
        
        # Regime-specific adjustments
        for regime, stats in regime_analysis.items():
            if stats.get("count", 0) < 20:
                continue
            
            hit_rate = stats.get("hit_rate", 0)
            
            if hit_rate < 0.15:
                recommendations["suppression_rules"].append({
                    "condition": f"regime == '{regime}'",
                    "action": "suppress_new_positions",
                    "reason": f"Low hit rate ({hit_rate:.1%}) in {regime} regime",
                })
            
            recommendations["regime_adjustments"][regime] = {
                "hit_rate": hit_rate,
                "sample_size": stats["count"],
            }
        
        return recommendations
    
    def print_report(self):
        """Print human-readable analysis report."""
        scorecard = self.generate_scorecard()
        
        print("\n" + "=" * 70)
        print("PHASE-5 LEARNING ANALYSIS REPORT")
        print("=" * 70)
        
        # Summary
        summary = scorecard.get("summary", {})
        print(f"\n📊 SUMMARY")
        print(f"   Total rows: {summary.get('total_rows', 0)}")
        print(f"   Resolved: {summary.get('resolved_rows', 0)} ({summary.get('resolution_rate', 0):.1%})")
        
        outcomes = summary.get("outcomes", {})
        if outcomes:
            print(f"   Hit rate: {outcomes.get('hit_rate', 0):.1%}")
            print(f"   Outcomes: {outcomes.get('hit', 0)} hits, {outcomes.get('miss', 0)} misses, {outcomes.get('neutral', 0)} neutral")
        
        returns = summary.get("returns", {})
        if returns:
            print(f"   Avg return: {returns.get('mean', 0):.2f}%")
        
        # Source attribution
        source = scorecard.get("source_attribution", {})
        if source:
            print(f"\n🎯 SOURCE ATTRIBUTION")
            for name, stats in source.items():
                if "_only" in name:
                    continue
                print(f"   {name}: {stats.get('hit_rate', 0):.1%} hit rate (n={stats.get('count', 0)})")
        
        # Rank decay
        decay = scorecard.get("rank_decay", {})
        if "pro30_rank_decay" in decay:
            print(f"\n📈 PRO30 RANK DECAY")
            for rank, stats in decay["pro30_rank_decay"].items():
                print(f"   {rank}: {stats.get('hit_rate', 0):.1%} (n={stats.get('count', 0)})")
        
        # Regime
        regime = scorecard.get("regime_analysis", {})
        if regime:
            print(f"\n🌡️ REGIME ANALYSIS")
            for name, stats in regime.items():
                print(f"   {name}: {stats.get('hit_rate', 0):.1%} (n={stats.get('count', 0)})")
        
        print("\n" + "=" * 70)


# =============================================================================
# Legacy Compatibility
# =============================================================================

def summarize_learning(records):
    """
    DEPRECATED: Use Phase5Analyzer instead.
    
    Kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "summarize_learning is deprecated, use Phase5Analyzer",
        DeprecationWarning,
        stacklevel=2
    )
    
    summary = {}
    
    for r in records or []:
        key = f"{r.get('source')}|{r.get('regime')}"
        summary.setdefault(key, {"count": 0, "hits": 0})
        
        summary[key]["count"] += 1
        if r.get("hit_7pct"):
            summary[key]["hits"] += 1
    
    for _, v in summary.items():
        v["hit_rate"] = round(v["hits"] / max(v["count"], 1), 2)
    
    return summary


# =============================================================================
# Singleton Access
# =============================================================================

_analyzer_instance: Optional[Phase5Analyzer] = None


def get_phase5_analyzer(base_path: Optional[Path] = None) -> Phase5Analyzer:
    """Get or create Phase5Analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None or base_path is not None:
        _analyzer_instance = Phase5Analyzer(base_path)
    return _analyzer_instance
