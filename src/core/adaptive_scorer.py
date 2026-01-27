"""
Adaptive Scorer Module

Replaces static scoring with learned weights based on historical outcomes.
Adjusts conviction scores based on feature performance patterns.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

from src.utils.time import utc_now

# Default weights path
WEIGHTS_PATH = "data/model_weights.json"
MODEL_HISTORY_PATH = "outputs/model_history.md"

# Minimum observations before adjusting weights
MIN_OBSERVATIONS = 50


@dataclass
class ModelWeights:
    """
    Learned weights for adaptive scoring.
    
    OPTIMIZED Jan 2026 based on backtest data:
    - Pro30: 50% hit rate (+10% in 7d), 68.8% hit +5%
    - Weekly Rank 1: 38% hit rate
    - Weekly Rank 2: 29% hit rate
    - Movers: 0% hit rate (disabled)
    """
    version: int = 1
    last_trained: Optional[str] = None
    observations: int = 0
    overall_hit_rate: float = 0.0
    
    # Core weights (OPTIMIZED)
    overlap_bonus: float = 2.0  # Increased from 1.5 - confluence is key
    
    # Source bonuses (OPTIMIZED based on hit rates)
    # Formula: bonus ∝ log(hit_rate / avg_hit_rate)
    source_bonus: Dict[str, float] = field(default_factory=lambda: {
        "pro30": 1.2,       # INCREASED: 50% hit rate (best performer)
        "weekly_top5": 0.0, # DECREASED: 27% hit rate (below average)
        "movers": -0.5,     # PENALIZED: 0% historical hit rate
    })
    
    # Sector bonuses (populated by training)
    sector_bonus: Dict[str, float] = field(default_factory=dict)
    
    # Feature adjustments (OPTIMIZED)
    high_rsi_penalty: float = -0.5   # INCREASED: RSI>70 correlates with losers
    volume_spike_bonus: float = 0.5  # INCREASED: volume confirmation is key
    rank_1_bonus: float = 0.8        # INCREASED: Rank 1 = 38% vs avg 27%
    rank_2_bonus: float = 0.3        # NEW: Rank 2 = 29% (still above average)
    
    # RSI sweet spot bonuses (NEW)
    rsi_sweet_spot_bonus: float = 0.4  # Bonus for RSI 55-65
    rsi_acceptable_penalty: float = 0.0  # No penalty for RSI 50-70 (outside sweet spot)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "version": self.version,
            "last_trained": self.last_trained,
            "observations": self.observations,
            "overall_hit_rate": self.overall_hit_rate,
            "weights": {
                "overlap_bonus": self.overlap_bonus,
                "source_bonus": self.source_bonus,
                "sector_bonus": self.sector_bonus,
                "high_rsi_penalty": self.high_rsi_penalty,
                "volume_spike_bonus": self.volume_spike_bonus,
                "rank_1_bonus": self.rank_1_bonus,
                "rank_2_bonus": self.rank_2_bonus,
                "rsi_sweet_spot_bonus": self.rsi_sweet_spot_bonus,
                "rsi_acceptable_penalty": self.rsi_acceptable_penalty,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelWeights":
        """Create from dict."""
        weights_data = data.get("weights", {})
        return cls(
            version=data.get("version", 1),
            last_trained=data.get("last_trained"),
            observations=data.get("observations", 0),
            overall_hit_rate=data.get("overall_hit_rate", 0.0),
            overlap_bonus=weights_data.get("overlap_bonus", 2.0),
            source_bonus=weights_data.get("source_bonus", {}),
            sector_bonus=weights_data.get("sector_bonus", {}),
            high_rsi_penalty=weights_data.get("high_rsi_penalty", -0.5),
            volume_spike_bonus=weights_data.get("volume_spike_bonus", 0.5),
            rank_1_bonus=weights_data.get("rank_1_bonus", 0.8),
            rank_2_bonus=weights_data.get("rank_2_bonus", 0.3),
            rsi_sweet_spot_bonus=weights_data.get("rsi_sweet_spot_bonus", 0.4),
            rsi_acceptable_penalty=weights_data.get("rsi_acceptable_penalty", 0.0),
        )


class AdaptiveScorer:
    """
    Adaptive scoring system using learned weights.
    
    Scores candidates based on:
    - Base technical/composite score
    - Source performance (pro30 vs weekly vs movers)
    - Scanner overlap (multiple scanner hits = higher conviction)
    - Sector trends
    - RSI and volume adjustments
    
    Usage:
        scorer = AdaptiveScorer()
        score = scorer.score(features_dict)
        scorer.retrain()  # Update weights from outcomes
    """
    
    def __init__(
        self,
        weights_path: str = WEIGHTS_PATH,
        min_observations: int = MIN_OBSERVATIONS,
    ):
        self.weights_path = Path(weights_path)
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_observations = min_observations
        self.weights = self._load_weights()
    
    def _load_weights(self) -> ModelWeights:
        """Load weights from file or return defaults."""
        if self.weights_path.exists():
            try:
                with open(self.weights_path, "r") as f:
                    data = json.load(f)
                weights = ModelWeights.from_dict(data)
                logger.debug(f"Loaded model weights v{weights.version} ({weights.observations} observations)")
                return weights
            except Exception as e:
                logger.warning(f"Could not load weights, using defaults: {e}")
        
        return ModelWeights()
    
    def _save_weights(self):
        """Save weights to file."""
        try:
            with open(self.weights_path, "w") as f:
                json.dump(self.weights.to_dict(), f, indent=2)
            logger.debug(f"Saved model weights v{self.weights.version}")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
    
    def score(self, features: Dict[str, Any]) -> float:
        """
        Compute conviction score using learned weights.
        
        OPTIMIZED Jan 2026 with:
        - RSI sweet spot bonuses
        - Rank 2 bonus
        - Volume tiering
        
        Args:
            features: Dict with candidate features including:
                - technical_score or composite_score (base score)
                - source: "weekly_top5", "pro30", "movers"
                - overlap_count: number of scanners that flagged this
                - overlap_sources: list of source names
                - sector: sector name
                - rsi14: RSI(14) value
                - volume_ratio_3d_to_20d: volume ratio
                - rank: weekly pick rank (1-5)
        
        Returns:
            Conviction score (float)
        """
        # Base score from technical analysis
        # Handle None values explicitly - .get() default only applies if key is missing
        base_score = features.get("composite_score")
        if base_score is None:
            base_score = features.get("technical_score")
        if base_score is None:
            base_score = 5.0  # Fallback default
        score = float(base_score)
        
        # Source bonus
        source = features.get("source", "")
        source_bonus = self.weights.source_bonus.get(source, 0)
        score += source_bonus
        
        # Overlap bonus (per additional scanner beyond the first) - INCREASED weight
        overlap_count = features.get("overlap_count", 1)
        if overlap_count > 1:
            score += self.weights.overlap_bonus * (overlap_count - 1)
        
        # Sector bonus
        sector = features.get("sector")
        if sector and sector in self.weights.sector_bonus:
            score += self.weights.sector_bonus[sector]
        
        # RSI-based adjustments (OPTIMIZED)
        rsi = features.get("rsi14")
        if rsi is not None:
            if 55 <= rsi <= 65:
                # Sweet spot bonus
                score += self.weights.rsi_sweet_spot_bonus
            elif rsi > 70:
                # Overbought penalty (INCREASED)
                score += self.weights.high_rsi_penalty
            elif 50 <= rsi <= 70:
                # Acceptable range - minimal adjustment
                score += self.weights.rsi_acceptable_penalty
        
        # Volume spike bonus (TIERED)
        vol_ratio = features.get("volume_ratio_3d_to_20d")
        if vol_ratio is not None:
            if vol_ratio >= 2.0:
                # Institutional-grade volume
                score += self.weights.volume_spike_bonus * 1.5
            elif vol_ratio >= 1.5:
                # Standard volume confirmation
                score += self.weights.volume_spike_bonus
            elif vol_ratio < 1.0:
                # Declining volume - slight penalty
                score -= 0.2
        
        # Rank bonuses for weekly picks (OPTIMIZED)
        rank = features.get("rank")
        if source == "weekly_top5":
            if rank == 1:
                score += self.weights.rank_1_bonus
            elif rank == 2:
                score += self.weights.rank_2_bonus
        
        return round(score, 2)
    
    def compute_confidence(self, features: Dict[str, Any]) -> str:
        """
        Compute confidence level based on features and historical patterns.
        
        Returns: "HIGH", "MEDIUM", or "LOW"
        """
        confidence_score = 0
        
        # Multiple scanner overlap = higher confidence
        overlap_count = features.get("overlap_count", 1)
        if overlap_count >= 3:
            confidence_score += 3
        elif overlap_count >= 2:
            confidence_score += 2
        
        # Source reliability
        source = features.get("source", "")
        source_hit_rate = self._get_source_hit_rate(source)
        if source_hit_rate >= 0.4:
            confidence_score += 2
        elif source_hit_rate >= 0.25:
            confidence_score += 1
        
        # Technical strength (handle None values explicitly)
        tech_score = features.get("technical_score")
        if tech_score is None:
            tech_score = features.get("composite_score")
        if tech_score is None:
            tech_score = 0
        if tech_score >= 8:
            confidence_score += 2
        elif tech_score >= 6:
            confidence_score += 1
        
        # Sufficient model observations
        if self.weights.observations >= self.min_observations:
            confidence_score += 1
        
        # Map to confidence level
        if confidence_score >= 6:
            return "HIGH"
        elif confidence_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_source_hit_rate(self, source: str) -> float:
        """Get historical hit rate for a source (UPDATED Jan 2026 backtest data)."""
        # Updated hit rates based on backtest (hit +10% in 7 days)
        default_rates = {
            "pro30": 0.50,       # 50% hit rate - best performer
            "weekly_top5": 0.286, # 28.6% hit rate
            "movers": 0.00,      # 0% hit rate (disabled)
        }
        return default_rates.get(source, 0.20)
    
    def retrain(self, force: bool = False) -> Dict[str, Any]:
        """
        Update weights based on latest outcomes.
        
        Args:
            force: If True, retrain even if below min_observations
        
        Returns:
            Dict with training results
        """
        try:
            from src.analytics.feature_analyzer import FeatureAnalyzer
            
            analyzer = FeatureAnalyzer()
            n_loaded = analyzer.load_outcomes()
            
            if n_loaded < self.min_observations and not force:
                return {
                    "status": "skipped",
                    "reason": f"Insufficient observations ({n_loaded} < {self.min_observations})",
                    "observations": n_loaded,
                }
            
            # Compute optimal weights
            result = analyzer.compute_weights(min_observations=self.min_observations if not force else 10)
            
            if result.get("status") != "success":
                return result
            
            # Update weights
            old_version = self.weights.version
            new_weights = result.get("weights", {})
            
            self.weights.version = old_version + 1
            self.weights.last_trained = utc_now().strftime("%Y-%m-%d")
            self.weights.observations = n_loaded
            self.weights.overall_hit_rate = result.get("overall_hit_rate", 0)
            
            # Apply new weights
            if "overlap_bonus" in new_weights:
                self.weights.overlap_bonus = new_weights["overlap_bonus"]
            if "source_bonus" in new_weights:
                self.weights.source_bonus = new_weights["source_bonus"]
            if "sector_bonus" in new_weights:
                self.weights.sector_bonus = new_weights["sector_bonus"]
            if "high_rsi_penalty" in new_weights:
                self.weights.high_rsi_penalty = new_weights["high_rsi_penalty"]
            if "volume_spike_bonus" in new_weights:
                self.weights.volume_spike_bonus = new_weights["volume_spike_bonus"]
            
            # Save updated weights
            self._save_weights()
            
            # Log to model history
            self._log_weight_changes(old_version, result)
            
            return {
                "status": "success",
                "old_version": old_version,
                "new_version": self.weights.version,
                "observations": n_loaded,
                "overall_hit_rate": self.weights.overall_hit_rate,
                "weights": self.weights.to_dict()["weights"],
            }
            
        except ImportError as e:
            logger.error(f"Could not import feature analyzer: {e}")
            return {"status": "error", "reason": str(e)}
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {"status": "error", "reason": str(e)}
    
    def _log_weight_changes(self, old_version: int, result: Dict[str, Any]):
        """Log weight changes to model history file."""
        try:
            history_path = Path(MODEL_HISTORY_PATH)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp = utc_now().strftime("%Y-%m-%d %H:%M UTC")
            
            lines = [
                f"\n### Model Update v{old_version} → v{self.weights.version} ({timestamp})",
                "",
                f"**Training Data:** {self.weights.observations} outcomes",
                f"**Overall Hit Rate:** {self.weights.overall_hit_rate*100:.1f}%",
                "",
                "**Weight Changes:**",
            ]
            
            weights = self.weights.to_dict()["weights"]
            for key, value in weights.items():
                if isinstance(value, dict):
                    lines.append(f"- {key}:")
                    for k, v in value.items():
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"- {key}: {value}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
            
            with open(history_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
            
            logger.debug(f"Logged weight changes to {history_path}")
        except Exception as e:
            logger.warning(f"Could not log weight changes: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        return {
            "version": self.weights.version,
            "last_trained": self.weights.last_trained,
            "observations": self.weights.observations,
            "overall_hit_rate": self.weights.overall_hit_rate,
            "min_observations": self.min_observations,
            "needs_training": self.weights.observations < self.min_observations,
            "weights": self.weights.to_dict()["weights"],
        }
    
    def should_retrain(self, new_outcomes_count: int = 10) -> bool:
        """
        Check if model should be retrained based on new outcomes.
        
        Recommends retraining after every N new closed positions.
        """
        try:
            from src.core.outcome_db import get_outcome_db
            
            db = get_outcome_db()
            stats = db.get_outcome_stats()
            current_outcomes = stats.get("total_outcomes", 0)
            
            # Retrain if we have enough new outcomes since last training
            return current_outcomes >= self.weights.observations + new_outcomes_count
        except Exception:
            return False


# Global instance
_scorer_instance: Optional[AdaptiveScorer] = None


def get_adaptive_scorer(weights_path: str = WEIGHTS_PATH) -> AdaptiveScorer:
    """Get or create the global AdaptiveScorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = AdaptiveScorer(weights_path)
    return _scorer_instance
