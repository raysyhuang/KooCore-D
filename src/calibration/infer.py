# src/calibration/infer.py
"""
Probability inference for calibration model.

Loads trained model and predicts P(hit) for new candidates.
"""
from __future__ import annotations
import os
from typing import Optional, List, Dict, Any
import pandas as pd

# Optional sklearn/joblib imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def infer_probability(
    model_path: str,
    row: dict,
    feature_cols: Optional[List[str]] = None,
) -> Optional[float]:
    """
    Infer hit probability for a single candidate.
    
    Args:
        model_path: Path to trained model (joblib format)
        row: Dict with feature values
        feature_cols: Feature column names (if None, uses model's features)
    
    Returns:
        Probability of hit (0-1), or None if inference fails
    """
    if not JOBLIB_AVAILABLE:
        return None
    
    if not os.path.exists(model_path):
        return None
    
    try:
        model = joblib.load(model_path)
        X = pd.DataFrame([row])
        
        # Get probability of positive class (hit)
        proba = model.predict_proba(X)
        p = float(proba[0, 1])
        return p
    except Exception:
        return None


def infer_probabilities_batch(
    model_path: str,
    rows: List[Dict[str, Any]],
) -> List[Optional[float]]:
    """
    Infer hit probabilities for multiple candidates.
    
    Args:
        model_path: Path to trained model
        rows: List of dicts with feature values
    
    Returns:
        List of probabilities (None for failed inferences)
    """
    if not JOBLIB_AVAILABLE or not os.path.exists(model_path):
        return [None] * len(rows)
    
    try:
        model = joblib.load(model_path)
        X = pd.DataFrame(rows)
        probas = model.predict_proba(X)[:, 1]
        return [float(p) for p in probas]
    except Exception:
        return [None] * len(rows)


def compute_expected_value(
    prob_hit: Optional[float],
    atr_pct: Optional[float],
    target_pct: float = 10.0,
) -> Optional[float]:
    """
    Compute expected value from probability and risk-adjusted payoff.
    
    EV = prob_hit * payoff_proxy
    
    Payoff proxy adjusts for volatility (higher ATR = easier to hit target).
    
    Args:
        prob_hit: Probability of hitting target
        atr_pct: ATR as % of price
        target_pct: Target return %
    
    Returns:
        Expected value score, or None if inputs missing
    """
    if prob_hit is None:
        return None
    
    # Payoff proxy: target relative to ATR
    # Higher ATR means target is fewer ATRs away, so easier but also riskier
    if atr_pct is not None and atr_pct > 0:
        payoff = max(1.0, target_pct / max(atr_pct, 1.0))
    else:
        payoff = 1.0
    
    ev = prob_hit * payoff
    return ev


def infer_probability_regime(
    model_dir: str,
    regime: str,
    row: dict,
) -> Optional[float]:
    """
    Infer probability using regime-specific model with fallback.
    
    Tries regime model first, then falls back to global model.
    
    Args:
        model_dir: Directory containing calibration models
        regime: Current market regime ("bull", "chop", "stress")
        row: Dict with feature values
    
    Returns:
        Probability of hit, or None if no model available
    """
    if not JOBLIB_AVAILABLE or not os.path.isdir(model_dir):
        return None
    
    # Try regime-specific model first
    regime_model = os.path.join(model_dir, f"calibration_{regime}.pkl")
    if os.path.exists(regime_model):
        p = infer_probability(regime_model, row)
        if p is not None:
            return p
    
    # Fall back to global model
    global_model = os.path.join(model_dir, "calibration_model.pkl")
    if os.path.exists(global_model):
        p = infer_probability(global_model, row)
        if p is not None:
            return p
    
    return None


def infer_probabilities_regime_batch(
    model_dir: str,
    regime: str,
    rows: List[Dict[str, Any]],
) -> List[Optional[float]]:
    """
    Batch inference using regime-specific model with fallback.
    
    Args:
        model_dir: Directory containing calibration models
        regime: Current market regime
        rows: List of feature dicts
    
    Returns:
        List of probabilities
    """
    if not JOBLIB_AVAILABLE or not os.path.isdir(model_dir):
        return [None] * len(rows)
    
    # Determine which model to use
    regime_model = os.path.join(model_dir, f"calibration_{regime}.pkl")
    global_model = os.path.join(model_dir, "calibration_model.pkl")
    
    model_path = None
    if os.path.exists(regime_model):
        model_path = regime_model
    elif os.path.exists(global_model):
        model_path = global_model
    
    if model_path is None:
        return [None] * len(rows)
    
    return infer_probabilities_batch(model_path, rows)
