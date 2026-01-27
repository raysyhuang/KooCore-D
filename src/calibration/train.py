# src/calibration/train.py
"""
Calibration model training.

Trains a probability model to predict P(+10% within N days) from features.
Uses logistic regression for simplicity and calibration quality.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List, Optional
import pandas as pd
import numpy as np

# Optional sklearn imports with fallback
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import brier_score_loss, roc_auc_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Governance imports
from .guardrails import (
    is_model_eligible,
    check_eligibility_details,
    filter_incomplete_rows,
    validate_no_future_features,
    FEATURE_COMPLETENESS_THRESHOLD,
)

from src.utils.time import utc_now, utc_now_iso_z


# Default feature columns for calibration
FEATURE_COLS = [
    "technical_score",
    "breakout_score",
    "momentum_adj",
    "rsi14",
    "atr_pct",
    "vol_ratio_3_20",
    "dist_52w_high_pct",
]


def train_calibration_model(
    df: pd.DataFrame,
    model_path: str,
    meta_path: str,
    feature_cols: Optional[List[str]] = None,
    is_regime_model: bool = True,
    regime: Optional[str] = None,
) -> dict:
    """
    Train a logistic regression calibration model.
    
    Args:
        df: Training data with features and 'hit' target column
        model_path: Path to save model (joblib format)
        meta_path: Path to save metadata (JSON)
        feature_cols: Feature column names (defaults to FEATURE_COLS)
        is_regime_model: Whether this is a regime-specific model
        regime: Regime name for metadata
    
    Returns:
        Dict with training metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for calibration training. Install with: pip install scikit-learn")
    
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    
    # G3.2: Validate no look-ahead features
    validate_no_future_features(df.columns.tolist())
    
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        raise ValueError(f"No feature columns found in data. Expected: {feature_cols}")
    
    # G1.2: Filter incomplete rows
    df = filter_incomplete_rows(df, available_cols, FEATURE_COMPLETENESS_THRESHOLD)
    
    # Drop rows with missing features or target
    df_clean = df.dropna(subset=available_cols + ["hit"])
    
    # G1.1: Check eligibility
    n_rows = len(df_clean)
    n_pos = int(df_clean["hit"].sum()) if "hit" in df_clean.columns else 0
    eligibility = check_eligibility_details(n_rows, n_pos, is_regime_model)
    
    if not eligibility["eligible"]:
        raise ValueError(
            f"Model eligibility failed: {eligibility['skip_reason']}. "
            f"Rows: {n_rows}, Positives: {n_pos}"
        )
    
    if len(df_clean) < 50:
        raise ValueError(f"Insufficient training data: {len(df_clean)} rows (need >= 50)")
    
    X = df_clean[available_cols]
    y = df_clean["hit"].astype(int)
    
    # Build pipeline with scaling
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    
    pipe.fit(X, y)
    
    # Compute training metrics
    y_pred_proba = pipe.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, y_pred_proba)
    auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else None
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True) if os.path.dirname(model_path) else None
    os.makedirs(os.path.dirname(meta_path), exist_ok=True) if os.path.dirname(meta_path) else None
    
    # Save model
    joblib.dump(pipe, model_path)
    
    # G4.2: Build version metadata with full audit info
    train_date = utc_now().strftime("%Y-%m-%d")
    
    # Get training date range
    train_start = None
    train_end = None
    if "asof_date" in df_clean.columns:
        dates = pd.to_datetime(df_clean["asof_date"])
        train_start = str(dates.min().date()) if not dates.empty else None
        train_end = str(dates.max().date()) if not dates.empty else None
    
    # Get git commit if available
    try:
        from ..governance.retrain_policy import get_git_commit_hash, generate_model_version
        git_commit = get_git_commit_hash()
        model_version = generate_model_version(regime or "global", train_date)
    except ImportError:
        git_commit = None
        model_version = f"calibration_{regime or 'global'}_{train_date}_v1"
    
    # Save metadata
    meta = {
        "model_version": model_version,
        "regime": regime,
        "features": available_cols,
        "rows": int(len(df_clean)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "model": "logistic_regression",
        "brier_score": float(brier),
        "auc_roc": float(auc) if auc is not None else None,
        "trained_at": utc_now_iso_z(),
        "training_window": f"{train_start} → {train_end}" if train_start and train_end else None,
        "train_start": train_start,
        "train_end": train_end,
        "git_commit": git_commit,
        "eligibility": eligibility,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return meta


def load_model_metadata(meta_path: str) -> dict:
    """Load calibration model metadata."""
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def train_calibration_model_by_regime(
    df: pd.DataFrame,
    out_dir: str,
    min_samples: int = 200,
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """
    Train separate calibration models per regime.
    
    Args:
        df: Training data with 'regime' column
        out_dir: Output directory for models
        min_samples: Minimum samples required per regime
        feature_cols: Feature columns (defaults to FEATURE_COLS)
    
    Returns:
        Dict with training results per regime
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for calibration training")
    
    # G3.2: Validate no look-ahead features
    validate_no_future_features(df.columns.tolist())
    
    if "regime" not in df.columns:
        raise ValueError("DataFrame must have 'regime' column for per-regime training")
    
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    regimes = sorted(df["regime"].dropna().unique().tolist())
    
    for rg in regimes:
        sub = df[df["regime"] == rg]
        
        # G1.1: Check eligibility before training
        n_pos = int(sub["hit"].sum()) if "hit" in sub.columns else 0
        eligibility = check_eligibility_details(len(sub), n_pos, is_regime_model=True)
        
        if not eligibility["eligible"]:
            results[rg] = {
                "status": "skipped",
                "reason": eligibility["skip_reason"],
                "eligibility": eligibility,
            }
            continue
        
        model_path = os.path.join(out_dir, f"calibration_{rg}.pkl")
        meta_path = os.path.join(out_dir, f"calibration_{rg}.json")
        
        try:
            meta = train_calibration_model(
                df=sub,
                model_path=model_path,
                meta_path=meta_path,
                feature_cols=feature_cols,
                is_regime_model=True,
                regime=rg,
            )
            results[rg] = {"status": "trained", "meta": meta}
        except Exception as e:
            results[rg] = {"status": "failed", "error": str(e)}
    
    # Train global fallback model with stricter eligibility
    n_pos_global = int(df["hit"].sum()) if "hit" in df.columns else 0
    global_eligibility = check_eligibility_details(len(df), n_pos_global, is_regime_model=False)
    
    if not global_eligibility["eligible"]:
        results["global"] = {
            "status": "skipped",
            "reason": global_eligibility["skip_reason"],
            "eligibility": global_eligibility,
        }
    else:
        try:
            global_model_path = os.path.join(out_dir, "calibration_model.pkl")
            global_meta_path = os.path.join(out_dir, "calibration_meta.json")
            global_meta = train_calibration_model(
                df=df,
                model_path=global_model_path,
                meta_path=global_meta_path,
                feature_cols=feature_cols,
                is_regime_model=False,
                regime="global",
            )
            results["global"] = {"status": "trained", "meta": global_meta}
        except Exception as e:
            results["global"] = {"status": "failed", "error": str(e)}
    
    # Write summary
    summary_path = os.path.join(out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results
