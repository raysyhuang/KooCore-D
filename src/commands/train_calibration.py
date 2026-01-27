# src/commands/train_calibration.py
"""
CLI command to train the probability calibration model.

Usage:
    # Single model
    python main.py train-calibration --snapshots data/snapshots/*.parquet \
                                     --outcomes data/outcomes/*.parquet \
                                     --model-path models/calibration_model.pkl \
                                     --meta-path models/calibration_meta.json
    
    # Per-regime models
    python main.py train-calibration --snapshots data/snapshots/*.parquet \
                                     --outcomes data/outcomes/*.parquet \
                                     --by-regime \
                                     --out-dir models/
"""
from __future__ import annotations
import argparse
import glob
import logging
import os

logger = logging.getLogger(__name__)


def cmd_train_calibration(args) -> int:
    """Train a probability calibration model from snapshots and outcomes."""
    import pandas as pd
    from src.calibration.dataset import build_dataset
    from src.calibration.train import (
        train_calibration_model,
        train_calibration_model_by_regime,
        SKLEARN_AVAILABLE,
    )
    
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for calibration training. Install with: pip install scikit-learn")
        return 1
    
    # Load snapshots
    snapshot_files = []
    for pattern in args.snapshots:
        snapshot_files.extend(glob.glob(pattern))
    
    if not snapshot_files:
        logger.error(f"No snapshot files found matching: {args.snapshots}")
        return 1
    
    logger.info(f"Loading {len(snapshot_files)} snapshot files...")
    snapshots = pd.concat([pd.read_parquet(f) for f in snapshot_files], ignore_index=True)
    logger.info(f"  Total snapshot rows: {len(snapshots)}")
    
    # Load outcomes
    outcome_files = []
    for pattern in args.outcomes:
        outcome_files.extend(glob.glob(pattern))
    
    if not outcome_files:
        logger.error(f"No outcome files found matching: {args.outcomes}")
        return 1
    
    logger.info(f"Loading {len(outcome_files)} outcome files...")
    outcomes = pd.concat([pd.read_parquet(f) for f in outcome_files], ignore_index=True)
    logger.info(f"  Total outcome rows: {len(outcomes)}")
    
    # Build dataset
    logger.info("Building calibration dataset...")
    df = build_dataset(snapshots, outcomes)
    logger.info(f"  Matched rows: {len(df)}")
    
    if len(df) < 50:
        logger.error(f"Insufficient matched data ({len(df)} rows). Need at least 50.")
        return 1
    
    # Check for by-regime training
    by_regime = getattr(args, 'by_regime', False)
    
    if by_regime:
        # Train per-regime models
        out_dir = getattr(args, 'out_dir', 'models')
        
        if "regime" not in df.columns:
            logger.warning("No 'regime' column in data, training global model only")
            by_regime = False
        else:
            logger.info(f"Training per-regime models to {out_dir}...")
            regimes = df["regime"].dropna().unique().tolist()
            logger.info(f"  Found regimes: {regimes}")
            
            try:
                results = train_calibration_model_by_regime(
                    df=df,
                    out_dir=out_dir,
                    min_samples=int(getattr(args, 'min_samples', 200)),
                )
                
                for rg, result in results.items():
                    if result["status"] == "trained":
                        meta = result["meta"]
                        logger.info(f"  {rg}: trained ({meta['rows']} rows, positive_rate={meta['positive_rate']:.2%})")
                    else:
                        logger.info(f"  {rg}: {result['status']} - {result.get('reason', result.get('error', ''))}")
                
                logger.info(f"✓ Training summary saved to {out_dir}/training_summary.json")
                return 0
            except Exception as e:
                logger.error(f"Per-regime training failed: {e}")
                return 1
    
    # Train single global model
    if not by_regime:
        logger.info("Training global calibration model...")
        model_path = getattr(args, 'model_path', None) or os.path.join(getattr(args, 'out_dir', 'models'), 'calibration_model.pkl')
        meta_path = getattr(args, 'meta_path', None) or os.path.join(getattr(args, 'out_dir', 'models'), 'calibration_meta.json')
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True) if os.path.dirname(model_path) else None
        
        try:
            meta = train_calibration_model(
                df=df,
                model_path=model_path,
                meta_path=meta_path,
            )
            logger.info(f"  Model saved to: {model_path}")
            logger.info(f"  Metadata saved to: {meta_path}")
            logger.info(f"  Training rows: {meta['rows']}")
            logger.info(f"  Positive rate: {meta['positive_rate']:.2%}")
            if meta.get('auc_roc'):
                logger.info(f"  AUC-ROC: {meta['auc_roc']:.3f}")
            logger.info(f"  Brier score: {meta['brier_score']:.4f}")
            
            return 0
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1
    
    return 0


def add_train_calibration_parser(subparsers) -> None:
    """Add train-calibration subcommand to argument parser."""
    parser = subparsers.add_parser(
        "train-calibration",
        help="Train probability calibration model from historical data",
    )
    parser.add_argument(
        "--snapshots",
        nargs="+",
        required=True,
        help="Glob patterns for snapshot parquet files",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        required=True,
        help="Glob patterns for outcome parquet files",
    )
    parser.add_argument(
        "--model-path",
        help="Output path for trained model (joblib) - for single model mode",
    )
    parser.add_argument(
        "--meta-path",
        help="Output path for model metadata (JSON) - for single model mode",
    )
    parser.add_argument(
        "--by-regime",
        action="store_true",
        help="Train separate models per regime (bull/chop/stress)",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Output directory for per-regime models (default: models)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum samples required per regime (default: 200)",
    )
    parser.set_defaults(func=cmd_train_calibration)
