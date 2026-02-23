"""
Learn Command Handler

CLI command for model retraining and feature analysis.
Triggers weight updates from outcome database.
"""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def cmd_learn(args) -> int:
    """
    Run model learning/retraining from outcome data.
    
    This command:
    1. Loads outcome data from the database
    2. Analyzes feature importance
    3. Computes optimal weights
    4. Updates model weights file
    5. Displays insights report
    """
    logger.info("=" * 60)
    logger.info("MODEL LEARNING - Updating Weights from Outcomes")
    logger.info("=" * 60)
    
    try:
        from src.core.outcome_db import get_outcome_db
        from src.core.adaptive_scorer import get_adaptive_scorer
        from src.analytics.feature_analyzer import FeatureAnalyzer, generate_insights_report
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        return 1
    
    # Get current model info
    scorer = get_adaptive_scorer()
    model_info = scorer.get_model_info()
    
    logger.info(f"\nCurrent Model: v{model_info['version']}")
    logger.info(f"Last Trained: {model_info.get('last_trained', 'Never')}")
    logger.info(f"Observations: {model_info.get('observations', 0)}")
    logger.info(f"Overall Hit Rate: {model_info.get('overall_hit_rate', 0)*100:.1f}%")
    
    # Get outcome database stats
    db = get_outcome_db()
    stats = db.get_outcome_stats()
    
    logger.info(f"\nOutcome Database:")
    logger.info(f"  Total Picks: {stats.get('total_picks', 0)}")
    logger.info(f"  Total Outcomes: {stats.get('total_outcomes', 0)}")
    logger.info(f"  Pending: {stats.get('pending_outcomes', 0)}")
    
    if stats.get("total_outcomes", 0) == 0:
        logger.warning("\n⚠ No outcomes recorded yet. Run scans and let positions close to build data.")
        return 0
    
    # Run feature analysis
    logger.info("\n[1/3] Analyzing Feature Importance...")
    analyzer = FeatureAnalyzer()
    n_loaded = analyzer.load_outcomes()
    
    if n_loaded == 0:
        logger.error("No outcome data loaded for analysis")
        return 1
    
    # Display insights report
    if getattr(args, "report", True):
        logger.info("\n[2/3] Generating Insights Report...")
        report = analyzer.generate_report()
        print("\n" + report)
    
    # Retrain model
    logger.info("\n[3/3] Retraining Model Weights...")
    
    force = getattr(args, "force", False)
    result = scorer.retrain(force=force)
    
    if result.get("status") == "skipped":
        logger.warning(f"\n⚠ Training skipped: {result.get('reason')}")
        logger.info("Use --force to train anyway with limited data")
        return 0
    
    if result.get("status") == "error":
        logger.error(f"\n✗ Training failed: {result.get('reason')}")
        return 1
    
    if result.get("status") == "success":
        logger.info(f"\n✓ Model updated: v{result.get('old_version')} → v{result.get('new_version')}")
        logger.info(f"  Observations: {result.get('observations')}")
        logger.info(f"  Hit Rate: {result.get('overall_hit_rate', 0)*100:.1f}%")
        
        # Display new weights
        weights = result.get("weights", {})
        logger.info("\n  Updated Weights:")
        logger.info(f"    overlap_bonus: {weights.get('overlap_bonus', 0)}")
        
        source_bonus = weights.get("source_bonus", {})
        if source_bonus:
            logger.info("    source_bonus:")
            for src, bonus in source_bonus.items():
                logger.info(f"      {src}: {bonus:+.2f}")
        
        sector_bonus = weights.get("sector_bonus", {})
        if sector_bonus:
            logger.info("    sector_bonus:")
            for sector, bonus in list(sector_bonus.items())[:5]:
                logger.info(f"      {sector}: {bonus:+.2f}")
        
        logger.info(f"    high_rsi_penalty: {weights.get('high_rsi_penalty', 0)}")
        logger.info(f"    volume_spike_bonus: {weights.get('volume_spike_bonus', 0)}")
    
    # Suggestions
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    # Get source stats for recommendations
    source_stats = stats.get("by_source", {})
    
    if source_stats:
        best_source = max(source_stats.items(), key=lambda x: x[1].get("hit_7pct_rate", 0))
        worst_source = min(source_stats.items(), key=lambda x: x[1].get("hit_7pct_rate", 0))
        
        if best_source[1].get("hit_7pct_rate", 0) > 0.3:
            logger.info(f"  ✓ {best_source[0]} performing well ({best_source[1]['hit_7pct_rate']*100:.0f}% hit rate)")
        
        if worst_source[1].get("hit_7pct_rate", 0) < 0.15:
            logger.info(f"  ⚠ Consider de-weighting {worst_source[0]} ({worst_source[1]['hit_7pct_rate']*100:.0f}% hit rate)")
    
    # Overlap recommendation
    overlap_analysis = analyzer.get_overlap_analysis()
    if overlap_analysis.get("by_overlap_count", {}):
        overlap_3 = overlap_analysis["by_overlap_count"].get(3, {})
        if overlap_3.get("hit_7pct_rate", 0) > 0.5:
            logger.info("  ✓ ALL-THREE overlaps are high conviction - prioritize these")
    
    # Data recommendation
    if stats.get("total_outcomes", 0) < 50:
        remaining = 50 - stats.get("total_outcomes", 0)
        logger.info(f"  ℹ Need ~{remaining} more outcomes for reliable model training")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ LEARNING COMPLETE")
    logger.info("=" * 60)
    
    return 0


def cmd_learn_status(args) -> int:
    """Display model learning status without retraining."""
    try:
        from src.core.outcome_db import get_outcome_db
        from src.core.adaptive_scorer import get_adaptive_scorer
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        return 1
    
    scorer = get_adaptive_scorer()
    model_info = scorer.get_model_info()
    
    print("\n" + "=" * 50)
    print("MODEL STATUS")
    print("=" * 50)
    print(f"  Version:       v{model_info['version']}")
    print(f"  Last Trained:  {model_info.get('last_trained', 'Never')}")
    print(f"  Observations:  {model_info.get('observations', 0)}")
    print(f"  Hit Rate:      {model_info.get('overall_hit_rate', 0)*100:.1f}%")
    print(f"  Needs Training: {'Yes' if model_info.get('needs_training') else 'No'}")
    
    db = get_outcome_db()
    stats = db.get_outcome_stats()
    
    print("\n" + "-" * 50)
    print("OUTCOME DATABASE")
    print("-" * 50)
    print(f"  Total Picks:   {stats.get('total_picks', 0)}")
    print(f"  Total Outcomes: {stats.get('total_outcomes', 0)}")
    print(f"  Pending:       {stats.get('pending_outcomes', 0)}")
    
    if stats.get("by_source"):
        print("\n  By Source:")
        for source, src_stats in stats["by_source"].items():
            hit_rate = src_stats.get("hit_7pct_rate", 0) * 100
            count = src_stats.get("count", 0)
            print(f"    {source}: {hit_rate:.0f}% hit rate (n={count})")
    
    print("=" * 50)
    
    return 0


def cmd_learn_export(args) -> int:
    """Export outcome data to CSV for external analysis."""
    try:
        from src.core.outcome_db import get_outcome_db
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        return 1
    
    db = get_outcome_db()
    df = db.get_training_data()
    
    if df.empty:
        print("No outcome data to export")
        return 0
    
    output_path = getattr(args, "output", None) or "outputs/outcomes_export.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(df)} outcomes to {output_path}")
    
    return 0


def cmd_learn_memory(args) -> int:
    """
    Sync outcome data to ChromaDB memory for similarity-based learning.
    
    This enables the debate system to retrieve similar past situations
    and learn from their outcomes.
    """
    logger.info("=" * 60)
    logger.info("MEMORY SYNC - Building Situation Memory from Outcomes")
    logger.info("=" * 60)
    
    try:
        from src.core.outcome_db import get_outcome_db
        from src.core.memory import get_trading_memory, build_situation_from_packet
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        logger.info("Install chromadb with: pip install chromadb")
        return 1
    
    memory = get_trading_memory()
    if not memory.is_available:
        logger.error("Memory system not available. Check OpenAI API key and chromadb installation.")
        return 1
    
    # Get current memory stats
    mem_stats = memory.get_stats()
    logger.info(f"\nCurrent Memory: {mem_stats.get('count', 0)} situations stored")
    
    # Load outcomes
    db = get_outcome_db()
    df = db.get_training_data()
    
    if df.empty:
        logger.warning("No outcome data available to sync")
        return 0
    
    logger.info(f"Found {len(df)} outcomes to process")
    
    # Process each outcome
    added = 0
    skipped = 0
    
    for _, row in df.iterrows():
        ticker = row.get("ticker", "UNK")
        entry_date = row.get("entry_date", "")
        
        # Build situation
        situation = {
            "ticker": ticker,
            "entry_date": entry_date,
            "entry_price": row.get("entry_price", 0),
            "sector": row.get("sector", ""),
            "source": row.get("source", "unknown"),
            "composite_score": row.get("composite_score", 0),
            "technical": {
                "rsi14": row.get("rsi14"),
                "volume_ratio_3d_to_20d": row.get("volume_ratio"),
                "within_5pct_52w_high": row.get("near_52w_high"),
            },
            "catalyst": {
                "title": row.get("catalyst_type", ""),
            },
            "headlines": [],
        }
        
        # Build outcome
        outcome = {
            "max_gain_7d": row.get("max_gain_7d", 0),
            "max_drawdown_7d": row.get("max_drawdown_7d", 0),
            "close_7d_return": row.get("close_7d_return", 0),
            "hit_10pct": row.get("hit_10pct", False),
            "hit_7pct": row.get("hit_7pct", False),
            "hit_5pct": row.get("hit_5pct", False),
            "days_to_hit_10pct": row.get("days_to_hit"),
            "outcome_date": row.get("outcome_date", ""),
        }
        
        # Build lesson
        hit = outcome.get("hit_7pct", False)
        gain = outcome.get("max_gain_7d", 0)
        dd = outcome.get("max_drawdown_7d", 0)
        
        if hit and gain > 15:
            lesson = f"Strong winner: {ticker} gained {gain:.1f}%. Technical setup with {situation['source']} signal worked well."
        elif hit:
            lesson = f"Modest winner: {ticker} gained {gain:.1f}%. Trade worked but required patience."
        elif gain > 0:
            lesson = f"Small gain: {ticker} gained {gain:.1f}% but didn't hit threshold. Consider tighter entries."
        elif dd < -10:
            lesson = f"Significant loss: {ticker} dropped {dd:.1f}%. Review entry timing and risk factors."
        else:
            lesson = f"Minor loss: {ticker} returned {gain:.1f}%. Trade thesis didn't play out."
        
        try:
            memory.add_situation(situation, outcome, lesson)
            added += 1
        except Exception as e:
            logger.warning(f"Failed to add {ticker}: {e}")
            skipped += 1
        
        # Progress
        if added % 50 == 0 and added > 0:
            logger.info(f"  Processed {added} situations...")
    
    # Final stats
    final_stats = memory.get_stats()
    
    logger.info("\n" + "=" * 60)
    logger.info("MEMORY SYNC COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Added: {added}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Total in memory: {final_stats.get('count', 0)}")
    if final_stats.get("hit_rate_10pct"):
        logger.info(f"  Memory hit rate: {final_stats.get('hit_rate_10pct', 0)*100:.1f}%")
    
    logger.info("\n✓ Memory system ready for debate analysis")
    
    return 0


# =============================================================================
# PHASE-5 LEARNING COMMANDS
# =============================================================================

def cmd_learn_resolve(args) -> int:
    """
    Resolve outcomes for Phase-5 learning rows.
    
    Usage:
        python main.py learn resolve [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run]
    """
    logger.info("=" * 60)
    logger.info("PHASE-5 OUTCOME RESOLUTION")
    logger.info("=" * 60)
    
    try:
        from src.learning.phase5_resolver import get_phase5_resolver
        from src.learning.phase5_store import get_phase5_store
    except ImportError as e:
        logger.error(f"Could not import Phase-5 modules: {e}")
        return 1
    
    # Get parameters
    start_date = getattr(args, "start", None)
    end_date = getattr(args, "end", None)
    dry_run = getattr(args, "dry_run", False)
    
    # Show current stats
    store = get_phase5_store()
    stats = store.get_stats()
    
    logger.info(f"\nCurrent Phase-5 Storage:")
    logger.info(f"  Total rows: {stats.get('total_rows', 0)}")
    logger.info(f"  Total outcomes: {stats.get('total_outcomes', 0)}")
    logger.info(f"  Resolution rate: {stats.get('resolution_rate', 0):.1%}")
    
    # Show resolvable dates
    resolver = get_phase5_resolver()
    resolvable_dates = resolver.get_resolvable_dates()
    
    if not resolvable_dates:
        logger.info("\n⚠ No rows ready for resolution (need 7+ trading days)")
        return 0
    
    logger.info(f"\nResolvable dates: {', '.join(resolvable_dates[:10])}")
    if len(resolvable_dates) > 10:
        logger.info(f"  ... and {len(resolvable_dates) - 10} more")
    
    # Resolve
    logger.info(f"\nResolving outcomes...")
    if dry_run:
        logger.info("  [DRY RUN - no writes]")
    
    result = resolver.resolve_outcomes(
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("RESOLUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Resolved: {result['resolved']}")
    logger.info(f"  Skipped (already resolved): {result['skipped']}")
    logger.info(f"  Not ready: {result['not_ready']}")
    logger.info(f"  Errors: {result['errors']}")
    
    return 0


def cmd_learn_merge(args) -> int:
    """
    Merge Phase-5 rows and outcomes into training dataset.
    
    Usage:
        python main.py learn merge
    """
    logger.info("=" * 60)
    logger.info("PHASE-5 DATA MERGE")
    logger.info("=" * 60)
    
    try:
        from src.learning.phase5_store import get_phase5_store
    except ImportError as e:
        logger.error(f"Could not import Phase-5 modules: {e}")
        return 1
    
    store = get_phase5_store()
    
    # Show pre-merge stats
    stats = store.get_stats()
    logger.info(f"\nPre-merge stats:")
    logger.info(f"  Row files: {stats.get('rows_files', 0)}")
    logger.info(f"  Total rows: {stats.get('total_rows', 0)}")
    logger.info(f"  Outcome files: {stats.get('outcome_files', 0)}")
    logger.info(f"  Total outcomes: {stats.get('total_outcomes', 0)}")
    
    # Merge
    logger.info(f"\nMerging to parquet...")
    merged_count = store.merge_rows_and_outcomes()
    
    logger.info("\n" + "=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Merged rows: {merged_count}")
    logger.info(f"  Output: {store.get_merged_path()}")
    
    return 0


def cmd_learn_analyze(args) -> int:
    """
    Analyze Phase-5 learning data and generate scorecard.
    
    Usage:
        python main.py learn analyze [--save]
    """
    logger.info("=" * 60)
    logger.info("PHASE-5 LEARNING ANALYSIS")
    logger.info("=" * 60)
    
    try:
        from src.learning.phase5_analyzer import get_phase5_analyzer
    except ImportError as e:
        logger.error(f"Could not import Phase-5 modules: {e}")
        return 1
    
    analyzer = get_phase5_analyzer()
    
    # Check data availability
    df = analyzer.df
    if df.empty:
        logger.warning("\n⚠ No Phase-5 data available")
        logger.info("Run 'python main.py learn merge' first to create merged dataset")
        return 0
    
    # Print report
    analyzer.print_report()
    
    # Optionally save scorecard
    if getattr(args, "save", True):
        path = analyzer.save_scorecard()
        logger.info(f"\n✓ Scorecard saved: {path}")
    
    # Show recommendations (Phase-6 preview)
    recommendations = analyzer.get_weight_recommendations()
    
    if recommendations.get("source_weights"):
        logger.info("\n📋 WEIGHT RECOMMENDATIONS (Phase-6 Preview):")
        for source, rec in recommendations["source_weights"].items():
            current = "?"
            suggested = rec.get("suggested_weight", 1.0)
            logger.info(f"  {source}: current={current} → suggested={suggested:.2f} "
                       f"(hit_rate={rec.get('observed_hit_rate', 0):.1%}, n={rec.get('sample_size', 0)})")
    
    if recommendations.get("suppression_rules"):
        logger.info("\n⚠ SUPPRESSION RULES SUGGESTED:")
        for rule in recommendations["suppression_rules"]:
            logger.info(f"  • {rule['condition']}: {rule['reason']}")
    
    return 0


def cmd_learn_stats(args) -> int:
    """
    Display Phase-5 storage statistics.
    
    Usage:
        python main.py learn stats
    """
    logger.info("=" * 60)
    logger.info("PHASE-5 STORAGE STATISTICS")
    logger.info("=" * 60)
    
    try:
        from src.learning.phase5_store import get_phase5_store
    except ImportError as e:
        logger.error(f"Could not import Phase-5 modules: {e}")
        return 1
    
    store = get_phase5_store()
    stats = store.get_stats()
    
    print(f"\n📊 Phase-5 Storage")
    print(f"   Row files: {stats.get('rows_files', 0)}")
    print(f"   Total rows: {stats.get('total_rows', 0)}")
    print(f"   Outcome files: {stats.get('outcome_files', 0)}")
    print(f"   Total outcomes: {stats.get('total_outcomes', 0)}")
    print(f"   Resolution rate: {stats.get('resolution_rate', 0):.1%}")
    print(f"   Merged rows: {stats.get('merged_rows', 0)}")
    print(f"\n   Base path: {store.base_path}")

    return 0


# =============================================================================
# PHASE-6 LEARNING COMMANDS
# =============================================================================

def cmd_learn_recommend(args) -> int:
    """
    Generate Phase-6 weight-update recommendations for human review.

    Usage:
        python main.py learn recommend [--config CONFIG]
    """
    logger.info("=" * 60)
    logger.info("PHASE-6 RECOMMENDATION ENGINE")
    logger.info("=" * 60)

    try:
        from src.learning.phase6_engine import Phase6RecommendationEngine
    except ImportError as e:
        logger.error(f"Could not import Phase-6 modules: {e}")
        return 1

    # Load config
    config_path = getattr(args, "config", None)
    phase6_cfg = _load_phase6_config(config_path)

    if not phase6_cfg:
        logger.error("Phase 6 config not found. Provide --config or add phase6: to default.yaml")
        return 1

    engine = Phase6RecommendationEngine(phase6_cfg)

    # Generate recommendations
    logger.info("\nGenerating recommendations...")
    rec = engine.generate_recommendations()

    # Display summary
    logger.info(f"\n  Total outcomes: {rec['total_outcomes']}")
    logger.info(f"  Overall hit rate: {rec['overall_hit_rate']:.1%}")

    if rec["source_recommendations"]:
        logger.info("\n  Source Recommendations:")
        for sr in rec["source_recommendations"]:
            delta_str = f"{sr['delta']:+.3f}" if sr['delta'] != 0 else "  0.000"
            logger.info(
                f"    {sr['source']:15s}: bonus {sr['current_bonus']:+.2f} -> "
                f"{sr['suggested_bonus']:+.3f} (delta {delta_str}) "
                f"[{sr['confidence']}] n={sr['n']}"
            )

    if rec["suppression_rules"]:
        logger.info("\n  Suppression Rules:")
        for rule in rec["suppression_rules"]:
            logger.info(f"    {rule['action']} {rule['source']}: {rule['reason']}")

    if rec["filter_recommendations"]:
        logger.info("\n  Filter Issues:")
        for f in rec["filter_recommendations"]:
            logger.info(f"    [{f['severity']}] {f['type']}: {f['recommendation']}")

    # Save
    path = engine.save_recommendations(rec)

    logger.info("\n" + "=" * 60)
    logger.info(f"Recommendations saved: {path}")
    logger.info("Review the JSON, then run: python main.py learn apply --confirm")
    logger.info("=" * 60)

    return 0


def cmd_learn_apply(args) -> int:
    """
    Apply Phase-6 recommendations to model weights.

    Usage:
        python main.py learn apply --confirm [--config CONFIG] [--dry-run]
    """
    import json as _json

    logger.info("=" * 60)
    logger.info("PHASE-6 APPLY RECOMMENDATIONS")
    logger.info("=" * 60)

    confirm = getattr(args, "confirm", False)
    dry_run = getattr(args, "dry_run", False)

    if not confirm and not dry_run:
        logger.error("Safety gate: must pass --confirm to apply changes (or --dry-run to preview)")
        return 1

    # Load config to find recommendation path
    config_path = getattr(args, "config", None)
    phase6_cfg = _load_phase6_config(config_path)
    rec_cfg = (phase6_cfg or {}).get("recommendations", {})
    rec_path = Path(rec_cfg.get("output_path", "outputs/phase6/recommendations.json"))

    if not rec_path.exists():
        logger.error(f"No recommendations found at {rec_path}")
        logger.info("Run 'python main.py learn recommend' first")
        return 1

    with open(rec_path) as f:
        rec = _json.load(f)

    logger.info(f"\nLoaded recommendations from {rec_path}")
    logger.info(f"  Generated: {rec.get('generated_at', 'unknown')}")
    logger.info(f"  Outcomes:  {rec.get('total_outcomes', 0)}")

    # Load current weights
    from src.core.adaptive_scorer import get_adaptive_scorer, WEIGHTS_PATH

    scorer = get_adaptive_scorer()
    current = dict(scorer.weights.source_bonus)

    # Show diff
    source_recs = rec.get("source_recommendations", [])
    if not source_recs:
        logger.info("\nNo source weight changes recommended.")
        return 0

    logger.info("\n  Proposed Weight Changes:")
    changes = {}
    for sr in source_recs:
        source = sr["source"]
        old_val = current.get(source, 0.0)
        new_val = sr["suggested_bonus"]
        changes[source] = new_val
        marker = " *" if abs(new_val - old_val) > 0.1 else ""
        logger.info(
            f"    {source:15s}: {old_val:+.2f} -> {new_val:+.3f} "
            f"(delta {sr['delta']:+.3f}, {sr['confidence']}){marker}"
        )

    suppression = rec.get("suppression_rules", [])
    if suppression:
        logger.info("\n  Suppression Warnings:")
        for rule in suppression:
            logger.info(f"    {rule['source']}: {rule['reason']}")

    if dry_run:
        logger.info("\n  [DRY RUN - no changes applied]")
        return 0

    # Backup current weights
    weights_path = Path(WEIGHTS_PATH)
    backup_dir = Path(rec_cfg.get("backup_dir", "outputs/phase6/backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)

    if weights_path.exists():
        import shutil
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup = backup_dir / f"model_weights_{ts}.json"
        shutil.copy2(weights_path, backup)
        logger.info(f"\n  Backed up weights to {backup}")

    # Apply changes
    for source, new_bonus in changes.items():
        scorer.weights.source_bonus[source] = new_bonus

    scorer.weights.version += 1
    scorer.weights.last_trained = datetime.utcnow().strftime("%Y-%m-%d")
    scorer._save_weights()

    logger.info(f"\n  Applied {len(changes)} source bonus updates")
    logger.info(f"  New model version: v{scorer.weights.version}")

    logger.info("\n" + "=" * 60)
    logger.info("PHASE-6 APPLY COMPLETE")
    logger.info("=" * 60)

    return 0


def _load_phase6_config(config_path: Optional[str] = None) -> Optional[dict]:
    """Load phase6 config from the given YAML path or fallback to default."""
    import yaml

    paths_to_try = []
    if config_path:
        paths_to_try.append(Path(config_path))
    paths_to_try.append(Path("config/phase6.yaml"))
    paths_to_try.append(Path("config/default.yaml"))

    for path in paths_to_try:
        if path.exists():
            try:
                with open(path) as f:
                    cfg = yaml.safe_load(f)
                phase6 = cfg.get("phase6")
                if phase6:
                    return phase6
            except Exception:
                continue

    return None
