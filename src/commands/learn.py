"""
Learn Command Handler

CLI command for model retraining and feature analysis.
Triggers weight updates from outcome database.
"""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

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
