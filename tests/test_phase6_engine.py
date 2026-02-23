"""
Tests for Phase 6 Recommendation Engine.

Covers Bayesian smoothing, source bonus computation,
confidence labels, filter/suppression logic, and end-to-end flow.
"""

from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.learning.phase6_engine import Phase6RecommendationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg():
    return {
        "min_observations": 20,
        "bayesian_prior_weight": 0.3,
        "bayesian_prior_hit_rate": 0.25,
        "recommendations": {
            "output_path": "outputs/phase6/recommendations.json",
            "backup_dir": "outputs/phase6/backups",
            "source_bonus_min": -1.5,
            "source_bonus_max": 3.0,
            "min_source_samples": 5,
        },
    }


@pytest.fixture
def engine(default_cfg):
    return Phase6RecommendationEngine(default_cfg)


@pytest.fixture
def sample_outcomes_df():
    """DataFrame mimicking outcome_db.get_training_data() output."""
    rows = []
    # weekly_top5: 6 trades, 4 hits (66.7%)
    for i in range(6):
        rows.append({
            "ticker": f"W{i}",
            "pick_date": f"2026-01-{10+i:02d}",
            "source": "weekly_top5",
            "entry_price": 100.0,
            "exit_reason": "target_hit" if i < 4 else "expired",
            "final_return_pct": 8.0 if i < 4 else -2.0,
            "max_return_pct": 10.0 if i < 4 else 1.0,
            "max_drawdown_pct": -2.0 if i < 4 else -5.0,
            "hit_7pct": i < 4,
            "days_held": 5,
        })
    # movers: 13 trades, 4 hits (30.8%)
    for i in range(13):
        rows.append({
            "ticker": f"M{i}",
            "pick_date": f"2026-01-{10+i:02d}",
            "source": "movers",
            "entry_price": 50.0,
            "exit_reason": "stopped" if i >= 4 else "target_hit",
            "final_return_pct": 7.0 if i < 4 else -11.0,
            "max_return_pct": 8.0 if i < 4 else 0.5,
            "max_drawdown_pct": -3.0 if i < 4 else -12.0,
            "hit_7pct": i < 4,
            "days_held": 6,
        })
    # hybrid_top3: 8 trades, 4 hits (50%)
    for i in range(8):
        rows.append({
            "ticker": f"H{i}",
            "pick_date": f"2026-01-{10+i:02d}",
            "source": "hybrid_top3",
            "entry_price": 75.0,
            "exit_reason": "target_hit" if i < 4 else "stopped",
            "final_return_pct": 9.0 if i < 4 else -4.0,
            "max_return_pct": 12.0 if i < 4 else 2.0,
            "max_drawdown_pct": -1.5 if i < 4 else -8.0,
            "hit_7pct": i < 4,
            "days_held": 5,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bayesian Smoothing
# ---------------------------------------------------------------------------

class TestBayesianSmoothing:
    def test_perfect_record_pulled_down(self, engine):
        """100% observed with small n should be pulled toward 25% prior."""
        smoothed = engine._bayesian_smooth(1.0, 5)
        # (5*1.0 + 6*0.25) / (5+6) = 6.5/11 ≈ 0.59
        assert 0.55 < smoothed < 0.65

    def test_zero_record_pulled_up(self, engine):
        """0% observed with small n should be pulled toward 25% prior."""
        smoothed = engine._bayesian_smooth(0.0, 5)
        # (5*0.0 + 6*0.25) / (5+6) = 1.5/11 ≈ 0.136
        assert 0.10 < smoothed < 0.20

    def test_large_sample_stays_close(self, engine):
        """With 100 observations, smoothing should have minimal effect."""
        smoothed = engine._bayesian_smooth(0.60, 100)
        # (100*0.6 + 6*0.25) / 106 = 61.5/106 ≈ 0.580
        assert abs(smoothed - 0.60) < 0.03

    def test_prior_dominates_at_zero_obs(self, engine):
        """With 0 observations, result equals the prior."""
        smoothed = engine._bayesian_smooth(0.0, 0)
        assert smoothed == pytest.approx(0.25)

    def test_k_value(self, engine):
        """k = prior_weight * min_observations = 0.3 * 20 = 6."""
        k = engine.prior_weight * engine.min_observations
        assert k == pytest.approx(6.0)

    def test_weekly_top5_scenario(self, engine):
        """weekly_top5: 66.7% with n=6 → smoothed ~46%."""
        smoothed = engine._bayesian_smooth(4 / 6, 6)
        # (6 * 0.667 + 6 * 0.25) / 12 = 5.5 / 12 ≈ 0.458
        assert 0.40 < smoothed < 0.52

    def test_movers_scenario(self, engine):
        """movers: 30.8% with n=13 → smoothed ~29%."""
        smoothed = engine._bayesian_smooth(4 / 13, 13)
        # (13 * 0.308 + 6 * 0.25) / 19 = 5.5 / 19 ≈ 0.289
        assert 0.25 < smoothed < 0.33


# ---------------------------------------------------------------------------
# Source Bonus Computation
# ---------------------------------------------------------------------------

class TestSourceBonus:
    def test_equal_rates_give_zero(self, engine):
        """When smoothed == overall, bonus should be 0."""
        bonus = engine._source_bonus_from_hit_rate(0.3, 0.3)
        assert bonus == pytest.approx(0.0)

    def test_better_source_positive_bonus(self, engine):
        """Source with 2x overall rate → log2(2) = +1.0."""
        bonus = engine._source_bonus_from_hit_rate(0.6, 0.3)
        assert bonus == pytest.approx(1.0)

    def test_worse_source_negative_bonus(self, engine):
        """Source with half overall rate → log2(0.5) = -1.0."""
        bonus = engine._source_bonus_from_hit_rate(0.15, 0.3)
        assert bonus == pytest.approx(-1.0)

    def test_clamp_upper(self, engine):
        """Extreme outperformance clamped to source_bonus_max."""
        bonus = engine._source_bonus_from_hit_rate(0.9, 0.05)
        assert bonus == engine.source_bonus_max

    def test_clamp_lower(self, engine):
        """Extreme underperformance clamped to source_bonus_min."""
        bonus = engine._source_bonus_from_hit_rate(0.01, 0.9)
        assert bonus == engine.source_bonus_min

    def test_zero_overall_returns_zero(self, engine):
        """Guard against division by zero."""
        bonus = engine._source_bonus_from_hit_rate(0.5, 0.0)
        assert bonus == 0.0

    def test_zero_smoothed_returns_zero(self, engine):
        bonus = engine._source_bonus_from_hit_rate(0.0, 0.5)
        assert bonus == 0.0


# ---------------------------------------------------------------------------
# Confidence Labels
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_high(self, engine):
        assert engine._confidence_label(30) == "HIGH"
        assert engine._confidence_label(100) == "HIGH"

    def test_medium(self, engine):
        assert engine._confidence_label(15) == "MEDIUM"
        assert engine._confidence_label(29) == "MEDIUM"

    def test_low(self, engine):
        assert engine._confidence_label(1) == "LOW"
        assert engine._confidence_label(14) == "LOW"


# ---------------------------------------------------------------------------
# Source Recommendations
# ---------------------------------------------------------------------------

class TestSourceRecommendations:
    def test_recommendations_computed(self, engine, sample_outcomes_df):
        recs = engine._compute_source_recommendations(
            sample_outcomes_df,
            {"weekly_top5": 0.0, "movers": -0.5, "hybrid_top3": 0.0},
        )
        assert len(recs) == 3  # three sources
        sources = {r["source"] for r in recs}
        assert sources == {"weekly_top5", "movers", "hybrid_top3"}

    def test_weekly_top5_gets_positive_bonus(self, engine, sample_outcomes_df):
        recs = engine._compute_source_recommendations(
            sample_outcomes_df,
            {"weekly_top5": 0.0, "movers": -0.5, "hybrid_top3": 0.0},
        )
        weekly = next(r for r in recs if r["source"] == "weekly_top5")
        assert weekly["suggested_bonus"] > 0

    def test_small_source_skipped(self, engine):
        """Sources with fewer than min_source_samples are excluded."""
        df = pd.DataFrame([
            {"ticker": "X", "source": "tiny", "hit_7pct": True, "final_return_pct": 5.0, "max_drawdown_pct": -1.0}
        ])
        recs = engine._compute_source_recommendations(df, {})
        assert len(recs) == 0

    def test_empty_df(self, engine):
        recs = engine._compute_source_recommendations(pd.DataFrame(), {})
        assert recs == []


# ---------------------------------------------------------------------------
# Filter Recommendations
# ---------------------------------------------------------------------------

class TestFilterRecommendations:
    def test_nan_entry_flagged(self, engine):
        df = pd.DataFrame([
            {"ticker": "GHOST1", "entry_price": float("nan"), "exit_reason": "expired", "max_drawdown_pct": -3, "days_held": 5},
            {"ticker": "GHOST2", "entry_price": 0.0, "exit_reason": "expired", "max_drawdown_pct": -2, "days_held": 5},
            {"ticker": "OK", "entry_price": 50.0, "exit_reason": "expired", "max_drawdown_pct": -1, "days_held": 5},
        ])
        filters = engine._compute_filter_recommendations(df)
        nan_filter = [f for f in filters if f["type"] == "nan_entry_price"]
        assert len(nan_filter) == 1
        assert nan_filter[0]["count"] == 2

    def test_stop_loss_breach_flagged(self, engine):
        df = pd.DataFrame([
            {"ticker": "A", "exit_reason": "stopped", "max_drawdown_pct": -12.0, "entry_price": 50.0, "days_held": 5},
            {"ticker": "B", "exit_reason": "stopped", "max_drawdown_pct": -9.0, "entry_price": 50.0, "days_held": 5},
            {"ticker": "C", "exit_reason": "target_hit", "max_drawdown_pct": -2.0, "entry_price": 50.0, "days_held": 5},
        ])
        filters = engine._compute_filter_recommendations(df)
        breach = [f for f in filters if f["type"] == "stop_loss_breach"]
        assert len(breach) == 1
        assert breach[0]["count"] == 2

    def test_no_issues_empty_list(self, engine):
        df = pd.DataFrame([
            {"ticker": "A", "entry_price": 50.0, "exit_reason": "target_hit", "max_drawdown_pct": -2.0, "days_held": 5},
        ])
        filters = engine._compute_filter_recommendations(df)
        assert len(filters) == 0


# ---------------------------------------------------------------------------
# Suppression Rules
# ---------------------------------------------------------------------------

class TestSuppressionRules:
    def test_low_hit_rate_suppressed(self, engine):
        """Source with 0% hit rate should be flagged for suppression."""
        rows = [{"ticker": f"X{i}", "source": "terrible", "hit_7pct": False, "final_return_pct": -5.0}
                for i in range(10)]
        df = pd.DataFrame(rows)
        rules = engine._compute_suppression_rules(df)
        assert len(rules) == 1
        assert rules[0]["source"] == "terrible"
        assert rules[0]["action"] == "SUPPRESS"

    def test_good_source_not_suppressed(self, engine):
        """Source with decent hit rate should not be suppressed."""
        rows = [{"ticker": f"X{i}", "source": "good", "hit_7pct": i < 5, "final_return_pct": 3.0}
                for i in range(10)]
        df = pd.DataFrame(rows)
        rules = engine._compute_suppression_rules(df)
        assert len(rules) == 0

    def test_small_source_skipped(self, engine):
        rows = [{"ticker": "X", "source": "tiny", "hit_7pct": False, "final_return_pct": -5.0}]
        df = pd.DataFrame(rows)
        rules = engine._compute_suppression_rules(df)
        assert len(rules) == 0


# ---------------------------------------------------------------------------
# End-to-End: generate_recommendations
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @patch("src.core.adaptive_scorer.get_adaptive_scorer")
    @patch("src.core.outcome_db.get_outcome_db")
    def test_generate_recommendations(self, mock_db, mock_scorer, engine, sample_outcomes_df):
        # Mock outcome db
        db_instance = MagicMock()
        db_instance.get_training_data.return_value = sample_outcomes_df
        mock_db.return_value = db_instance

        # Mock scorer
        scorer_instance = MagicMock()
        scorer_instance.weights.source_bonus = {"weekly_top5": 0.0, "movers": -0.5, "hybrid_top3": 0.0}
        scorer_instance.get_model_info.return_value = {"version": 1}
        mock_scorer.return_value = scorer_instance

        rec = engine.generate_recommendations({"regime": "bull"})

        assert rec["phase"] == 6
        assert rec["total_outcomes"] == len(sample_outcomes_df)
        assert len(rec["source_recommendations"]) == 3
        assert isinstance(rec["filter_recommendations"], list)
        assert isinstance(rec["suppression_rules"], list)
        assert rec["context"] == {"regime": "bull"}

    def test_save_and_load(self, engine, tmp_path):
        engine.output_path = tmp_path / "rec.json"
        engine.backup_dir = tmp_path / "backups"

        rec = {"phase": 6, "test": True}
        path = engine.save_recommendations(rec)

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["phase"] == 6

    def test_save_creates_backup(self, engine, tmp_path):
        engine.output_path = tmp_path / "rec.json"
        engine.backup_dir = tmp_path / "backups"

        # First save
        engine.save_recommendations({"version": 1})
        # Second save should backup first
        engine.save_recommendations({"version": 2})

        backups = list((tmp_path / "backups").glob("*.json"))
        assert len(backups) == 1

        loaded = json.loads(engine.output_path.read_text())
        assert loaded["version"] == 2
