"""Tests for the app-level acceptance layer."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.acceptance import classify_tier, run_acceptance, score_day_quality


@dataclass
class FakeSignal:
    ticker: str
    score: float
    components: dict


class TestScoreDayQuality:
    def test_high_quality_day(self):
        dq = score_day_quality(
            breadth_pct=0.55,
            regime="bull",
            eligible_count=25,
            universe_size=800,
            top_score=85,
        )
        assert dq.score >= 55
        assert dq.components["regime"] == 20.0  # bull

    def test_low_quality_bear_day(self):
        dq = score_day_quality(
            breadth_pct=0.25,
            regime="bear",
            eligible_count=2,
            universe_size=800,
            top_score=68,
        )
        assert dq.score < 35
        assert dq.components["regime"] == 0.0  # bear

    def test_choppy_moderate(self):
        dq = score_day_quality(
            breadth_pct=0.40,
            regime="choppy",
            eligible_count=10,
            universe_size=800,
            top_score=75,
        )
        assert 25 < dq.score < 65

    def test_diversity_penalty_sector(self):
        """High sector concentration should reduce day quality."""
        base = score_day_quality(
            breadth_pct=0.50, regime="bull", eligible_count=20,
            universe_size=800, top_score=80, sector_concentration=0.20,
        )
        concentrated = score_day_quality(
            breadth_pct=0.50, regime="bull", eligible_count=20,
            universe_size=800, top_score=80, sector_concentration=0.70,
        )
        assert concentrated.score < base.score
        assert concentrated.components["diversity"] < base.components["diversity"]

    def test_diversity_penalty_engine(self):
        """High engine concentration should reduce day quality."""
        diverse = score_day_quality(
            breadth_pct=0.50, regime="bull", eligible_count=20,
            universe_size=800, top_score=80, engine_concentration=0.60,
        )
        mono = score_day_quality(
            breadth_pct=0.50, regime="bull", eligible_count=20,
            universe_size=800, top_score=80, engine_concentration=0.95,
        )
        assert mono.score < diverse.score

    def test_eligible_count_drives_density(self):
        """Signal density should use eligible_count, not a truncated number."""
        low = score_day_quality(
            breadth_pct=0.50, regime="bull", eligible_count=2,
            universe_size=800, top_score=80,
        )
        high = score_day_quality(
            breadth_pct=0.50, regime="bull", eligible_count=30,
            universe_size=800, top_score=80,
        )
        assert high.components["signal_density"] > low.components["signal_density"]


class TestClassifyTier:
    def test_tier_a(self):
        components = {"a": 80, "b": 70, "c": 60, "d": 55}
        assert classify_tier(82, components) == "A"

    def test_tier_b(self):
        components = {"a": 80, "b": 70, "c": 60, "d": 30}
        assert classify_tier(75, components) == "B"

    def test_tier_c_low_score(self):
        components = {"a": 80, "b": 70, "c": 60}
        assert classify_tier(68, components) == "C"

    def test_tier_c_weak_components(self):
        components = {"a": 90, "b": 30, "c": 20, "d": 10}
        assert classify_tier(72, components) == "C"


class TestRunAcceptance:
    def _make_candidates(self, scores_and_components):
        result = []
        for i, (score, comps) in enumerate(scores_and_components):
            sig = FakeSignal(ticker=f"00000{i}.SZ", score=score, components=comps)
            result.append(("mean_reversion", sig))
        return result

    def test_full_mode_accepts_ab_rejects_c(self):
        candidates = self._make_candidates([
            (85, {"a": 90, "b": 80, "c": 70, "d": 60}),  # A
            (75, {"a": 80, "b": 60, "c": 55, "d": 50}),  # B
            (68, {"a": 90, "b": 30, "c": 20, "d": 10}),  # C
        ])
        result = run_acceptance(
            candidates=candidates,
            breadth_pct=0.55,
            regime="bull",
            universe_size=800,
            config={"dq_full_threshold": 55, "dq_selective_threshold": 35},
        )
        assert len(result.accepted) == 2  # A + B, not C
        assert not result.abstained
        assert result.mode == "full"
        assert result.eligible_count == 3

    def test_selective_mode_only_a(self):
        candidates = self._make_candidates([
            (85, {"a": 90, "b": 80, "c": 70, "d": 60}),  # A
            (75, {"a": 80, "b": 60, "c": 55, "d": 50}),  # B
        ])
        result = run_acceptance(
            candidates=candidates,
            breadth_pct=0.35,
            regime="choppy",
            universe_size=800,
            config={"dq_full_threshold": 55, "dq_selective_threshold": 35},
        )
        if result.mode == "selective":
            assert all(tier == "A" for _, _, tier in result.accepted)

    def test_abstention_on_terrible_day(self):
        candidates = self._make_candidates([
            (70, {"a": 60, "b": 50, "c": 40, "d": 30}),
        ])
        result = run_acceptance(
            candidates=candidates,
            breadth_pct=0.20,
            regime="bear",
            universe_size=800,
            config={"dq_full_threshold": 55, "dq_selective_threshold": 35},
        )
        assert result.abstained
        assert len(result.accepted) == 0
        assert result.mode == "abstain"

    def test_empty_candidates(self):
        result = run_acceptance(
            candidates=[],
            breadth_pct=0.55,
            regime="bull",
            universe_size=800,
        )
        assert len(result.accepted) == 0
        assert result.abstained
        assert result.eligible_count == 0

    def test_full_candidate_set_not_truncated(self):
        """Acceptance should see all 10 candidates, not a pre-capped list."""
        candidates = self._make_candidates([
            (85, {"a": 90, "b": 80, "c": 70, "d": 60}),  # A
            (83, {"a": 85, "b": 75, "c": 65, "d": 55}),  # A
            (80, {"a": 80, "b": 70, "c": 60, "d": 50}),  # A
            (78, {"a": 80, "b": 70, "c": 55, "d": 50}),  # B
            (76, {"a": 75, "b": 65, "c": 55, "d": 50}),  # B
            (74, {"a": 70, "b": 60, "c": 55, "d": 50}),  # B
            (72, {"a": 65, "b": 55, "c": 50, "d": 45}),  # C (one component < 50)
            (70, {"a": 60, "b": 50, "c": 40, "d": 30}),  # C
            (69, {"a": 55, "b": 45, "c": 40, "d": 30}),  # C
            (67, {"a": 50, "b": 40, "c": 35, "d": 25}),  # C
        ])
        result = run_acceptance(
            candidates=candidates,
            breadth_pct=0.55,
            regime="bull",
            universe_size=800,
            config={"dq_full_threshold": 55, "dq_selective_threshold": 35, "max_full": 5},
        )
        # Should see all 10, accept up to 5 from A+B, reject C
        assert result.eligible_count == 10
        assert len(result.accepted) == 5  # 3 A + 2 B, capped at 5
        assert all(tier in ("A", "B") for _, _, tier in result.accepted)
        # C-tier should be in rejected
        c_rejected = [r for r in result.rejected if r[2] == "C"]
        assert len(c_rejected) >= 3

    def test_info_map_drives_concentration(self):
        """When info_map is provided, sector concentration affects day quality."""
        candidates = self._make_candidates([
            (85, {"a": 90, "b": 80, "c": 70, "d": 60}),
            (83, {"a": 85, "b": 75, "c": 65, "d": 55}),
            (80, {"a": 80, "b": 70, "c": 60, "d": 50}),
        ])
        # All same sector
        info_same = {
            "000000.SZ": {"industry": "银行"},
            "000001.SZ": {"industry": "银行"},
            "000002.SZ": {"industry": "银行"},
        }
        result_same = run_acceptance(
            candidates=candidates,
            breadth_pct=0.55, regime="bull", universe_size=800,
            info_map=info_same,
        )
        # Different sectors
        info_diff = {
            "000000.SZ": {"industry": "银行"},
            "000001.SZ": {"industry": "医药"},
            "000002.SZ": {"industry": "电子"},
        }
        result_diff = run_acceptance(
            candidates=candidates,
            breadth_pct=0.55, regime="bull", universe_size=800,
            info_map=info_diff,
        )
        assert result_diff.day_quality.score >= result_same.day_quality.score
