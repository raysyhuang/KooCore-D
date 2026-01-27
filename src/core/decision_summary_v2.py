from dataclasses import dataclass


@dataclass
class DecisionSummary:
    decision: str
    confidence: float
    allow_new_positions: bool
    reason: str
    regime: str
    weekly_ready: bool
    pro30_ready: bool
    confluence_ready: bool
    conviction_ready: bool
    hybrid_fallback: bool

    def render(self) -> str:
        return f"""
==============================
KOOCORE DECISION SUMMARY
==============================
Decision: {self.decision}
Confidence: {self.confidence:.2f}

Market Regime:
- State: {self.regime}

Strategy Readiness:
- Weekly: {"READY" if self.weekly_ready else "NOT READY"}
- Pro30: {"READY" if self.pro30_ready else "NOT READY"}
- Confluence: {"READY" if self.confluence_ready else "NOT READY"}

Signal Integrity:
- Conviction Ranker: {"OK" if self.conviction_ready else "NO PICKS"}
- Hybrid Mode: {"FALLBACK" if self.hybrid_fallback else "NORMAL"}

Action:
- New Positions Allowed: {self.allow_new_positions}

Reason:
- {self.reason}
""".strip()


def build_decision_summary_v2(ctx) -> DecisionSummary:
    defensive = (
        ctx["regime"] == "chop"
        or not ctx["conviction_ready"]
        or ctx["hybrid_fallback"]
    )

    allow_new_positions = not defensive

    decision = "HOLD / DEFENSIVE MODE" if defensive else "ACTIVE / SELECTIVE"

    reason = (
        "Sideways regime with no conviction or multi-signal confirmation"
        if defensive
        else "Conviction and regime aligned"
    )

    confidence = 0.45 if defensive else 0.75

    return DecisionSummary(
        decision=decision,
        confidence=confidence,
        allow_new_positions=allow_new_positions,
        reason=reason,
        regime=ctx["regime"],
        weekly_ready=ctx["weekly_ready"],
        pro30_ready=ctx["pro30_ready"],
        confluence_ready=ctx["confluence_ready"],
        conviction_ready=ctx["conviction_ready"],
        hybrid_fallback=ctx["hybrid_fallback"],
    )
