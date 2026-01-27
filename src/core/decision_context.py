def build_decision_context(
    regime: str,
    weekly_count: int,
    pro30_count: int,
    confluence_count: int,
    conviction_count: int,
    hybrid_sources: list,
):
    return {
        "regime": regime,
        "weekly_ready": weekly_count > 0,
        "pro30_ready": pro30_count > 0,
        "confluence_ready": confluence_count > 0,
        "conviction_ready": conviction_count > 0,
        "hybrid_fallback": len(set(hybrid_sources)) == 1,  # Movers-only
    }
