from src.api.engine_endpoint import _to_legacy_picks_payload


def test_legacy_picks_payload_groups_by_strategy_and_dedupes():
    payload = {
        "run_date": "2026-02-18",
        "picks": [
            {"ticker": "AAPL", "strategy": "hybrid_weekly", "metadata": {"sources": ["weekly"]}},
            {"ticker": "MSFT", "strategy": "momentum", "metadata": {"sources": ["pro30"]}},
            {"ticker": "TSLA", "strategy": "breakout", "metadata": {"sources": ["movers"]}},
            {"ticker": "MSFT", "strategy": "momentum", "metadata": {"sources": ["pro30"]}},
        ],
    }

    legacy = _to_legacy_picks_payload(payload)
    day = legacy["picks_data"]["2026-02-18"]

    assert day["weekly"] == ["AAPL"]
    assert day["pro30"] == ["MSFT"]
    assert day["movers"] == ["TSLA"]
