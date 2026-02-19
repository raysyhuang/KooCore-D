from src.api.engine_endpoint import _map_hybrid_to_payload


def test_map_hybrid_payload_populates_risk_fields():
    hybrid = {
        "hybrid_top3": [
            {
                "ticker": "AAPL",
                "composite_score": 4.5,
                "sources": ["weekly"],
                "current_price": 100.0,
                "target": {"target_price_for_10pct": 110.0},
                "scores": {},
                "rank": 1,
            }
        ],
        "weighted_picks": [
            {
                "ticker": "MSFT",
                "hybrid_score": 3.8,
                "sources": ["pro30"],
                "current_price": 200.0,
            }
        ],
        "summary": {
            "weekly_top5_count": 5,
            "pro30_candidates_count": 30,
            "movers_count": 0,
        },
    }

    payload = _map_hybrid_to_payload(hybrid, "2026-02-18")

    assert payload["picks"], "Expected non-empty mapped picks"
    for pick in payload["picks"]:
        assert pick["stop_loss"] is not None and pick["stop_loss"] > 0
        assert pick["target_price"] is not None and pick["target_price"] > 0
