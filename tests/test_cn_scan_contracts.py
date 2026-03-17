from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.core.universe import get_top_n_cn_by_market_cap
from src.pipelines.scanner import _sort_signal_candidates


def test_market_cap_universe_requires_tushare_ranking(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    import src.core.cn_data as cn_data

    monkeypatch.setattr(
        cn_data,
        "_ak_fetch_basic_info",
        lambda: pd.DataFrame({"ticker": ["000001.SZ", "000002.SZ"]}),
    )

    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN"):
        get_top_n_cn_by_market_cap()


def test_sort_signal_candidates_breaks_ties_by_adv_then_market_cap():
    candidates = [
        ("sniper", SimpleNamespace(ticker="AAA", score=80.0)),
        ("mean_reversion", SimpleNamespace(ticker="BBB", score=80.0)),
        ("sniper", SimpleNamespace(ticker="CCC", score=80.0)),
        ("mean_reversion", SimpleNamespace(ticker="DDD", score=75.0)),
    ]
    data_map = {
        "AAA": pd.DataFrame({"Close": [10] * 20, "Volume": [100] * 20}),
        "BBB": pd.DataFrame({"Close": [10] * 20, "Volume": [200] * 20}),
        "CCC": pd.DataFrame({"Close": [10] * 20, "Volume": [200] * 20}),
        "DDD": pd.DataFrame({"Close": [10] * 20, "Volume": [500] * 20}),
    }
    info_map = {
        "AAA": {"market_cap": 300},
        "BBB": {"market_cap": 100},
        "CCC": {"market_cap": 500},
        "DDD": {"market_cap": 1000},
    }

    ranked = _sort_signal_candidates(candidates, data_map, info_map)

    assert [sig.ticker for _, sig in ranked] == ["CCC", "BBB", "AAA", "DDD"]
