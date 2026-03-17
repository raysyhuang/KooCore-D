from __future__ import annotations

import pandas as pd

from src.signals.mean_reversion import score_mean_reversion


def _build_df() -> pd.DataFrame:
    closes = [95.0] * 55 + [100.0, 100.0, 100.0, 100.0, 100.0]
    volumes = [2_000_000] * len(closes)
    return pd.DataFrame(
        {
            "close": closes,
            "volume": volumes,
            "gap_pct": [0.0] * len(closes),
        }
    )


def _build_features() -> dict:
    return {
        "rsi_2": 3.0,
        "pct_above_sma200": 5.0,
        "pct_above_sma50": 3.0,
        "sma_50": 102.0,
        "sma_200": 98.0,
        "streak": -3,
        "dist_from_5d_low": 0.5,
        "rvol": 0.8,
        "close": 100.0,
        "atr_14": 4.0,
    }


def test_mean_reversion_respects_configurable_exit_and_entry_params():
    signal = score_mean_reversion(
        ticker="300001.SZ",
        df=_build_df(),
        features=_build_features(),
        stop_atr_mult=1.0,
        target_1_atr_mult=2.0,
        target_2_atr_mult=3.5,
        max_entry_atr_mult=0.1,
        holding_period=5,
    )

    assert signal is not None
    assert signal.stop_loss == 96.0
    assert signal.target_1 == 108.0
    assert signal.max_entry_price == 100.4
    assert signal.holding_period == 5


def test_mean_reversion_atr_mode_uses_atr_targets():
    signal = score_mean_reversion(
        ticker="300001.SZ",
        df=_build_df(),
        features=_build_features(),
        target_mode="atr",
        target_1_atr_mult=2.0,
        target_2_atr_mult=3.5,
    )

    assert signal is not None
    assert signal.target_1 == 108.0
    assert signal.target_2 == 114.0
