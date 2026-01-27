"""
Regime Gate Functions

Market regime checking (SPY/VIX filters).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional


def check_regime(params: dict, asof_date: Optional[str] = None) -> dict:
    """
    Check market regime: SPY above MA20 AND VIX <= threshold.
    
    Args:
        params: Dict with regime gate parameters:
            - spy_symbol: SPY ticker (default: "SPY")
            - vix_symbol: VIX ticker (default: "^VIX")
            - spy_ma_days: MA period (default: 20)
            - vix_max: Maximum VIX threshold (default: 25.0)
    
    Returns:
        Dict with:
            - ok: bool (True if regime is OK)
            - spy_last: float
            - spy_ma: float
            - vix_last: float
            - spy_above_ma: bool
            - vix_ok: bool
            - message: str
    """
    out = {
        "ok": True,
        "spy_last": np.nan,
        "spy_ma": np.nan,
        "vix_last": np.nan,
        "spy_above_ma": None,
        "vix_ok": None,
        "message": ""
    }
    
    try:
        spy_symbol = params.get("spy_symbol", "SPY")
        vix_symbol = params.get("vix_symbol", "^VIX")
        spy_ma_days = params.get("spy_ma_days", 20)
        vix_max = params.get("vix_max", 25.0)

        if asof_date:
            end_dt = pd.to_datetime(asof_date).to_pydatetime()
            start_dt = end_dt - pd.Timedelta(days=120)
            spy = yf.download(spy_symbol, start=start_dt, end=end_dt + pd.Timedelta(days=1), interval="1d", progress=False)
            vix = yf.download(vix_symbol, start=start_dt, end=end_dt + pd.Timedelta(days=1), interval="1d", progress=False)
        else:
            # Pull a small window for SPY/VIX (to "now")
            spy = yf.download(spy_symbol, period="3mo", interval="1d", progress=False)
            vix = yf.download(vix_symbol, period="3mo", interval="1d", progress=False)

        spy_close = spy["Close"].dropna()
        vix_close = vix["Close"].dropna()

        if len(spy_close) < spy_ma_days + 1 or len(vix_close) < 5:
            out["message"] = "Regime data insufficient; skipping gate."
            return out

        # Extract scalar values properly
        spy_last_val = spy_close.iloc[-1]
        if isinstance(spy_last_val, pd.Series):
            spy_last_val = spy_last_val.iloc[0]
        out["spy_last"] = float(spy_last_val)
        
        spy_ma_val = spy_close.tail(spy_ma_days).mean()
        if isinstance(spy_ma_val, pd.Series):
            spy_ma_val = spy_ma_val.iloc[0]
        out["spy_ma"] = float(spy_ma_val)
        
        vix_last_val = vix_close.iloc[-1]
        if isinstance(vix_last_val, pd.Series):
            vix_last_val = vix_last_val.iloc[0]
        out["vix_last"] = float(vix_last_val)

        out["spy_above_ma"] = out["spy_last"] >= out["spy_ma"]
        out["vix_ok"] = out["vix_last"] <= vix_max

        out["ok"] = bool(out["spy_above_ma"] and out["vix_ok"])
        out["message"] = (
            f"SPY={out['spy_last']:.2f} vs MA{spy_ma_days}={out['spy_ma']:.2f} "
            f"({'OK' if out['spy_above_ma'] else 'RISK-OFF'}); "
            f"VIX={out['vix_last']:.2f} (<= {vix_max:.2f} is {'OK' if out['vix_ok'] else 'RISK-OFF'})."
        )
        return out
    except Exception as e:
        out["message"] = f"Regime gate error; skipping gate. ({e})"
        return out

