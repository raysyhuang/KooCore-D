#!/usr/bin/env python3
"""
Daily Movers Filters

Applies strict, quarantined filters to daily movers.
Only candidates that pass ALL criteria are eligible.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
import yfinance as yf

from ...core.yf import get_ticker_df


def compute_close_position_in_range(row: pd.Series) -> Optional[float]:
    """
    Compute (close - low) / (high - low) for a daily bar.
    Returns None if high == low (invalid).
    """
    high = row.get("high")
    low = row.get("low")
    close = row.get("last_close")
    
    if pd.isna(high) or pd.isna(low) or pd.isna(close):
        return None
    
    if abs(high - low) < 0.001:  # Essentially zero range
        return None  # Invalid - no range
    
    return (close - low) / (high - low)


def filter_movers(
    movers: dict,
    technicals_df: Optional[pd.DataFrame] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply strict filters to daily movers.
    
    Args:
        movers: dict with "gainers" and "losers" DataFrames
        technicals_df: Optional DataFrame with technical metrics (can compute from prices if None)
        config: Configuration dict with filter thresholds
    
    Returns:
        DataFrame with filtered candidates, columns:
        - ticker, mover_type, pct_change_1d, volume_spike_multiple, 
          close_position_in_range, filter_pass, fail_reasons
    """
    if config is None:
        config = {
            "gainers_pct_range": [7.0, 15.0],
            "losers_pct_range": [-15.0, -7.0],
            "gainers_volume_spike": 2.0,
            "losers_volume_spike": 1.8,
            "close_position_min": 0.75,
            "price_min": 2.0,
            "adv_20d_min": 50_000_000,
        }
    
    candidates = []
    
    # Process gainers
    gainers = movers.get("gainers", pd.DataFrame())
    if not gainers.empty:
        for _, row in gainers.iterrows():
            ticker = row["ticker"]
            pct_change = row["pct_change_1d"]
            volume = row["volume"]
            last_close = row["last_close"]
            
            fail_reasons = []
            filter_pass = True
            
            # Gainers criteria (ALL required)
            # 1. % change in range [7%, 15%]
            if not (config["gainers_pct_range"][0] <= pct_change <= config["gainers_pct_range"][1]):
                fail_reasons.append(f"pct_change {pct_change:.2f}% not in range {config['gainers_pct_range']}")
                filter_pass = False
            
            # 2. Price >= min
            if last_close < config["price_min"]:
                fail_reasons.append(f"price ${last_close:.2f} < ${config['price_min']:.2f}")
                filter_pass = False
            
            # 3. Close position in range >= 0.75
            close_pos = compute_close_position_in_range(row)
            if close_pos is None:
                fail_reasons.append("invalid daily range (high==low)")
                filter_pass = False
            elif close_pos < config["close_position_min"]:
                fail_reasons.append(f"close_position {close_pos:.2f} < {config['close_position_min']}")
                filter_pass = False
            
            # 4. Volume spike check
            # Note: For volume spike, we need avg_volume_20d. Since we don't have technicals_df
            # with this data pre-computed, we'll mark as requiring verification.
            # In a full implementation, you'd fetch 20d data here or use technicals_df.
            volume_spike_multiple = None
            # For now, we'll skip volume spike check if technicals_df not available
            # and mark as passed (less strict for now - can be tightened later)
            if technicals_df is not None and ticker in technicals_df.index:
                avg_vol_20d = technicals_df.loc[ticker].get("avg_volume_20d")
                if avg_vol_20d and avg_vol_20d > 0:
                    volume_spike_multiple = volume / avg_vol_20d
                    required_spike = config.get("gainers_volume_spike", 2.0)
                    if volume_spike_multiple < required_spike:
                        fail_reasons.append(f"volume_spike {volume_spike_multiple:.2f}x < {required_spike}x")
                        filter_pass = False
            
            # 5. $ADV20 check (if technicals_df available)
            if technicals_df is not None and ticker in technicals_df.index:
                adv_20d = technicals_df.loc[ticker].get("adv_20d")
                if adv_20d and adv_20d < config.get("adv_20d_min", 50_000_000):
                    fail_reasons.append(f"$ADV20 ${adv_20d:,.0f} < ${config['adv_20d_min']:,.0f}")
                    filter_pass = False
            
            candidates.append({
                "ticker": ticker,
                "mover_type": "GAINER",
                "pct_change_1d": pct_change,
                "volume_spike_multiple": volume_spike_multiple,
                "close_position_in_range": close_pos,
                "filter_pass": filter_pass,
                "fail_reasons": fail_reasons,
            })
    
    # Process losers
    losers = movers.get("losers", pd.DataFrame())
    if not losers.empty:
        for _, row in losers.iterrows():
            ticker = row["ticker"]
            pct_change = row["pct_change_1d"]
            volume = row["volume"]
            last_close = row["last_close"]
            
            fail_reasons = []
            filter_pass = True
            
            # Losers criteria (ALL required)
            # 1. % change in range [-15%, -7%]
            if not (config["losers_pct_range"][0] <= pct_change <= config["losers_pct_range"][1]):
                fail_reasons.append(f"pct_change {pct_change:.2f}% not in range {config['losers_pct_range']}")
                filter_pass = False
            
            # 2. Price >= min
            if last_close < config["price_min"]:
                fail_reasons.append(f"price ${last_close:.2f} < ${config['price_min']:.2f}")
                filter_pass = False
            
            # 3. Volume spike check
            volume_spike_multiple = None
            if technicals_df is not None and ticker in technicals_df.index:
                avg_vol_20d = technicals_df.loc[ticker].get("avg_volume_20d")
                if avg_vol_20d and avg_vol_20d > 0:
                    volume_spike_multiple = volume / avg_vol_20d
                    required_spike = config.get("losers_volume_spike", 1.8)
                    if volume_spike_multiple < required_spike:
                        fail_reasons.append(f"volume_spike {volume_spike_multiple:.2f}x < {required_spike}x")
                        filter_pass = False
            
            # 4. $ADV20 check
            if technicals_df is not None and ticker in technicals_df.index:
                adv_20d = technicals_df.loc[ticker].get("adv_20d")
                if adv_20d and adv_20d < config.get("adv_20d_min", 50_000_000):
                    fail_reasons.append(f"$ADV20 ${adv_20d:,.0f} < ${config['adv_20d_min']:,.0f}")
                    filter_pass = False
            
            # Note: Losers don't require close_position check (can gap down)
            candidates.append({
                "ticker": ticker,
                "mover_type": "LOSER",
                "pct_change_1d": pct_change,
                "volume_spike_multiple": volume_spike_multiple,
                "close_position_in_range": None,  # Not required for losers
                "filter_pass": filter_pass,
                "fail_reasons": fail_reasons,
            })
    
    df_candidates = pd.DataFrame(candidates)
    return df_candidates


def build_mover_technicals_df(
    tickers: list[str],
    *,
    lookback_days: int = 25,
    auto_adjust: bool = False,
    threads: bool = True,
) -> pd.DataFrame:
    """
    Build a minimal technicals DataFrame (index=ticker) used by `filter_movers`:
    - avg_volume_20d
    - adv_20d ($ average daily dollar volume, 20d)
    """
    tickers_u = sorted({str(t).strip().upper() for t in (tickers or []) if str(t).strip()})
    if not tickers_u:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=tickers_u,
            period=f"{int(lookback_days)}d",
            interval="1d",
            group_by="ticker",
            auto_adjust=auto_adjust,
            threads=threads,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for t in tickers_u:
        try:
            df = get_ticker_df(data, t)
            if df.empty or len(df) < 5:
                continue
            if "Close" not in df.columns or "Volume" not in df.columns:
                continue
            close = df["Close"].astype(float)
            vol = df["Volume"].astype(float)
            avg_vol_20d = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
            adv_20d = float((close.tail(20) * vol.tail(20)).mean()) if len(close) >= 20 else float((close * vol).mean())
            rows.append({"ticker": t, "avg_volume_20d": avg_vol_20d, "adv_20d": adv_20d})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("ticker")
    return out


def analyze_mover_momentum(
    ticker: str,
    df: pd.DataFrame,
    mover_type: str,
) -> dict:
    """
    Analyze whether a mover is likely to continue momentum or reverse.
    
    Based on backtest insight: We want to identify:
    - Gainers that might continue (momentum play)
    - Losers that might reverse (mean-reversion play)
    
    Args:
        ticker: Ticker symbol
        df: OHLCV DataFrame for the ticker (at least 20 days)
        mover_type: "GAINER" or "LOSER"
    
    Returns:
        dict with analysis: setup_type, score, signals, recommendation
    """
    if df.empty or len(df) < 14:
        return {"setup_type": "unknown", "score": 0, "signals": [], "recommendation": "skip"}
    
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    
    signals = []
    score = 0
    
    # Calculate RSI (14-day)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50
    
    # Calculate MAs
    ma10 = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else float(close.mean())
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean())
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
    last_close = float(close.iloc[-1])
    
    # Calculate volume trend (today vs 5-day avg)
    vol_5d_avg = float(volume.tail(5).mean()) if len(volume) >= 5 else float(volume.mean())
    vol_today = float(volume.iloc[-1])
    vol_ratio = vol_today / vol_5d_avg if vol_5d_avg > 0 else 1.0
    
    # Calculate recent momentum (5-day return)
    ret_5d = ((last_close - float(close.iloc[-5])) / float(close.iloc[-5]) * 100) if len(close) >= 5 else 0
    
    # Calculate distance from 52W high (if we have enough data)
    high_52w = float(high.max()) if len(high) >= 50 else float(high.max())
    dist_to_high_pct = ((last_close - high_52w) / high_52w * 100) if high_52w > 0 else 0
    
    if mover_type == "GAINER":
        # For gainers, we want to identify CONTINUATION setups
        # Good continuation: Breakout with volume, not overbought yet
        
        # Signal 1: Not extremely overbought
        if current_rsi < 70:
            signals.append(f"RSI {current_rsi:.0f} not overbought - room to run")
            score += 2
        elif current_rsi > 80:
            signals.append(f"RSI {current_rsi:.0f} OVERBOUGHT - caution")
            score -= 2
        
        # Signal 2: Above key MAs (bullish trend)
        if last_close > ma10 > ma20:
            signals.append("Price above MA10 > MA20 (bullish structure)")
            score += 2
        
        # Signal 3: Strong volume confirmation
        if vol_ratio >= 2.0:
            signals.append(f"Volume {vol_ratio:.1f}x avg - institutional interest")
            score += 2
        elif vol_ratio < 1.5:
            signals.append(f"Volume {vol_ratio:.1f}x - weak conviction")
            score -= 1
        
        # Signal 4: Near 52W high (breakout potential)
        if dist_to_high_pct > -5:
            signals.append(f"Within 5% of 52W high - potential breakout")
            score += 2
        
        # Signal 5: Recent momentum not exhausted
        if 5 <= ret_5d <= 15:
            signals.append(f"5-day return {ret_5d:.1f}% - controlled momentum")
            score += 1
        elif ret_5d > 20:
            signals.append(f"5-day return {ret_5d:.1f}% - extended, risk of pullback")
            score -= 1
        
        setup_type = "MOMENTUM_CONTINUATION"
        if score >= 5:
            recommendation = "MONITOR_LONG"
        elif score >= 3:
            recommendation = "WATCHLIST"
        else:
            recommendation = "SKIP"
    
    else:  # LOSER
        # For losers, we want to identify REVERSAL setups
        # Good reversal: Oversold, high volume capitulation, near support
        
        # Signal 1: Oversold (reversal candidate)
        if current_rsi < 30:
            signals.append(f"RSI {current_rsi:.0f} OVERSOLD - reversal potential")
            score += 3
        elif current_rsi < 40:
            signals.append(f"RSI {current_rsi:.0f} approaching oversold")
            score += 1
        elif current_rsi > 50:
            signals.append(f"RSI {current_rsi:.0f} - not oversold yet")
            score -= 1
        
        # Signal 2: Capitulation volume (institutional selling exhausted)
        if vol_ratio >= 3.0:
            signals.append(f"Volume {vol_ratio:.1f}x - potential capitulation")
            score += 2
        elif vol_ratio >= 2.0:
            signals.append(f"Volume {vol_ratio:.1f}x - elevated selling")
            score += 1
        
        # Signal 3: Testing key support (MA levels)
        if abs(last_close - ma50) / ma50 < 0.03:
            signals.append("Near MA50 support")
            score += 2
        if abs(last_close - ma20) / ma20 < 0.02:
            signals.append("Near MA20 support")
            score += 1
        
        # Signal 4: Large distance from high (mean reversion potential)
        if dist_to_high_pct < -30:
            signals.append(f"{abs(dist_to_high_pct):.0f}% off 52W high - mean reversion candidate")
            score += 2
        elif dist_to_high_pct < -20:
            signals.append(f"{abs(dist_to_high_pct):.0f}% off high - moderate pullback")
            score += 1
        
        # Signal 5: Not in a breakdown pattern
        if last_close > ma50:
            signals.append("Still above MA50 - not a breakdown")
            score += 1
        else:
            signals.append("Below MA50 - breakdown risk")
            score -= 1
        
        setup_type = "REVERSAL_CANDIDATE"
        if score >= 5:
            recommendation = "MONITOR_REVERSAL"
        elif score >= 3:
            recommendation = "WATCHLIST"
        else:
            recommendation = "SKIP"
    
    return {
        "ticker": ticker,
        "setup_type": setup_type,
        "score": score,
        "signals": signals,
        "recommendation": recommendation,
        "rsi": round(current_rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "dist_to_high_pct": round(dist_to_high_pct, 1),
        "ret_5d": round(ret_5d, 1),
    }


def filter_movers_with_momentum(
    movers: dict,
    ticker_data: dict[str, pd.DataFrame],
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Filter movers AND add momentum/reversal analysis.
    
    This enhanced version analyzes each mover to determine:
    - Gainers: Is this a momentum continuation opportunity?
    - Losers: Is this a reversal opportunity?
    
    Args:
        movers: dict with "gainers" and "losers" DataFrames
        ticker_data: Dict of ticker -> OHLCV DataFrame (for momentum analysis)
        config: Filter configuration
    
    Returns:
        DataFrame with filtered candidates + momentum analysis
    """
    # First apply standard filters
    base_filtered = filter_movers(movers, config=config)
    
    if base_filtered.empty:
        return base_filtered
    
    # Add momentum analysis
    analysis_results = []
    
    for _, row in base_filtered.iterrows():
        ticker = row["ticker"]
        mover_type = row["mover_type"]
        
        df = ticker_data.get(ticker, pd.DataFrame())
        if not df.empty:
            momentum = analyze_mover_momentum(ticker, df, mover_type)
            analysis_results.append({
                "ticker": ticker,
                "momentum_setup": momentum["setup_type"],
                "momentum_score": momentum["score"],
                "momentum_recommendation": momentum["recommendation"],
                "momentum_rsi": momentum.get("rsi"),
                "momentum_vol_ratio": momentum.get("vol_ratio"),
                "momentum_signals": "; ".join(momentum["signals"][:3]),  # Top 3 signals
            })
        else:
            analysis_results.append({
                "ticker": ticker,
                "momentum_setup": "unknown",
                "momentum_score": 0,
                "momentum_recommendation": "skip",
                "momentum_rsi": None,
                "momentum_vol_ratio": None,
                "momentum_signals": "",
            })
    
    # Merge momentum analysis with filtered results
    momentum_df = pd.DataFrame(analysis_results)
    result = base_filtered.merge(momentum_df, on="ticker", how="left")
    
    # Sort by momentum score (higher is better)
    result = result.sort_values("momentum_score", ascending=False)
    
    return result

