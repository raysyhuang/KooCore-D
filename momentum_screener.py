"""
30-Day Momentum Screener PRO (Dynamic Universe + Regime Gate + News + Earnings + LLM Packets)

Requirements:
  pip install yfinance pandas numpy lxml

Outputs:
  All files are saved to a date-based folder (YYYY-MM-DD) with date-stamped filenames:
  - YYYY-MM-DD/30d_momentum_candidates_YYYY-MM-DD.csv
  - YYYY-MM-DD/30d_breakout_candidates_YYYY-MM-DD.csv
  - YYYY-MM-DD/30d_reversal_candidates_YYYY-MM-DD.csv
  - YYYY-MM-DD/news_dump_YYYY-MM-DD.csv
  - YYYY-MM-DD/llm_packets_YYYY-MM-DD.txt

Disclaimer:
  Candidate generator only. Not financial advice.
"""

from __future__ import annotations
import os
import sys
import time
import datetime as dt
from io import StringIO
from typing import Callable
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import yfinance as yf  # pyright: ignore[reportMissingImports]

# Leveraged ETFs (treated separately - will be detected dynamically)
LEVERAGED_ETFS = set()  # Can be populated dynamically if needed

# Dilution and catalyst keywords for headline scanning
DILUTION_KEYWORDS = [
    "offering", "secondary", "atm", "at-the-market", "convertible",
    "shelf", "dilution", "warrant", "equity raise", "registered direct"
]

CATALYST_KEYWORDS = {
    "m&a": ["acquire", "acquisition", "merger", "buyout", "takeover"],
    "contract": ["contract", "award", "partnership", "agreement", "deal"],
    "guidance": ["guidance", "raises", "beats", "outlook", "forecasts"],
    "fda/clinical": ["fda", "phase", "trial", "clinical", "pdufa"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto"],
}

# ============================================================
# 2) Parameters (tune these)
# ============================================================
PARAMS = dict(
    # Universe
    universe_mode="SP500+NASDAQ100+R2000",  # options: "SP500", "SP500+NASDAQ100", "SP500+NASDAQ100+R2000"
    universe_cache_file="universe_cache.csv",
    universe_cache_max_age_days=7,

    # Attention Pool (EOD by default)
    attention_rvol_min=1.8,
    attention_atr_pct_min=3.5,
    attention_min_abs_day_move_pct=3.0,
    attention_lookback_days=120,
    attention_chunk_size=200,

    # Quality filters
    price_min=7.0,
    avg_vol_min=1_000_000,             # share volume
    avg_dollar_vol_min=20_000_000,      # NEW: liquidity gate ($20M/day)
    rvol_min=2.0,
    atr_pct_min=4.0,
    near_high_max_pct=8.0,
    rsi_reversal_max=35.0,

    # Setup-specific tightening (optional)
    breakout_rsi_min=55.0,             # NEW: avoid weak "near-high" names
    reversal_dist_to_high_min_pct=15.0,# NEW: avoid tiny pullbacks flagged as reversal
    reversal_rsi_max=32.0,             # NEW: slightly stricter than 35 for mean reversion

    # Output controls
    top_n_breakout=15,
    top_n_reversal=15,
    top_n_total=25,

    # Lookbacks
    lookback_days=365,

    # Regime gate
    enable_regime_gate=True,
    spy_symbol="SPY",
    vix_symbol="^VIX",
    spy_ma_days=20,
    vix_max=25.0,
    regime_action="WARN",

    # Intraday mode (OPTIONAL)
    enable_intraday_attention=False,    # set True if you want prorated intraday RVOL
    intraday_interval="5m",
    intraday_lookback_days=5,          # yfinance has limits; keep small
    market_open_buffer_min=20,         # ignore first N minutes (noise)
    intraday_rvol_min=2.0,

    # News
    news_max_items=25,
    packet_headlines=12,
    throttle_sec=0.15,

    # Manual include (from file)
    manual_include_file="tickers/manual_include_tickers.txt",  # one ticker per line
    r2000_include_file="tickers/r2000.txt",  # R2000 tickers (optional)
    manual_include_mode="ALWAYS",  # "ALWAYS" or "ONLY_IF_IN_UNIVERSE"
)


# ============================================================
# Indicators
# ============================================================
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def is_market_open() -> bool:
    """
    Rough check if US equity market is currently open (9:30 AM - 4:00 PM ET, weekdays).
    Returns True if market is likely open, False otherwise.
    """
    try:
        now = pd.Timestamp.now(tz="America/New_York")
        # Check if weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        # Check time: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close
    except Exception:
        # If timezone conversion fails, assume market might be open (conservative)
        return True

def headline_flags(titles: list[str]) -> dict:
    """
    Scan headlines for dilution risk and catalyst keywords.
    Returns dict with dilution_flag (0/1) and catalyst_tags (comma-separated string).
    """
    t = " | ".join([str(x).lower() for x in titles if str(x).strip()])
    dilution = any(k in t for k in DILUTION_KEYWORDS)
    tags = []
    for k, kws in CATALYST_KEYWORDS.items():
        if any(w in t for w in kws):
            tags.append(k)
    return {"dilution_flag": int(dilution), "catalyst_tags": ",".join(sorted(set(tags)))}


# ============================================================
# Dynamic Universe Functions
# ============================================================
def get_sp500_universe() -> list[str]:
    """
    Get S&P 500 ticker list from Wikipedia.
    Returns list of tickers (handles BRK.B -> BRK-B conversion for Yahoo).
    """
    import requests  # pyright: ignore[reportMissingModuleSource]
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Retry logic with timeout
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            table = pd.read_html(StringIO(response.text))[0]
            tickers = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            return tickers
        except Exception as e:
            if attempt < 2:
                time.sleep(1)  # Brief pause before retry
                continue
            # Final fallback: try without headers
            try:
                table = pd.read_html(url, timeout=10)[0]
                tickers = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
                return tickers
            except Exception as e2:
                print(f"Error fetching S&P 500 universe after retries: {e2}")
                return []


def _read_wiki_table(url: str, table_pick: Callable, retries: int = 3) -> pd.DataFrame:
    # robust read_html: sometimes table index shifts
    for attempt in range(retries):
        try:
            tables = pd.read_html(url, timeout=10)
            return table_pick(tables)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            raise

def _normalize_ticker_for_yahoo(sym: str) -> str:
    return str(sym).strip().replace(".", "-")

def load_tickers_from_file(filepath: str) -> list[str]:
    """Load tickers from a single file, one per line."""
    if not filepath or not os.path.exists(filepath):
        return []

    out: list[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # allow comments: "MSTR  # bitcoin proxy"
            t = line.split("#")[0].strip()
            t = _normalize_ticker_for_yahoo(t)
            if t:
                out.append(t)

    return out

def load_manual_include_tickers(p: dict) -> list[str]:
    """Load tickers from manual include file(s). Supports both manual_include_file and r2000_include_file."""
    all_tickers = []
    
    # Load from manual include file
    manual_file = p.get("manual_include_file", "")
    if manual_file:
        all_tickers.extend(load_tickers_from_file(manual_file))
    
    # Load from R2000 file if specified
    r2000_file = p.get("r2000_include_file", "")
    if r2000_file:
        all_tickers.extend(load_tickers_from_file(r2000_file))
    
    return sorted(set(all_tickers))

def get_nasdaq100_universe() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        df = _read_wiki_table(url, lambda tables: next(t for t in tables if "Ticker" in t.columns))
        tickers = df["Ticker"].astype(str).map(_normalize_ticker_for_yahoo).tolist()
        return sorted(set(tickers))
    except Exception:
        return []

def get_russell2000_universe() -> list[str]:
    """
    Best-effort Russell 2000 universe via iShares IWM holdings CSV.
    Robust to variable header/disclaimer blocks by auto-locating the real header row.
    """
    import requests  # pyright: ignore[reportMissingModuleSource]

    url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?tab=all&fileType=csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/octet-stream,*/*",
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        text = r.text

        # Find the first line that looks like the true table header.
        # iShares typically uses "Ticker" as a column.
        lines = text.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            # Normalize whitespace; look for a CSV header containing Ticker/Symbol
            l = line.strip().strip("\ufeff")
            if l.startswith("Ticker,") or (("Ticker" in l.split(",")) and ("Name" in l or "Issuer" in l or "Sector" in l)):
                header_idx = i
                break

        if header_idx is None:
            return []

        # Rebuild CSV from the detected header downwards
        csv_text = "\n".join(lines[header_idx:])

        df = pd.read_csv(StringIO(csv_text))
        # Find ticker column robustly
        ticker_col = None
        for c in df.columns:
            if str(c).strip().lower() in {"ticker", "symbol"}:
                ticker_col = c
                break
        if ticker_col is None:
            # fallback: any column containing 'ticker' or 'symbol'
            cols = [c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()]
            if not cols:
                return []
            ticker_col = cols[0]

        tickers = (
            df[ticker_col]
            .dropna()
            .astype(str)
            .map(_normalize_ticker_for_yahoo)
            .tolist()
        )

        # Filter obvious non-tickers and skip disclaimer rows
        # Valid tickers should be short, alphanumeric (with dashes), and not contain disclaimer keywords
        disclaimer_keywords = ["content", "blackrock", "ishares", "prospectus", "copyright", "trademark"]
        tickers_filtered = []
        for t in tickers:
            t_clean = str(t).strip()
            # Skip if it's a disclaimer row (too long or contains disclaimer keywords)
            if len(t_clean) > 10 or any(keyword in t_clean.lower() for keyword in disclaimer_keywords):
                continue
            # Must be valid ticker format
            if t_clean.isascii() and 1 <= len(t_clean) <= 6 and t_clean.replace("-", "").isalnum():
                tickers_filtered.append(t_clean)
        
        # Sanity check - should have a reasonable number of tickers (Russell 2000 has ~2000 stocks)
        if len(tickers_filtered) > 500:
            return sorted(set(tickers_filtered))
        else:
            return []

    except Exception:
        return []

def load_universe_from_cache(p: dict) -> list[str]:
    path = p.get("universe_cache_file", "universe_cache.csv")
    max_age = int(p.get("universe_cache_max_age_days", 7))
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if "Ticker" not in df.columns or "asof" not in df.columns:
            return []
        asof = pd.to_datetime(df["asof"].iloc[0], errors="coerce")
        if pd.isna(asof):
            return []
        age_days = (pd.Timestamp.utcnow().normalize() - asof.tz_localize(None).normalize()).days
        if age_days > max_age:
            return []
        tickers = df["Ticker"].dropna().astype(str).tolist()
        return sorted(set(tickers))
    except Exception:
        return []

def save_universe_to_cache(tickers: list[str], p: dict) -> None:
    path = p.get("universe_cache_file", "universe_cache.csv")
    df = pd.DataFrame({"Ticker": sorted(set(tickers))})
    df["asof"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    df.to_csv(path, index=False)

def get_dynamic_universe(p: dict) -> list[str]:
    mode = p.get("universe_mode", "SP500+NASDAQ100+R2000").upper()
    
    # Set cache filename to include mode to avoid cache reuse across different modes
    original_cache_file = p.get("universe_cache_file", "universe_cache.csv")
    mode_cache_file = f"universe_cache_{mode.replace('+', '_')}.csv"
    p["universe_cache_file"] = mode_cache_file
    
    try:
        cached = load_universe_from_cache(p)
        if cached:
            return cached

        sp = get_sp500_universe()
        n100 = get_nasdaq100_universe() if "NASDAQ100" in mode else []
        r2k = get_russell2000_universe() if "R2000" in mode else []
        
        # Warn if R2000 fetch failed when it was requested
        if not r2k and "R2000" in mode:
            print("[WARN] Russell 2000 fetch failed; proceeding with SP500 + NASDAQ100 only.")

        tickers = sorted(set(sp + n100 + r2k))
        # if Russell fetch fails, you still get sp + n100
        if tickers:
            save_universe_to_cache(tickers, p)
        
        return tickers
    finally:
        # Restore original cache_file setting to avoid side effects
        p["universe_cache_file"] = original_cache_file


def _minutes_since_open(ts_local: pd.Timestamp) -> int:
    # assumes US equities; rough but useful. For best accuracy, integrate pandas_market_calendars later.
    open_time = ts_local.normalize() + pd.Timedelta(hours=9, minutes=30)
    return int(max(0, (ts_local - open_time).total_seconds() // 60))

def _intraday_prorated_rvol(ticker: str, p: dict) -> float:
    """
    Estimate intraday RVOL:
    (today volume so far / expected volume so far) where expected volume so far
    is derived from prior days' intraday cumulative volume at the same time-of-day.
    Falls back to NaN on any failure.
    """
    try:
        interval = p.get("intraday_interval", "5m")
        days = int(p.get("intraday_lookback_days", 5))
        df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
        if df is None or df.empty or "Volume" not in df.columns:
            return np.nan

        # yfinance returns tz-aware sometimes; normalize to naive
        idx = pd.to_datetime(df.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        df = df.copy()
        df.index = idx
        df = df.dropna()

        if len(df) < 50:
            return np.nan

        now_local = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        mins = _minutes_since_open(now_local)
        if mins < int(p.get("market_open_buffer_min", 20)):
            return np.nan

        # Split by date
        df["date"] = df.index.date
        today = now_local.date()
        today_df = df[df["date"] == today]
        hist_df = df[df["date"] != today]

        if today_df.empty or hist_df.empty:
            return np.nan

        # cumulative volume so far today
        v_today = float(today_df["Volume"].sum())

        # expected cumulative volume by same time-of-day:
        # find bars with timestamp <= now_local time for each prior day
        t_cut = now_local.time()
        exp = []
        for d, g in hist_df.groupby("date"):
            g2 = g[g.index.time <= t_cut]
            if len(g2) >= 3:
                exp.append(float(g2["Volume"].sum()))
        if len(exp) < 2:
            return np.nan

        v_exp = float(np.median(exp))
        if v_exp <= 0:
            return np.nan

        return v_today / v_exp
    except Exception:
        return np.nan

def build_attention_pool(
    tickers: list[str],
    p: dict
) -> list[str]:
    """
    Build dynamic attention pool using objective signals (RVOL, ATR%, price moves).
    This answers: "What is the market paying attention to right now?"
    """
    # Check if market is open and handle intraday mode accordingly
    market_open = is_market_open()
    enable_intraday = p.get("enable_intraday_attention", False)
    
    if market_open and not enable_intraday:
        print("[WARNING] Market appears to be open, but intraday attention is disabled.")
        print("  EOD RVOL from partial day volume may produce false positives.")
        print("  Recommendation: Run after market close (best accuracy) or set enable_intraday_attention=True")
        # For automation, we continue but warn - user can adjust params
    elif market_open and enable_intraday:
        print("[INFO] Market is open, using intraday prorated RVOL for attention pool.")
    
    pool = []
    chunk_size = p.get("attention_chunk_size", 200)
    lookback_days = p.get("attention_lookback_days", 120)
    
    price_min = p.get("price_min", 5.0)
    avg_vol_min = p.get("avg_vol_min", 1_000_000)
    rvol_min = p.get("attention_rvol_min", 1.8)
    atr_pct_min = p.get("attention_atr_pct_min", 3.5)
    min_abs_day_move_pct = p.get("attention_min_abs_day_move_pct", None)

    print(f"Building attention pool from {len(tickers)} tickers (chunk size: {chunk_size})...")

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            data = yf.download(
                tickers=chunk,
                period=f"{lookback_days}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )

            for t in chunk:
                try:
                    df = data[t].dropna()
                    if len(df) < 40:
                        continue

                    close = df["Close"]
                    vol = df["Volume"]
                    
                    last = float(close.iloc[-1])
                    if last < price_min:
                        continue

                    avg20 = float(vol.tail(20).mean())
                    if avg20 < avg_vol_min:
                        continue

                    rvol = float(vol.iloc[-1] / avg20) if avg20 else np.nan
                    
                    # Intraday mode override if enabled
                    if p.get("enable_intraday_attention", False):
                        rvol_i = _intraday_prorated_rvol(t, p)
                        # if intraday is available, use it; else fall back to EOD rvol
                        if not np.isnan(rvol_i):
                            rvol = float(rvol_i)
                            if rvol < p.get("intraday_rvol_min", 2.0):
                                continue
                    
                    if np.isnan(rvol) or rvol < rvol_min:
                        continue

                    a = float(atr(df, 14).iloc[-1])
                    atr_pct_val = float(a / last * 100) if last else np.nan
                    if np.isnan(atr_pct_val) or atr_pct_val < atr_pct_min:
                        continue

                    if min_abs_day_move_pct is not None and len(df) >= 2:
                        prev = float(close.iloc[-2])
                        day_move = abs((last - prev) / prev * 100) if prev else 0.0
                        if day_move < min_abs_day_move_pct:
                            continue

                    pool.append(t)
                except Exception:
                    continue
        except Exception:
            continue

        # Progress indicator
        if (i + chunk_size) % (chunk_size * 5) == 0:
            print(f"  Processed {min(i + chunk_size, len(tickers))}/{len(tickers)} tickers, found {len(pool)} in pool so far...")

    return sorted(set(pool))


# ============================================================
# Regime Gate: SPY above MA20 AND VIX <= threshold
# ============================================================
def check_regime(p: dict) -> dict:
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
        # Pull a small window for SPY/VIX
        spy = yf.download(p["spy_symbol"], period="3mo", interval="1d", progress=False)
        vix = yf.download(p["vix_symbol"], period="3mo", interval="1d", progress=False)

        spy_close = spy["Close"].dropna()
        vix_close = vix["Close"].dropna()

        if len(spy_close) < p["spy_ma_days"] + 1 or len(vix_close) < 5:
            out["message"] = "Regime data insufficient; skipping gate."
            return out

        # Fix FutureWarning: extract scalar values properly
        spy_last_val = spy_close.iloc[-1]
        if isinstance(spy_last_val, pd.Series):
            spy_last_val = spy_last_val.iloc[0]
        out["spy_last"] = float(spy_last_val)
        
        spy_ma_val = spy_close.tail(p["spy_ma_days"]).mean()
        if isinstance(spy_ma_val, pd.Series):
            spy_ma_val = spy_ma_val.iloc[0]
        out["spy_ma"] = float(spy_ma_val)
        
        vix_last_val = vix_close.iloc[-1]
        if isinstance(vix_last_val, pd.Series):
            vix_last_val = vix_last_val.iloc[0]
        out["vix_last"] = float(vix_last_val)

        out["spy_above_ma"] = out["spy_last"] >= out["spy_ma"]
        out["vix_ok"] = out["vix_last"] <= p["vix_max"]

        out["ok"] = bool(out["spy_above_ma"] and out["vix_ok"])
        out["message"] = (
            f"SPY={out['spy_last']:.2f} vs MA{p['spy_ma_days']}={out['spy_ma']:.2f} "
            f"({'OK' if out['spy_above_ma'] else 'RISK-OFF'}); "
            f"VIX={out['vix_last']:.2f} (<= {p['vix_max']:.2f} is {'OK' if out['vix_ok'] else 'RISK-OFF'})."
        )
        return out
    except Exception as e:
        out["message"] = f"Regime gate error; skipping gate. ({e})"
        return out


# ============================================================
# Screener
# ============================================================
def screen_universe(tickers: list[str], p: dict) -> dict:
    """
    Returns dict:
      - breakout_df
      - reversal_df
      - combined_df (optional top_n_total merged)
    """
    if not tickers:
        return {"breakout_df": pd.DataFrame(), "reversal_df": pd.DataFrame(), "combined_df": pd.DataFrame()}

    # Download data for all tickers (yfinance handles errors internally and continues)
    # Some tickers may fail to download (network/SSL errors) - they'll be skipped
    print(f"  Downloading price data for {len(tickers)} tickers (this may take several minutes)...")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        data = yf.download(
            tickers=tickers,
            period=f"{p['lookback_days']}d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    print(f"  Download complete. Processing tickers...")

    breakout_rows, reversal_rows = [], []
    failed_count = 0
    total_tickers = len(tickers)
    progress_interval = max(50, total_tickers // 20)  # Show progress every 5% or every 50 tickers, whichever is larger

    for idx, t in enumerate(tickers):
        # Show progress every N tickers
        if (idx + 1) % progress_interval == 0 or (idx + 1) == total_tickers:
            pct = ((idx + 1) / total_tickers) * 100
            print(f"  Progress: {idx + 1}/{total_tickers} ({pct:.1f}%) | Found: {len(breakout_rows)} breakouts, {len(reversal_rows)} reversals", end="\r")
        try:
            # Access ticker data - will raise KeyError if download failed for this ticker
            df = data[t].dropna()
            if len(df) < 120:
                continue

            close = df["Close"]
            high = df["High"]
            vol = df["Volume"]

            last = float(close.iloc[-1])
            if last < p["price_min"]:
                continue

            avg20 = float(vol.tail(20).mean())
            if avg20 < p["avg_vol_min"]:
                continue

            adv20 = float((close.tail(20) * vol.tail(20)).mean())
            if adv20 < p.get("avg_dollar_vol_min", 0):
                continue

            rvol_val = float(vol.iloc[-1] / avg20) if avg20 else np.nan
            if np.isnan(rvol_val) or rvol_val < p["rvol_min"]:
                continue

            a = float(atr(df, 14).iloc[-1])
            atr_pct_val = float(a / last * 100) if last else np.nan
            if np.isnan(atr_pct_val) or atr_pct_val < p["atr_pct_min"]:
                continue

            rsi14_val = float(rsi(close, 14).iloc[-1])

            # Correct 52W high: use High, not Close
            high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
            dist_52w_high_pct = float((high_52w - last) / high_52w * 100) if high_52w else np.nan

            # MA structure (for trend confirmation)
            ma20 = float(sma(close, 20).iloc[-1]) if len(close) >= 20 else np.nan
            ma50 = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else np.nan
            
            above_ma20 = last >= ma20 if np.isfinite(ma20) else False
            above_ma50 = last >= ma50 if np.isfinite(ma50) else False
            
            # Returns
            ret20 = float((last / float(close.iloc[-21]) - 1) * 100) if len(close) > 21 else np.nan
            ret5 = float((last / float(close.iloc[-6]) - 1) * 100) if len(close) > 6 else np.nan

            # Setup flags with structure requirements
            is_breakout = (
                dist_52w_high_pct <= p["near_high_max_pct"]
                and rsi14_val >= p.get("breakout_rsi_min", 0)
                and above_ma20
                and above_ma50
                and (np.isnan(ret20) or ret20 > 0)
            )
            is_reversal = (
                rsi14_val <= p.get("reversal_rsi_max", p["rsi_reversal_max"])
                and (dist_52w_high_pct >= p.get("reversal_dist_to_high_min_pct", 0))
            )

            if not (is_breakout or is_reversal):
                continue

            # Normalized scoring: TapeScore + StructureScore + SetupBonus
            tape_score = (rvol_val * 2.0) + (atr_pct_val * 1.4)
            
            # Structure score components
            rsi_structure = max(0.0, (70 - abs(rsi14_val - 62))) / 20.0  # prefer RSI ~ 58–66
            dist_structure = max(0.0, (100 - dist_52w_high_pct)) / 20.0  # prefer closer to highs
            ma_structure = (2.0 if above_ma20 and above_ma50 else (1.0 if above_ma20 else 0.0))
            
            structure_score = rsi_structure + (dist_structure * 0.5) + ma_structure
            
            if is_breakout:
                setup_bonus = 4.0
                score = tape_score + structure_score + setup_bonus
                breakout_rows.append({
                    "Ticker": t,
                    "Last": round(last, 2),
                    "RVOL": round(rvol_val, 2),
                    "ATR%": round(atr_pct_val, 2),
                    "RSI14": round(rsi14_val, 1),
                    "Dist_to_52W_High%": round(dist_52w_high_pct, 2),
                    "$ADV20": round(adv20, 0),
                    "MA20": round(ma20, 2) if np.isfinite(ma20) else np.nan,
                    "MA50": round(ma50, 2) if np.isfinite(ma50) else np.nan,
                    "Above_MA20": int(above_ma20),
                    "Above_MA50": int(above_ma50),
                    "Ret20d%": round(ret20, 2) if np.isfinite(ret20) else np.nan,
                    "Setup": "Breakout",
                    "Score": round(score, 2),
                    "Is_Leveraged_ETF": t in LEVERAGED_ETFS
                })

            if is_reversal:
                # For reversals, reward higher ATR and more "room back to highs"
                reversal_structure = min(6.0, dist_52w_high_pct / 8.0)  # more room = better
                setup_bonus = 3.0
                score = tape_score + (structure_score * 0.7) + reversal_structure + setup_bonus
                reversal_rows.append({
                    "Ticker": t,
                    "Last": round(last, 2),
                    "RVOL": round(rvol_val, 2),
                    "ATR%": round(atr_pct_val, 2),
                    "RSI14": round(rsi14_val, 1),
                    "Dist_to_52W_High%": round(dist_52w_high_pct, 2),
                    "$ADV20": round(adv20, 0),
                    "MA20": round(ma20, 2) if np.isfinite(ma20) else np.nan,
                    "MA50": round(ma50, 2) if np.isfinite(ma50) else np.nan,
                    "Above_MA20": int(above_ma20),
                    "Above_MA50": int(above_ma50),
                    "Ret20d%": round(ret20, 2) if np.isfinite(ret20) else np.nan,
                    "Ret5d%": round(ret5, 2) if np.isfinite(ret5) else np.nan,
                    "Setup": "Reversal",
                    "Score": round(score, 2),
                    "Is_Leveraged_ETF": t in LEVERAGED_ETFS
                })

        except Exception:
            failed_count += 1
            continue

    # Clear the progress line and print summary
    print()  # New line after progress indicator
    if failed_count > 0:
        print(f"  Skipped {failed_count} ticker(s) due to download/data errors (network issues, invalid tickers, etc.)")
    print(f"  Screening complete: {len(breakout_rows)} breakout candidates, {len(reversal_rows)} reversal candidates")

    # Create DataFrames, handling empty case
    if breakout_rows:
        breakout_df = pd.DataFrame(breakout_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    else:
        breakout_df = pd.DataFrame(columns=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Setup", "Score", "Is_Leveraged_ETF"])
    
    if reversal_rows:
        reversal_df = pd.DataFrame(reversal_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    else:
        reversal_df = pd.DataFrame(columns=["Ticker", "Last", "RVOL", "ATR%", "RSI14", "Dist_to_52W_High%", "$ADV20", "MA20", "MA50", "Above_MA20", "Above_MA50", "Ret20d%", "Ret5d%", "Setup", "Score", "Is_Leveraged_ETF"])

    breakout_df = breakout_df.head(p.get("top_n_breakout", 15)) if not breakout_df.empty else breakout_df
    reversal_df = reversal_df.head(p.get("top_n_reversal", 15)) if not reversal_df.empty else reversal_df

    combined_df = pd.concat([breakout_df, reversal_df], ignore_index=True)
    combined_df = combined_df.sort_values("Score", ascending=False).reset_index(drop=True)
    combined_df = combined_df.head(p.get("top_n_total", 25)) if not combined_df.empty else combined_df

    return {"breakout_df": breakout_df, "reversal_df": reversal_df, "combined_df": combined_df}


# ============================================================
# News pulling (yfinance)
# ============================================================
def fetch_news_for_tickers(tickers: list[str], max_items: int, throttle_sec: float = 0.0) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            items = tk.news or []
            for it in items[:max_items]:
                ts = pd.to_datetime(it.get("providerPublishTime", None), unit="s", utc=True, errors="coerce")
                title = (it.get("title", "") or "").strip()
                if not title:
                    continue
                link = (it.get("link", "") or "").strip()
                pub = (it.get("publisher", "") or "").strip()
                rows.append({
                    "Ticker": t,
                    "published_utc": ts,
                    "published_local": (ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET") if pd.notna(ts) else ""),
                    "title": title,
                    "publisher": pub,
                    "link": link,
                    "type": (it.get("type", "") or "").strip(),
                })
        except Exception:
            pass
        if throttle_sec and throttle_sec > 0:
            time.sleep(throttle_sec)

    if not rows:
        return pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"])

    df = pd.DataFrame(rows)
    # Dedup by (Ticker, title) keep most recent
    df = df.sort_values(["Ticker", "published_utc"], ascending=[True, False])
    df = df.drop_duplicates(subset=["Ticker", "title"], keep="first")
    return df.reset_index(drop=True)


# ============================================================
# Earnings date attempt (best-effort)
# ============================================================
def get_next_earnings_date(ticker: str) -> str:
    """
    Best-effort attempt to fetch the NEXT (future) earnings date.
    
    Note: yfinance get_earnings_dates(limit=1) often returns the most recent historical
    date, not the next one. This function fetches multiple dates and filters for the
    next future date. Returns "Unknown" if no future dates are available in yfinance
    (which is common, as future earnings dates may not be published yet).
    """
    today = pd.Timestamp.now(tz=None).normalize()
    try:
        tk = yf.Ticker(ticker)

        # Pull several earnings dates and pick the next future one
        try:
            ed = tk.get_earnings_dates(limit=12)
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                # Convert index to datetime, handling timezone issues
                dates = pd.to_datetime(ed.index, errors="coerce")
                # Remove timezone and normalize to date-only for comparison
                if dates.tz is not None:
                    dates = dates.tz_localize(None)
                dates = dates.normalize()
                # Filter for future dates (on or after today)
                future = dates[dates >= today]
                if len(future) > 0:
                    return future.min().strftime("%Y-%m-%d")
        except Exception:
            pass

        # Fallback: try calendar method
        try:
            cal = tk.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                # Try to find any datetime-like cell that's in the future
                for val in cal.values.flatten().tolist():
                    if isinstance(val, (pd.Timestamp, np.datetime64)):
                        dt = pd.to_datetime(val)
                        if dt.tz is not None:
                            dt = dt.tz_localize(None)
                        if dt >= today:
                            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    except Exception:
        pass

    return "Unknown"


# ============================================================
# Manual headlines loader
# ============================================================
def load_manual_headlines(filepath: str = "manual_headlines.csv") -> pd.DataFrame:
    """
    Load manually added headlines from CSV file.
    Expected format: Ticker,Date,Source,Headline
    Returns empty DataFrame if file doesn't exist or has no valid rows.
    """
    try:
        df = pd.read_csv(filepath, comment="#")
        # Clean up: remove leading/trailing whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
        # Filter out comment rows and empty rows
        df = df[df["Ticker"].astype(str).str.strip().ne("")]
        df = df[df["Headline"].astype(str).str.strip().ne("")]
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"])


# ============================================================
# Catalyst completeness and trade plan helpers
# ============================================================
def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "N/A"

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def compute_catalyst_completeness(
    ticker: str,
    earnings_date: str,
    news_df: pd.DataFrame,
    manual_headlines_df: pd.DataFrame | None = None,
    lookback_days: int = 14,
) -> dict:
    """
    Produces a 0–100 "Catalyst Completeness" score (NOT bullishness).
    Higher = you actually have dated catalysts / headlines.
    Penalizes Unknown earnings + no headlines.

    Returns dict:
      - completeness_score (0-100)
      - penalties (list[str])
      - headline_count_recent
      - has_manual
      - earnings_known (bool)
    """
    penalties = []
    earnings_known = (earnings_date != "Unknown")

    # Recent headline count
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    count_recent = 0
    if news_df is not None and not news_df.empty:
        tmp = news_df[news_df["Ticker"].astype(str).str.strip().eq(ticker)].copy()
        if "published_utc" in tmp.columns:
            tmp["published_utc"] = pd.to_datetime(tmp["published_utc"], utc=True, errors="coerce")
            tmp = tmp[tmp["published_utc"].notna()]
            tmp = tmp[tmp["published_utc"] >= cutoff]
        # Dedup by title to avoid overstating
        if "title" in tmp.columns:
            tmp = tmp[tmp["title"].astype(str).str.strip().ne("")]
            tmp = tmp.drop_duplicates(subset=["title"], keep="first")
        count_recent = int(len(tmp))

    has_manual = False
    if manual_headlines_df is not None and not manual_headlines_df.empty:
        m = manual_headlines_df[manual_headlines_df["Ticker"].astype(str).str.strip().eq(ticker)]
        has_manual = bool(len(m) > 0)

    # Base completeness
    score = 70

    if not earnings_known:
        score -= 20
        penalties.append("Earnings date Unknown (penalize catalyst clarity).")

    if count_recent == 0 and not has_manual:
        score -= 25
        penalties.append("No recent headlines found (penalize narrative visibility).")
    elif count_recent < 3 and not has_manual:
        score -= 10
        penalties.append("Few recent headlines (weak narrative visibility).")
    elif count_recent >= 8:
        score += 5  # capped bonus, completeness only

    if has_manual:
        score += 10  # you intentionally injected catalysts

    score = int(max(0, min(100, score)))

    return {
        "completeness_score": score,
        "penalties": penalties,
        "headline_count_recent": count_recent,
        "has_manual": has_manual,
        "earnings_known": earnings_known,
    }

def standard_trade_plan_guidance(setup: str, last: float, atr_pct: float) -> dict:
    """
    Returns standardized plan scaffolding.
    Stops and targets are expressed in ATR units and percent terms.
    This is NOT a recommendation; it's a template for consistent analysis.
    """
    atr_pct = _safe_float(atr_pct, np.nan)
    last = _safe_float(last, np.nan)

    # ATR in dollars (approx) if needed
    atr_d = (last * atr_pct / 100.0) if (np.isfinite(last) and np.isfinite(atr_pct)) else np.nan

    # Conservative defaults
    if setup == "Breakout":
        entry = "Entry: Break & hold above prior day high OR key resistance on ≥1.5× volume; avoid chasing if >2× ATR extension."
        stop = "Stop: 1.2× ATR below breakout level (or below prior day low if tighter and logical)."
        tp = "TPs: TP1 = +1.0× ATR, TP2 = +2.0× ATR; trail stop after TP1."
    else:  # Reversal
        entry = "Entry: Reclaim MA20 (or prior swing level) AND RSI curl up; confirm with a green day + elevated volume."
        stop = "Stop: Below recent swing low OR 1.0–1.3× ATR below entry (whichever is tighter and logical)."
        tp = "TPs: TP1 = MA50 / prior supply zone, TP2 = gap-fill / next resistance; reduce into strength."

    size = "Position sizing: risk 1–2% of account per trade. Shares = (AccountRisk$) / (StopDistance$)."

    return {
        "atr_dollar_est": atr_d,
        "entry_template": entry,
        "stop_template": stop,
        "tp_template": tp,
        "size_template": size,
    }


# ============================================================
# LLM packet builder
# ============================================================
def build_llm_packet(
    ticker: str,
    metrics_row: pd.Series,
    news_df: pd.DataFrame,
    max_headlines: int,
    regime_info: dict,
    manual_headlines_df: pd.DataFrame = None
) -> str:
    earnings_date = get_next_earnings_date(ticker)

    # Filter news for this ticker
    if not news_df.empty:
        n = news_df[(news_df["Ticker"] == ticker) & news_df["title"].astype(str).str.strip().ne("")].copy()
        # Prefer freshest
        if "published_utc" in n.columns:
            n["published_utc"] = pd.to_datetime(n["published_utc"], utc=True, errors="coerce")
            n = n.sort_values("published_utc", ascending=False)
        # Dedup titles
        n = n.drop_duplicates(subset=["title"], keep="first").head(max_headlines)
    else:
        n = pd.DataFrame()

    headlines = []
    headline_titles = []  # For keyword scanning

    # Manual headlines first
    if manual_headlines_df is not None and not manual_headlines_df.empty:
        manual = manual_headlines_df[manual_headlines_df["Ticker"].astype(str).str.strip().eq(ticker)]
        if not manual.empty:
            for _, r in manual.iterrows():
                date_str = r.get("Date", "N/A")
                source = r.get("Source", "Manual")
                headline = r.get("Headline", "")
                if str(headline).strip():
                    headline_titles.append(str(headline).strip())
                    headlines.append(f"- [{date_str}] {source}: {headline}")

    # yfinance headlines
    if not n.empty:
        for _, r in n.iterrows():
            ts = r.get("published_utc", pd.NaT)
            ts_str = ts.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(ts) else "N/A"
            publisher = (r.get("publisher", "") or "").strip()
            title = (r.get("title", "") or "").strip()
            link = (r.get("link", "") or "").strip()
            if title:
                headline_titles.append(title)
                if link:
                    headlines.append(f"- [{ts_str}] {publisher}: {title}\n  {link}")
                else:
                    headlines.append(f"- [{ts_str}] {publisher}: {title}")

    if not headlines:
        headlines.append("- (No headlines returned. Treat catalysts as Unknown. Check: earnings guidance, SEC filings, major contracts, sector/BTC correlation, terminal/X headlines.)")
    
    # Compute headline flags (dilution risk, catalyst tags)
    flags = headline_flags(headline_titles)

    # Completeness score (penalize Unknown + missing headlines)
    comp = compute_catalyst_completeness(
        ticker=ticker,
        earnings_date=earnings_date,
        news_df=news_df,
        manual_headlines_df=manual_headlines_df,
        lookback_days=14,
    )

    # Standard plan templates (by setup)
    setup = str(metrics_row.get("Setup", "")).strip() or "Unknown"
    last = _safe_float(metrics_row.get("Last", np.nan))
    atr_pct_val = _safe_float(metrics_row.get("ATR%", np.nan))
    plan = standard_trade_plan_guidance(setup=setup, last=last, atr_pct=atr_pct_val)

    leveraged_note = "YES (leveraged ETF: separate risk bucket)" if bool(metrics_row.get("Is_Leveraged_ETF", False)) else "NO"

    # Build a consistent "what to check next" list when data is missing
    missing_checks = []
    if earnings_date == "Unknown":
        missing_checks.append("Confirm next earnings date (company IR site / Nasdaq earnings calendar).")
    if comp["headline_count_recent"] == 0 and not comp["has_manual"]:
        missing_checks.append("Pull 14–30d headlines from a second source (SEC, PR, major outlets, terminal).")
    missing_checks.append("Check upcoming: FDA/clinical readouts, contracts, secondary offering/ATM, lockup expiry, guidance updates.")
    missing_checks.append("Check technical context: multi-year resistance, gap levels, supply zones, post-earnings drift behavior.")

    if np.isfinite(plan["atr_dollar_est"]):
        metrics_block = (
            f"Screener metrics:\n"
            f"- Last: {metrics_row.get('Last')}\n"
            f"- RVOL: {metrics_row.get('RVOL')}\n"
            f"- ATR%: {metrics_row.get('ATR%')} (≈ ${plan['atr_dollar_est']:.2f} ATR/day)\n"
        )
    else:
        metrics_block = (
            f"Screener metrics:\n"
            f"- Last: {metrics_row.get('Last')}\n"
            f"- RVOL: {metrics_row.get('RVOL')}\n"
            f"- ATR%: {metrics_row.get('ATR%')}\n"
        )
    
    # Add remaining fields
    metrics_block += (
        f"- RSI14: {metrics_row.get('RSI14')}\n"
        f"- Dist_to_52W_High%: {metrics_row.get('Dist_to_52W_High%')}\n"
    )
    
    # Add MA structure if available
    if "Above_MA20" in metrics_row and "Above_MA50" in metrics_row:
        ma20_val = metrics_row.get("MA20", np.nan)
        ma50_val = metrics_row.get("MA50", np.nan)
        above_ma20 = int(metrics_row.get("Above_MA20", 0))
        above_ma50 = int(metrics_row.get("Above_MA50", 0))
        ret20 = metrics_row.get("Ret20d%", np.nan)
        ma20_str = f"{ma20_val:.2f}" if np.isfinite(ma20_val) else "N/A"
        ma50_str = f"{ma50_val:.2f}" if np.isfinite(ma50_val) else "N/A"
        metrics_block += f"\n- MA20: {ma20_str}"
        metrics_block += f"\n- MA50: {ma50_str}"
        metrics_block += f"\n- Above_MA20: {above_ma20} (1=yes, 0=no)"
        metrics_block += f"\n- Above_MA50: {above_ma50} (1=yes, 0=no)"
        if np.isfinite(ret20):
            metrics_block += f"\n- Ret20d%: {ret20:.2f}%"
    
    metrics_block += (
        f"\n- Setup: {setup}\n"
        f"- Score: {metrics_row.get('Score')}\n"
        f"- Leveraged ETF: {leveraged_note}\n"
        f"- Dilution risk flag (headline scan): {flags['dilution_flag']}\n"
        f"- Catalyst tags (headline scan): {flags['catalyst_tags'] or 'None'}\n"
    )

    regime_block = (
        f"Market regime snapshot (best-effort):\n"
        f"- {regime_info.get('message','(not available)')}\n"
        f"- Regime Gate OK: {regime_info.get('ok', True)}\n"
    )

    completeness_block = (
        f"Catalyst completeness (penalize Unknowns):\n"
        f"- Completeness Score: {comp['completeness_score']}/100\n"
        f"- Earnings known: {comp['earnings_known']} (earnings={earnings_date})\n"
        f"- Recent headline count (14d): {comp['headline_count_recent']}\n"
        f"- Manual headlines: {comp['has_manual']}\n"
        + ("" if not comp["penalties"] else "- Penalties:\n  " + "\n  ".join([f"* {x}" for x in comp["penalties"]]) + "\n")
    )

    plan_block = (
        f"Standard trade plan template (by setup):\n"
        f"- {plan['entry_template']}\n"
        f"- {plan['stop_template']}\n"
        f"- {plan['tp_template']}\n"
        f"- {plan['size_template']}\n"
    )

    next_checks_block = "If data is missing, explicitly state Unknown and list checks:\n- " + "\n- ".join(missing_checks)

    prompt = f"""==============================
TICKER: {ticker}
==============================

Role: Momentum hedge fund analyst. You are running a probability audit (NOT a price predictor).
Objective: Decide if {ticker} has >60% probability of achieving +10% within 30 calendar days.

Hard constraints:
- You MUST penalize "Unknown" earnings + missing headlines (no assuming hidden catalysts).
- You MUST still score based on tape structure: RVOL, ATR%, RSI, distance to 52W high, setup type.
- You MUST output at most 2 BUY ratings across the entire batch (if you are given multiple tickers).
- If catalyst data is weak, default to WATCH/IGNORE even if technicals look good.
- If Dilution risk flag = 1 (headline scan), cap Verdict at WATCH unless there is a clear positive catalyst that outweighs it.
- For Breakouts, require structural confirmation: Above_MA20=1 and Above_MA50=1; otherwise downgrade Technical Alignment.

Inputs:
{regime_block}
Earnings (best-effort): {earnings_date}

{metrics_block}

{completeness_block}

Recent headlines:
{chr(10).join(headlines)}

{plan_block}

{next_checks_block}

Scoring rubric (0–100):
- Catalyst Immediacy (0–30)
- Narrative Velocity (0–25)
- Volatility Fit (0–20)
- Technical Alignment (0–25)

Output format (STRICT):
- Total Score: X/100
- Verdict: BUY / WATCH / IGNORE
- Setup: Breakout or Reversal
- 1-line Spark: what specifically could trigger +10%
- 1-line Trap: what invalidates the thesis
- Trade Plan: entry trigger, stop anchor, TP1/TP2, position size rule (risk 1–2%)
"""
    return prompt


# ============================================================
# Pipeline runner (shared by CLI + Local API)
# ============================================================
def run_pipeline(p: dict) -> dict:
    """Run the full screener pipeline and return artifacts in-memory.

    Returns dict with keys:
      - regime_info: dict
      - tickers_to_screen: list[str]
      - candidates: pd.DataFrame
      - news_df: pd.DataFrame
      - manual_headlines_df: pd.DataFrame
      - packets: list[str]
    """

    # Regime gate
    regime_info = {"ok": True, "message": "Regime gate disabled."}
    if p.get("enable_regime_gate", True):
        regime_info = check_regime(p)
        print(f"[REGIME] {regime_info.get('message','')}")
        if not regime_info.get("ok", True):
            action = p.get("regime_action", "WARN").upper()
            if action == "BLOCK":
                print("[REGIME] BLOCK enabled: returning no candidates.")
                return {
                    "regime_info": regime_info,
                    "tickers_to_screen": [],
                    "candidates": pd.DataFrame(),
                    "breakout_df": pd.DataFrame(),
                    "reversal_df": pd.DataFrame(),
                    "news_df": pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"]),
                    "manual_headlines_df": pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"]),
                    "packets": [],
                }
            print("[REGIME] WARN: continuing, but consider smaller size / fewer trades.")

    # Build dynamic attention pool from dynamic universe
    print("\n[1/4] Building dynamic attention pool from universe...")
    broad = get_dynamic_universe(p)
    
    if not broad:
        print("Failed to fetch universe. Cannot proceed without dynamic universe.")
        return {
            "regime_info": regime_info,
            "tickers_to_screen": [],
            "candidates": pd.DataFrame(),
            "breakout_df": pd.DataFrame(),
            "reversal_df": pd.DataFrame(),
            "news_df": pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"]),
            "manual_headlines_df": pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"]),
            "packets": [],
        }
    
    print(f"Fetched {len(broad)} universe tickers.")
    attention = build_attention_pool(broad, p)
    print(f"\nAttention pool: {len(attention)} tickers showing market attention today.")

    # Load manual picks from both files (manual_include_file and r2000_include_file)
    manual_picks = load_manual_include_tickers(p)
    
    # Show breakdown of what was loaded
    manual_file = p.get("manual_include_file", "")
    r2000_file = p.get("r2000_include_file", "")
    if manual_file and os.path.exists(manual_file):
        manual_tickers = load_tickers_from_file(manual_file)
        if manual_tickers:
            print(f"Manual include file ({manual_file}): {len(manual_tickers)} tickers")
    if r2000_file and os.path.exists(r2000_file):
        r2000_tickers = load_tickers_from_file(r2000_file)
        if r2000_tickers:
            print(f"R2000 file ({r2000_file}): {len(r2000_tickers)} tickers")

    mode = str(p.get("manual_include_mode", "ALWAYS")).upper()
    if mode == "ONLY_IF_IN_UNIVERSE":
        broad_set = set(broad)
        manual_picks = [t for t in manual_picks if t in broad_set]
        print(f"Manual picks after universe filter: {len(manual_picks)} tickers")

    tickers_to_screen = sorted(set(attention + manual_picks))

    print(f"Total tickers to screen (attention + manual): {len(tickers_to_screen)}")
    
    if not tickers_to_screen:
        print("No tickers to screen. Try lowering attention_rvol_min or attention_atr_pct_min.")
        return {
            "regime_info": regime_info,
            "tickers_to_screen": [],
            "candidates": pd.DataFrame(),
            "breakout_df": pd.DataFrame(),
            "reversal_df": pd.DataFrame(),
            "news_df": pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"]),
            "manual_headlines_df": pd.DataFrame(columns=["Ticker", "Date", "Source", "Headline"]),
            "packets": [],
        }
    
    print(f"Sample (first 10 of {len(tickers_to_screen)}): {', '.join(tickers_to_screen[:10])}")

    print(f"\n[2/5] Applying quality filters to {len(tickers_to_screen)} tickers...")
    screened = screen_universe(tickers_to_screen, p)
    breakout_df = screened["breakout_df"]
    reversal_df = screened["reversal_df"]
    candidates = screened["combined_df"]

    # Pull recent news headlines (yfinance)
    news_df = pd.DataFrame(columns=["Ticker","published_utc","published_local","title","publisher","link","type"])
    if not candidates.empty:
        tickers = candidates["Ticker"].tolist()
        news_df = fetch_news_for_tickers(tickers, max_items=p["news_max_items"], throttle_sec=p["throttle_sec"])

        # Clean news_df: remove rows with empty/null titles (yfinance sometimes returns empty rows)
        if not news_df.empty:
            news_df = news_df.dropna(subset=["title"])
            news_df = news_df[news_df["title"].astype(str).str.strip().ne("")]

    # Load manual headlines if available
    manual_headlines_df = load_manual_headlines("manual_headlines.csv")
    if not manual_headlines_df.empty:
        manual_count = len(manual_headlines_df)
        print(f"Loaded {manual_count} manual headline(s) from manual_headlines.csv")

    # Build LLM packets
    packets: list[str] = []
    if not candidates.empty:
        for _, row in candidates.iterrows():
            t = row["Ticker"]
            packets.append(build_llm_packet(
                ticker=t,
                metrics_row=row,
                news_df=news_df,
                max_headlines=p["packet_headlines"],
                regime_info=regime_info,
                manual_headlines_df=manual_headlines_df,
            ))

    return {
        "regime_info": regime_info,
        "tickers_to_screen": tickers_to_screen,
        "candidates": candidates,
        "breakout_df": breakout_df,
        "reversal_df": reversal_df,
        "news_df": news_df,
        "manual_headlines_df": manual_headlines_df,
        "packets": packets,
    }

# ============================================================
# Main
# ============================================================
def main() -> int:
    p = PARAMS

    # Create date-based output folder
    today = dt.date.today()
    date_str = today.strftime("%Y-%m-%d")
    output_dir = os.path.join(os.getcwd(), date_str)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")

    # Run screener pipeline (reusable for CLI + API)
    pipeline = run_pipeline(p)
    regime_info = pipeline["regime_info"]
    candidates = pipeline["candidates"]
    breakout_df = pipeline.get("breakout_df", pd.DataFrame())
    reversal_df = pipeline.get("reversal_df", pd.DataFrame())
    news_df = pipeline["news_df"]
    manual_headlines_df = pipeline["manual_headlines_df"]

    if candidates.empty:
        print("No candidates matched filters. Try lowering rvol_min or atr_pct_min.")
        return 0

    # Optionally: separate leveraged ETFs so they don't dominate
    # Here we keep them, but you can filter them out by uncommenting next line:
    # candidates = candidates[~candidates["Is_Leveraged_ETF"]].reset_index(drop=True)

    print(f"Found {len(candidates)} total candidates.")
    
    # Write separate breakout and reversal CSVs with date in filename
    if not breakout_df.empty:
        breakout_file = os.path.join(output_dir, f"30d_breakout_candidates_{date_str}.csv")
        breakout_df.to_csv(breakout_file, index=False)
        print(f"Saved: {breakout_file} ({len(breakout_df)} candidates)")
    
    if not reversal_df.empty:
        reversal_file = os.path.join(output_dir, f"30d_reversal_candidates_{date_str}.csv")
        reversal_df.to_csv(reversal_file, index=False)
        print(f"Saved: {reversal_file} ({len(reversal_df)} candidates)")
    
    # Write combined CSV with date in filename
    candidates_file = os.path.join(output_dir, f"30d_momentum_candidates_{date_str}.csv")
    candidates.to_csv(candidates_file, index=False)
    print(f"Saved: {candidates_file} ({len(candidates)} total candidates)")

    print("\n[3/5] Pulling recent news headlines (yfinance)...")
    # Reuse pipeline output (already fetched + cleaned)
    news_df = pipeline["news_df"]
    news_file = os.path.join(output_dir, f"news_dump_{date_str}.csv")
    news_df.to_csv(news_file, index=False)
    if news_df.empty:
        print(f"Saved: {news_file} (empty - no news found for candidates)")
    else:
        print(f"Saved: {news_file} ({len(news_df)} headline(s))")

    print("\n[4/5] Building LLM packets (with earnings-date attempt)...")
    packets = pipeline["packets"]

    print(f"\n[5/5] Writing packets TXT: llm_packets_{date_str}.txt")
    packets_file = os.path.join(output_dir, f"llm_packets_{date_str}.txt")
    with open(packets_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(packets))

    print(f"\nDone. All outputs saved to: {output_dir}/")
    print(f" - 30d_momentum_candidates_{date_str}.csv")
    print(f" - 30d_breakout_candidates_{date_str}.csv")
    print(f" - 30d_reversal_candidates_{date_str}.csv")
    print(f" - news_dump_{date_str}.csv")
    print(f" - llm_packets_{date_str}.txt")
    
    # Automatically run LLM analysis if candidates were found
    if not candidates.empty:
        print(f"\n[6/6] Starting automatic LLM analysis...")
        import subprocess
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            analyze_script = os.path.join(script_dir, "analyze_both.py")
            result = subprocess.run(
                [sys.executable, analyze_script, "--date", date_str],
                cwd=script_dir,
                check=False  # Don't fail if analysis fails
            )
            if result.returncode == 0:
                print("LLM analysis completed successfully!")
            else:
                print(f"LLM analysis completed with warnings (exit code: {result.returncode})")
        except Exception as e:
            print(f"Warning: Could not run automatic LLM analysis: {e}")
            print(f"You can manually run: python analyze_both.py --date {date_str}")
    
    print("\nTip: open llm_packets file and paste one ticker block into ChatGPT to score + plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

