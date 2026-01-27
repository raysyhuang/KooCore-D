"""
Options Activity Module

Fetches unusual options activity to enhance the 4-factor scoring model.
Supports multiple data providers with fallback chain.

Priority order:
1. Polygon.io (if POLYGON_API_KEY set) - Full options chain with volume, OI, IV
2. Yahoo Finance (fallback) - Basic options data
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, TypedDict
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)

from src.utils.time import utc_today


class OptionsEvidence(TypedDict):
    """Options evidence structure for scoring."""
    call_put_ratio: Optional[float]
    call_volume: Optional[int]
    put_volume: Optional[int]
    call_oi: Optional[int]
    put_oi: Optional[int]
    unusual_volume_multiple: Optional[float]
    largest_bullish_premium_usd: Optional[float]
    iv_rank: Optional[float]
    implied_volatility: Optional[float]
    notable_contracts: list[dict]
    data_source: str


@dataclass
class OptionsScore:
    """Options activity score result."""
    score: float  # 0-10 scale
    evidence: OptionsEvidence
    data_gaps: list[str] = field(default_factory=list)
    cap_applied: Optional[float] = None


def fetch_options_snapshot_polygon(
    ticker: str, 
    api_key: Optional[str] = None,
    limit: int = 250,
) -> Optional[dict]:
    """
    Fetch comprehensive options snapshot from Polygon.io.
    
    This uses the Options Snapshot endpoint which provides:
    - All active contracts with current greeks
    - Day's volume and open interest
    - Implied volatility
    - Last quote/trade data
    
    Requires POLYGON_API_KEY environment variable.
    Works with Basic plan and above.
    """
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.debug("No POLYGON_API_KEY found for options data")
        return None
    
    try:
        # Use the options chain snapshot endpoint
        # https://polygon.io/docs/options/get_v3_snapshot_options__underlyingasset
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        params = {
            "limit": limit,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 403:
            logger.debug(f"Polygon options API requires upgraded plan for {ticker}")
            # Fall back to contracts endpoint (available on free tier)
            return fetch_options_contracts_polygon(ticker, api_key)
        
        if response.status_code != 200:
            logger.debug(f"Polygon options snapshot failed: {response.status_code}")
            return fetch_options_contracts_polygon(ticker, api_key)
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            logger.debug(f"No options snapshot data for {ticker}")
            return fetch_options_contracts_polygon(ticker, api_key)
        
        # Aggregate data from all contracts
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        iv_values = []
        notable_contracts = []
        
        for contract in results:
            details = contract.get("details", {})
            day = contract.get("day", {})
            greeks = contract.get("greeks", {})
            
            contract_type = details.get("contract_type", "").lower()
            volume = day.get("volume", 0) or 0
            open_interest = contract.get("open_interest", 0) or 0
            # IV is at top level in Polygon response, not in greeks
            iv = contract.get("implied_volatility")
            
            if contract_type == "call":
                total_call_volume += volume
                total_call_oi += open_interest
            elif contract_type == "put":
                total_put_volume += volume
                total_put_oi += open_interest
            
            # Track IV values for averaging (filter out extreme values)
            # IV should be between 5% and 200% for meaningful contracts
            if iv is not None and 0.05 <= iv <= 2.0:
                iv_values.append(iv)
            
            # Track notable high-volume contracts
            if volume >= 1000:
                strike = details.get("strike_price")
                expiration = details.get("expiration_date")
                notable_contracts.append({
                    "ticker": details.get("ticker"),
                    "type": contract_type,
                    "strike": strike,
                    "expiration": expiration,
                    "volume": volume,
                    "open_interest": open_interest,
                    "iv": iv,
                })
        
        # Calculate call/put ratio
        total_volume = total_call_volume + total_put_volume
        call_put_ratio = None
        if total_put_volume > 0:
            call_put_ratio = total_call_volume / total_put_volume
        elif total_call_volume > 0:
            call_put_ratio = float('inf')
        
        # Calculate average IV
        avg_iv = sum(iv_values) / len(iv_values) if iv_values else None
        
        # Sort notable contracts by volume
        notable_contracts.sort(key=lambda x: x.get("volume", 0), reverse=True)
        
        return {
            "call_volume": total_call_volume,
            "put_volume": total_put_volume,
            "total_volume": total_volume,
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "total_oi": total_call_oi + total_put_oi,
            "call_put_ratio": call_put_ratio,
            "implied_volatility": avg_iv,
            "contracts_analyzed": len(results),
            "notable_contracts": notable_contracts[:10],  # Top 10
            "source": "polygon_snapshot"
        }
        
    except Exception as e:
        logger.debug(f"Polygon options snapshot failed for {ticker}: {e}")
        # Fall back to contracts endpoint
        return fetch_options_contracts_polygon(ticker, api_key)


def fetch_options_contracts_polygon(
    ticker: str, 
    api_key: Optional[str] = None
) -> Optional[dict]:
    """
    Fetch basic options contract info from Polygon.io.
    
    This is the fallback that works on free tier.
    Only provides contract counts, not volume/OI.
    """
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        return None
    
    try:
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "expired": "false",
            "limit": 250,
            "apiKey": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        contracts = data.get("results", [])
        
        if not contracts:
            return None
        
        # Count calls vs puts
        calls = [c for c in contracts if c.get("contract_type") == "call"]
        puts = [c for c in contracts if c.get("contract_type") == "put"]
        
        call_put_ratio = len(calls) / len(puts) if puts else None
        
        # Get near-term expirations (within 30 days)
        today = utc_today()
        near_term_contracts = []
        
        for c in contracts:
            exp_str = c.get("expiration_date")
            if exp_str:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    if (exp_date - today).days <= 30:
                        near_term_contracts.append(c)
                except (ValueError, TypeError):
                    pass  # Skip invalid expiration dates
        
        return {
            "total_contracts": len(contracts),
            "call_contracts": len(calls),
            "put_contracts": len(puts),
            "call_put_ratio": call_put_ratio,
            "near_term_contracts": len(near_term_contracts),
            "source": "polygon_contracts"
        }
        
    except Exception as e:
        logger.debug(f"Polygon contracts fetch failed for {ticker}: {e}")
        return None


def fetch_options_activity_yahoo(ticker: str) -> Optional[dict]:
    """
    Fetch options data from Yahoo Finance.
    
    This is the final fallback when Polygon is unavailable.
    Provides volume and open interest for near-term options.
    """
    try:
        import yfinance as yf
        
        tk = yf.Ticker(ticker)
        
        # Get all available expiration dates
        expirations = tk.options
        if not expirations:
            return None
        
        # Focus on near-term options (next 2 expirations)
        near_term = expirations[:2] if len(expirations) >= 2 else expirations
        
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        iv_values = []
        notable_contracts = []
        
        for exp in near_term:
            try:
                opt_chain = tk.option_chain(exp)
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                if not calls.empty:
                    call_vol = calls['volume'].fillna(0).sum()
                    call_oi = calls['openInterest'].fillna(0).sum()
                    total_call_volume += int(call_vol)
                    total_call_oi += int(call_oi)
                    
                    # Get IV from calls
                    if 'impliedVolatility' in calls.columns:
                        ivs = calls['impliedVolatility'].dropna()
                        iv_values.extend(ivs.tolist())
                    
                    # Find notable contracts
                    for _, row in calls[calls['volume'] >= 500].iterrows():
                        notable_contracts.append({
                            "type": "call",
                            "strike": row.get('strike'),
                            "expiration": exp,
                            "volume": int(row.get('volume', 0)),
                            "open_interest": int(row.get('openInterest', 0)),
                            "iv": row.get('impliedVolatility'),
                        })
                
                if not puts.empty:
                    put_vol = puts['volume'].fillna(0).sum()
                    put_oi = puts['openInterest'].fillna(0).sum()
                    total_put_volume += int(put_vol)
                    total_put_oi += int(put_oi)
                    
                    # Get IV from puts
                    if 'impliedVolatility' in puts.columns:
                        ivs = puts['impliedVolatility'].dropna()
                        iv_values.extend(ivs.tolist())
                        
            except Exception as e:
                logger.debug(f"Yahoo option chain error for {ticker} {exp}: {e}")
                continue
        
        if total_call_volume == 0 and total_put_volume == 0:
            return None
        
        call_put_ratio = None
        if total_put_volume > 0:
            call_put_ratio = total_call_volume / total_put_volume
        elif total_call_volume > 0:
            call_put_ratio = float('inf')
        
        avg_iv = sum(iv_values) / len(iv_values) if iv_values else None
        
        # Sort notable contracts by volume
        notable_contracts.sort(key=lambda x: x.get("volume", 0), reverse=True)
        
        return {
            "call_volume": total_call_volume,
            "put_volume": total_put_volume,
            "total_volume": total_call_volume + total_put_volume,
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "call_put_ratio": call_put_ratio,
            "implied_volatility": avg_iv,
            "notable_contracts": notable_contracts[:10],
            "source": "yahoo"
        }
        
    except Exception as e:
        logger.debug(f"Yahoo options fetch failed for {ticker}: {e}")
        return None


def compute_options_score(ticker: str, api_key: Optional[str] = None) -> OptionsScore:
    """
    Compute options activity score (0-10) for a ticker.
    
    Data source priority:
    1. Polygon.io snapshot (requires POLYGON_API_KEY)
    2. Polygon.io contracts (free tier fallback)
    3. Yahoo Finance (final fallback)
    
    Scoring rubric (total 10 points):
    - Call/Put Volume Ratio (0-3 points):
      +3.0 if ratio >= 3.0 (very bullish flow)
      +2.0 if ratio >= 2.0 (bullish flow)
      +1.0 if ratio >= 1.5 (moderately bullish)
      
    - Total Volume vs OI (0-2 points):
      +2.0 if volume/OI >= 0.5 (high activity day)
      +1.0 if volume/OI >= 0.25
      
    - Implied Volatility Rank (0-2 points):
      +2.0 if IV is 40-70% (elevated but not extreme)
      +1.0 if IV is 30-40% or 70-80%
      
    - Notable Contracts (0-3 points):
      +1.5 per high-volume bullish contract (up to 3 points)
    
    If no options data available, caps score at 3.0.
    """
    data_gaps = []
    evidence: OptionsEvidence = {
        "call_put_ratio": None,
        "call_volume": None,
        "put_volume": None,
        "call_oi": None,
        "put_oi": None,
        "unusual_volume_multiple": None,
        "largest_bullish_premium_usd": None,
        "iv_rank": None,
        "implied_volatility": None,
        "notable_contracts": [],
        "data_source": "none"
    }
    
    score = 0.0
    cap_applied = None
    
    # Try Polygon first (best data with paid plan - full snapshot with greeks)
    options_data = fetch_options_snapshot_polygon(ticker, api_key)
    
    # Fall back to Yahoo if Polygon failed
    if not options_data:
        options_data = fetch_options_activity_yahoo(ticker)
    
    if options_data:
        # Populate evidence
        evidence["data_source"] = options_data.get("source", "unknown")
        evidence["call_put_ratio"] = options_data.get("call_put_ratio")
        evidence["call_volume"] = options_data.get("call_volume")
        evidence["put_volume"] = options_data.get("put_volume")
        evidence["call_oi"] = options_data.get("call_oi")
        evidence["put_oi"] = options_data.get("put_oi")
        evidence["implied_volatility"] = options_data.get("implied_volatility")
        evidence["notable_contracts"] = options_data.get("notable_contracts", [])
        
        # 1. Score Call/Put Volume Ratio (0-3 points)
        cp_ratio = options_data.get("call_put_ratio")
        if cp_ratio is not None and cp_ratio != float('inf'):
            if cp_ratio >= 3.0:
                score += 3.0
            elif cp_ratio >= 2.0:
                score += 2.0
            elif cp_ratio >= 1.5:
                score += 1.0
            elif cp_ratio <= 0.5:
                # Very bearish flow - slight penalty
                score -= 0.5
        
        # 2. Score Volume vs Open Interest (0-2 points)
        total_volume = options_data.get("total_volume", 0) or options_data.get("call_volume", 0) + options_data.get("put_volume", 0)
        total_oi = (options_data.get("call_oi", 0) or 0) + (options_data.get("put_oi", 0) or 0)
        
        if total_oi > 0 and total_volume > 0:
            volume_oi_ratio = total_volume / total_oi
            evidence["unusual_volume_multiple"] = round(volume_oi_ratio, 2)
            
            if volume_oi_ratio >= 0.5:
                score += 2.0
            elif volume_oi_ratio >= 0.25:
                score += 1.0
        
        # 3. Score Implied Volatility (0-2 points)
        iv = options_data.get("implied_volatility")
        if iv is not None:
            # Convert to percentage if needed
            iv_pct = iv * 100 if iv < 1 else iv
            
            if 40 <= iv_pct <= 70:
                score += 2.0
            elif 30 <= iv_pct < 40 or 70 < iv_pct <= 80:
                score += 1.0
        else:
            data_gaps.append("Implied volatility data unavailable")
        
        # 4. Score Notable Contracts (0-3 points)
        notable = options_data.get("notable_contracts", [])
        bullish_notable = [c for c in notable if c.get("type") == "call" and c.get("volume", 0) >= 1000]
        
        if len(bullish_notable) >= 3:
            score += 3.0
        elif len(bullish_notable) >= 2:
            score += 2.0
        elif len(bullish_notable) >= 1:
            score += 1.0
        
        # Log what we found
        logger.debug(
            f"Options data for {ticker}: source={evidence['data_source']}, "
            f"C/P ratio={cp_ratio}, volume={total_volume}, "
            f"IV={iv}, score={score:.1f}"
        )
        
    else:
        # No options data available at all
        data_gaps.append("No options data available from any source; score capped at 3.0")
        cap_applied = 3.0
        logger.debug(f"No options data available for {ticker}")
    
    # Apply cap if data is missing
    if cap_applied is not None:
        score = min(score, cap_applied)
    
    # Ensure score is within bounds
    score = max(0.0, min(10.0, score))
    
    return OptionsScore(
        score=round(score, 2),
        evidence=evidence,
        data_gaps=data_gaps,
        cap_applied=cap_applied
    )


def get_iv_rank(ticker: str, api_key: Optional[str] = None) -> Optional[float]:
    """
    Get IV Rank (Implied Volatility Rank) for a ticker.
    
    IV Rank = (Current IV - 52W Low IV) / (52W High IV - 52W Low IV) * 100
    
    Returns value 0-100, or None if unavailable.
    """
    # First try to get current IV from options
    options_data = fetch_options_snapshot_polygon(ticker, api_key)
    if not options_data:
        options_data = fetch_options_activity_yahoo(ticker)
    
    current_iv = None
    if options_data:
        current_iv = options_data.get("implied_volatility")
    
    if current_iv is None:
        # Fall back to historical volatility proxy
        try:
            import yfinance as yf
            
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1y")
            
            if hist.empty:
                return None
            
            # Use historical volatility as proxy
            returns = hist['Close'].pct_change().dropna()
            current_vol = returns.tail(20).std() * (252 ** 0.5)  # Annualized
            
            rolling_vol = returns.rolling(20).std() * (252 ** 0.5)
            vol_52w_high = rolling_vol.max()
            vol_52w_low = rolling_vol.min()
            
            if vol_52w_high == vol_52w_low:
                return 50.0
            
            iv_rank = (current_vol - vol_52w_low) / (vol_52w_high - vol_52w_low) * 100
            return round(max(0, min(100, iv_rank)), 1)
            
        except Exception as e:
            logger.debug(f"IV Rank calculation failed for {ticker}: {e}")
            return None
    
    # TODO: Need historical IV data to compute true IV Rank
    # For now, return a proxy based on current IV level
    # 30% IV is roughly 50th percentile for most stocks
    iv_pct = current_iv * 100 if current_iv < 1 else current_iv
    estimated_rank = min(100, max(0, (iv_pct - 15) * 2))
    
    return round(estimated_rank, 1)


# Convenience function
def get_options_summary(ticker: str) -> dict:
    """
    Get a human-readable options summary for a ticker.
    
    Useful for debugging and LLM context.
    """
    result = compute_options_score(ticker)
    
    summary = {
        "ticker": ticker,
        "options_score": result.score,
        "data_source": result.evidence["data_source"],
        "call_put_ratio": result.evidence["call_put_ratio"],
        "call_volume": result.evidence["call_volume"],
        "put_volume": result.evidence["put_volume"],
        "implied_volatility": result.evidence["implied_volatility"],
        "data_gaps": result.data_gaps,
    }
    
    # Add interpretation
    cp_ratio = result.evidence["call_put_ratio"]
    if cp_ratio is not None and cp_ratio != float('inf'):
        if cp_ratio >= 2.0:
            summary["interpretation"] = "Bullish options flow"
        elif cp_ratio >= 1.0:
            summary["interpretation"] = "Neutral to slightly bullish"
        elif cp_ratio >= 0.5:
            summary["interpretation"] = "Neutral to slightly bearish"
        else:
            summary["interpretation"] = "Bearish options flow"
    else:
        summary["interpretation"] = "Insufficient data"
    
    return summary
