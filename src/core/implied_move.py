"""
Options-Implied Move Calculator

Uses ATM straddle prices to determine what the market expects.
This replaces subjective LLM catalyst scoring with objective market data.

The implied move is derived from:
- ATM (at-the-money) call price + put price = straddle price
- Straddle price / spot price = expected move (approximately)

Higher implied move = market expects bigger move = higher catalyst potential
"""

from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from src.utils.time import utc_now


def get_implied_move(
    ticker: str,
    target_days: int = 7,
    polygon_api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calculate market-implied expected move from options prices.
    
    Uses ATM straddle pricing to estimate expected volatility.
    
    Args:
        ticker: Stock ticker symbol
        target_days: Days until target (default 7 for weekly horizon)
        polygon_api_key: Polygon API key (uses env var if not provided)
    
    Returns:
        Dict with:
            - implied_move_pct: Expected move as percentage
            - straddle_price: ATM straddle cost
            - spot_price: Current stock price
            - atm_strike: ATM strike price used
            - expiration: Option expiration date used
            - days_to_expiry: Days until expiration
            - call_price: ATM call price
            - put_price: ATM put price
            - confidence: Data quality score (0-1)
        Or None if data unavailable
    """
    api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.debug(f"No Polygon API key for implied move calculation")
        return None
    
    try:
        import requests
        
        # ═══════════════════════════════════════════════════════════════════════════
        # Step 1: Get current stock price
        # ═══════════════════════════════════════════════════════════════════════════
        spot_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
        spot_resp = requests.get(spot_url, params={"apiKey": api_key}, timeout=10)
        spot_data = spot_resp.json()
        
        if spot_data.get("status") != "OK" or not spot_data.get("results"):
            logger.debug(f"Could not get spot price for {ticker}")
            return None
        
        spot_price = float(spot_data["results"][0]["c"])
        
        # ═══════════════════════════════════════════════════════════════════════════
        # Step 2: Find nearest option expiration to target_days
        # ═══════════════════════════════════════════════════════════════════════════
        target_date = utc_now() + timedelta(days=target_days)
        
        # Get options chain
        chain_url = f"https://api.polygon.io/v3/reference/options/contracts"
        chain_params = {
            "underlying_ticker": ticker,
            "expiration_date.gte": utc_now().strftime("%Y-%m-%d"),
            "expiration_date.lte": (utc_now() + timedelta(days=target_days + 14)).strftime("%Y-%m-%d"),
            "limit": 250,
            "apiKey": api_key,
        }
        chain_resp = requests.get(chain_url, params=chain_params, timeout=10)
        chain_data = chain_resp.json()
        
        if chain_data.get("status") != "OK" or not chain_data.get("results"):
            logger.debug(f"Could not get options chain for {ticker}")
            return None
        
        contracts = chain_data["results"]
        
        # Find unique expirations
        expirations = sorted(set(c["expiration_date"] for c in contracts))
        if not expirations:
            logger.debug(f"No expirations found for {ticker}")
            return None
        
        # Pick expiration closest to target_days
        def _expiry_dt(expiry_str: str) -> datetime:
            return datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        best_expiry = min(expirations, key=lambda x: abs(
            (_expiry_dt(x) - target_date).days
        ))
        
        days_to_expiry = (_expiry_dt(best_expiry) - utc_now()).days
        
        # ═══════════════════════════════════════════════════════════════════════════
        # Step 3: Find ATM strike (closest to spot)
        # ═══════════════════════════════════════════════════════════════════════════
        expiry_contracts = [c for c in contracts if c["expiration_date"] == best_expiry]
        
        strikes = sorted(set(c["strike_price"] for c in expiry_contracts))
        if not strikes:
            logger.debug(f"No strikes found for {ticker} expiry {best_expiry}")
            return None
        
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        
        # Get ATM call and put contracts
        atm_call = next((c for c in expiry_contracts 
                        if c["strike_price"] == atm_strike and c["contract_type"] == "call"), None)
        atm_put = next((c for c in expiry_contracts 
                       if c["strike_price"] == atm_strike and c["contract_type"] == "put"), None)
        
        if not atm_call or not atm_put:
            logger.debug(f"Could not find ATM call/put for {ticker} at strike {atm_strike}")
            return None
        
        # ═══════════════════════════════════════════════════════════════════════════
        # Step 4: Get option prices (last trade)
        # ═══════════════════════════════════════════════════════════════════════════
        def get_option_price(contract_ticker: str) -> Optional[float]:
            """Get last trade price for an option contract."""
            price_url = f"https://api.polygon.io/v2/last/trade/{contract_ticker}"
            try:
                price_resp = requests.get(price_url, params={"apiKey": api_key}, timeout=10)
                price_data = price_resp.json()
                
                if price_data.get("status") == "OK" and price_data.get("results"):
                    return float(price_data["results"]["p"])
            except Exception as e:
                logger.debug(f"Error getting option price for {contract_ticker}: {e}")
            return None
        
        call_price = get_option_price(atm_call["ticker"])
        put_price = get_option_price(atm_put["ticker"])
        
        # If last trade not available, try snapshot for quotes
        if call_price is None or put_price is None:
            try:
                snapshot_url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
                snapshot_params = {
                    "strike_price": atm_strike,
                    "expiration_date": best_expiry,
                    "apiKey": api_key,
                }
                snap_resp = requests.get(snapshot_url, params=snapshot_params, timeout=10)
                snap_data = snap_resp.json()
                
                if snap_data.get("status") == "OK" and snap_data.get("results"):
                    for opt in snap_data["results"]:
                        if opt.get("details", {}).get("contract_type") == "call" and call_price is None:
                            quote = opt.get("last_quote", {})
                            if quote.get("bid") and quote.get("ask"):
                                call_price = (float(quote["bid"]) + float(quote["ask"])) / 2
                        elif opt.get("details", {}).get("contract_type") == "put" and put_price is None:
                            quote = opt.get("last_quote", {})
                            if quote.get("bid") and quote.get("ask"):
                                put_price = (float(quote["bid"]) + float(quote["ask"])) / 2
            except Exception as e:
                logger.debug(f"Error getting snapshot for {ticker}: {e}")
        
        if call_price is None or put_price is None:
            logger.debug(f"Could not get call/put prices for {ticker}")
            return None
        
        # ═══════════════════════════════════════════════════════════════════════════
        # Step 5: Calculate implied move
        # ═══════════════════════════════════════════════════════════════════════════
        straddle_price = call_price + put_price
        implied_move_pct = (straddle_price / spot_price) * 100
        
        # Adjust for time to expiration if significantly different from target
        # (rough adjustment - proper would use sqrt(time) but this is approximate)
        if days_to_expiry > 0 and days_to_expiry != target_days:
            time_adjustment = (target_days / days_to_expiry) ** 0.5
            implied_move_pct = implied_move_pct * time_adjustment
        
        # Confidence based on data quality
        confidence = 0.9
        if days_to_expiry > target_days + 5:
            confidence = 0.7  # Expiration is far from target
        elif days_to_expiry < target_days - 3:
            confidence = 0.75  # Expiration is before target
        
        return {
            "implied_move_pct": round(implied_move_pct, 2),
            "straddle_price": round(straddle_price, 2),
            "spot_price": round(spot_price, 2),
            "atm_strike": atm_strike,
            "expiration": best_expiry,
            "days_to_expiry": days_to_expiry,
            "call_price": round(call_price, 2),
            "put_price": round(put_price, 2),
            "confidence": confidence,
            "call_ticker": atm_call["ticker"],
            "put_ticker": atm_put["ticker"],
        }
        
    except Exception as e:
        logger.warning(f"Failed to get implied move for {ticker}: {e}")
        return None


def compute_catalyst_score_from_implied_move(
    implied_move: Optional[Dict[str, Any]],
    target_move_pct: float = 10.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Convert implied move to catalyst score (0-10).
    
    The logic:
    - If market expects MORE than target move, score high (market agrees with thesis)
    - If market expects LESS than target move, score lower (contrarian bet)
    - This is objective, market-based scoring vs. subjective LLM guessing
    
    Args:
        implied_move: Output from get_implied_move()
        target_move_pct: Target move percentage (default 10% for weekly scanner)
    
    Returns:
        Tuple of (score 0-10, evidence_dict)
    """
    evidence = {
        "implied_move_pct": None,
        "target_move_pct": target_move_pct,
        "market_expects_target": None,
        "ratio_to_target": None,
        "interpretation": None,
        "data_source": "options_implied",
    }
    
    if implied_move is None:
        return 3.0, {**evidence, "data_gap": "Options data unavailable", "score_capped": True}
    
    impl_move = implied_move["implied_move_pct"]
    ratio = impl_move / target_move_pct if target_move_pct > 0 else 0
    
    evidence["implied_move_pct"] = impl_move
    evidence["ratio_to_target"] = round(ratio, 2)
    evidence["expiration"] = implied_move.get("expiration")
    evidence["straddle_price"] = implied_move.get("straddle_price")
    evidence["spot_price"] = implied_move.get("spot_price")
    evidence["days_to_expiry"] = implied_move.get("days_to_expiry")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORING LOGIC
    # ═══════════════════════════════════════════════════════════════════════════
    
    if ratio >= 1.5:
        # Market expects very big move (150%+ of target)
        score = 9.0
        evidence["market_expects_target"] = True
        evidence["interpretation"] = "Market expects significant move - high conviction setup"
    
    elif ratio >= 1.2:
        # Market expects move exceeding target
        score = 8.0
        evidence["market_expects_target"] = True
        evidence["interpretation"] = "Market expects move above target - favorable setup"
    
    elif ratio >= 1.0:
        # Market expects roughly target move
        score = 7.0
        evidence["market_expects_target"] = True
        evidence["interpretation"] = "Market expects target achievable"
    
    elif ratio >= 0.8:
        # Market expects slightly less than target
        score = 5.5
        evidence["market_expects_target"] = False
        evidence["interpretation"] = "Market expects smaller move - target is stretch"
    
    elif ratio >= 0.6:
        # Market expects much less than target
        score = 4.0
        evidence["market_expects_target"] = False
        evidence["interpretation"] = "Contrarian bet - market doesn't expect target"
    
    else:
        # Market expects very low volatility
        score = 2.5
        evidence["market_expects_target"] = False
        evidence["interpretation"] = "Low implied vol - unlikely to hit target without catalyst"
    
    # Adjust for data confidence
    confidence = implied_move.get("confidence", 0.7)
    if confidence < 0.8:
        original_score = score
        score = score * 0.9  # Slight penalty for less reliable data
        evidence["confidence_adjustment"] = round(score - original_score, 2)
    
    return round(score, 1), evidence


def get_implied_move_batch(
    tickers: list[str],
    target_days: int = 7,
    polygon_api_key: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Get implied moves for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        target_days: Days until target
        polygon_api_key: Polygon API key
    
    Returns:
        Dict mapping ticker -> implied_move_data (or None if unavailable)
    """
    results = {}
    
    for ticker in tickers:
        try:
            result = get_implied_move(ticker, target_days, polygon_api_key)
            results[ticker] = result
        except Exception as e:
            logger.warning(f"Error getting implied move for {ticker}: {e}")
            results[ticker] = None
    
    return results


def format_implied_move_summary(implied_move: Dict[str, Any]) -> str:
    """
    Format implied move data for display.
    
    Args:
        implied_move: Output from get_implied_move()
    
    Returns:
        Formatted string summary
    """
    if implied_move is None:
        return "Implied move: N/A (options data unavailable)"
    
    return (
        f"Implied move: {implied_move['implied_move_pct']:.1f}% "
        f"(straddle ${implied_move['straddle_price']:.2f} / "
        f"spot ${implied_move['spot_price']:.2f}) "
        f"exp {implied_move['expiration']} "
        f"({implied_move['days_to_expiry']}d)"
    )
