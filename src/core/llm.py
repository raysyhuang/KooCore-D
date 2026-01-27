"""
LLM Integration

Functions for calling LLM APIs and ranking candidates.
Includes bull/bear debate system and memory integration.
"""

from __future__ import annotations
import os
import json
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import debate and memory modules (optional features)
try:
    from src.core.debate import run_batch_debate, format_debate_summary, debate_result_to_dict
    DEBATE_AVAILABLE = True
except ImportError:
    DEBATE_AVAILABLE = False
    logger.debug("Debate module not available")

try:
    from src.core.memory import get_trading_memory, enrich_packet_with_memory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logger.debug("Memory module not available")


def build_weekly_scanner_prompt(packets: list[dict]) -> str:
    """
    Build the LLM prompt for Weekly Momentum Scanner ranking.
    
    Args:
        packets: List of candidate packets
    
    Returns:
        Complete prompt string
    """
    prompt = """# **🎯 Weekly Momentum Scanner (NASDAQ + NYSE) — Top 5 Candidates for ≥10% Move in Next 7 Trading Days**

## **Mission**

Identify **Top 5 U.S.-listed stocks** (NASDAQ/NYSE; any market cap) that have the **highest probability of a ≥10% price move upward within the next 7 trading days**, using a **4-factor model** with **equal weights (25% each)**:
1. Technical Momentum
2. Upcoming Catalyst
3. Options Activity
4. Sentiment Momentum

This is a probabilistic screen, not a guarantee. Prefer **evidence-backed**, **time-stamped**, **source-cited** reasoning.

---
## **0) Execution Rules (anti-hallucination + reproducibility)**

1. **Data freshness targets** (best-effort):
   * Price/volume/technicals: ≤ 15 min delay
   * Options: ≤ 15 min delay
   * Social sentiment: ≤ 1 hour delay
   * Catalyst/news: must be **source-cited** with timestamp (use search tool citations when available).
2. **No guessing**: if any key metric is unavailable, set it to null, explain in data_gaps, and **cap** the relevant factor score (see scoring rules).
3. **Cross-check** catalysts/news with **≥2 independent sources** when possible. If sources conflict, downgrade the Catalyst score and note it.
4. **Output must be valid JSON only** (no markdown, no commentary).
5. Include run_timestamp_utc and asof_* timestamps per data type so results can be reproduced.

---
## **1) Universe & Liquidity Gate**

Universe: **All common stocks + ADRs listed on NASDAQ/NYSE**.
**Hard liquidity filters (must pass):** 
* avg_dollar_volume_20d >= 50_000_000 (USD)
* price >= 2.00 (avoid ultra-illiquid microcaps unless explicitly requested)
**Exclusions (must exclude unless user overrides):** 
* Price up **> 15%** in the last **5** trading days (avoid already-exploded names)
* Earnings **> 10 calendar days out** **AND** no other major catalyst within 7 trading days
* Active trading halts / delisting notices / "going concern" shock without a clear, tradable setup (flag if detected)

---
## **2) Scoring Model (0–10 each factor, then 25% weighted)**

### **Composite Score**
composite_score = 0.25*technical + 0.25*catalyst + 0.25*options + 0.25*sentiment

### **2A) Technical Momentum Score (0–10)**
**LOCKED - Do NOT modify. Technical scores are provided by Python and are final.**
You may only reference the technical_score and technical_evidence from the packet.

### **2B) Catalyst Score (0–10)**
Catalyst must be **within the next 7 trading days** and **source-cited**.
Point rubric:
* +4.0: High-impact scheduled event (earnings, FDA decision date, major regulatory decision, investor day) with confirmed date/time
* +3.0: Credible breaking catalyst (M&A talks, contract award, major product launch) supported by reputable sources
* +2.0: Strong analyst actions (multiple upgrades/pt raises) or conference spotlight with expected newsflow
* +1.0: Secondary catalyst (sector tailwind + company-specific hook)
**Penalty rules:** 
* −2.0 if catalyst is rumor-only / single-source / low credibility
* −2.0 if the event timing is unclear (no date)
* −1.0 to −3.0 if catalyst is likely already priced (explain)
**Missing data rule:** if you cannot verify any catalyst with sources → cap catalyst_cap = 4.

### **2C) Options Activity Score (0–10)**
**Options data is now available in packets when `options_data_available` is true.**
Check the packet for:
* `options_score`: Pre-computed score (0-10) from Python
* `options_evidence`: Contains call_put_ratio, unusual_volume_multiple, iv_rank, notable_contracts, etc.
* `options_data_gaps`: Any missing data reasons
* `options_cap_applied`: If score was capped due to missing data

**Scoring rubric (if options_score is provided, use it; otherwise compute from evidence):**
* +2.5 if **call/put volume ratio ≥ 2.0** on **≥ 2×** normal options volume
* +2.5 if **≥ $1M premium** bullish blocks/sweeps in calls (note strikes/expiry)
* +2.0 if **IV Rank 60–80** (elevated, not extreme)
* +1.5 if upside **call skew** indicates demand (define metric used)
* +1.5 if **OI building** in near-term OTM calls (trend, not one print)

**Missing data rule:** If `options_data_available` is false or `options_score` is null → cap options_cap = 3 and note in data_gaps.

### **2D) Sentiment Momentum Score (0–10)**
**Sentiment data is now available in packets when `sentiment_data_available` is true.**
Check the packet for:
* `sentiment_score`: Pre-computed score (0-10) from Python
* `sentiment_evidence`: Contains twitter, reddit, stocktwits, news_tone data
* `sentiment_data_gaps`: Any missing data reasons
* `sentiment_cap_applied`: If score was capped due to missing data

**Scoring rubric (if sentiment_score is provided, use it; otherwise compute from evidence):**
* Twitter/X (40% of sentiment): mention velocity, quality accounts, engagement, bullish vs bearish balance
* Other Platforms (60%): Reddit (20%), StockTwits (20%), News/Analysts (20%)

**Missing data rule:** If `sentiment_data_available` is false or `sentiment_score` is null → cap sentiment_cap = 4 and note in data_gaps.

---
## **3) Tie-breakers & Risk Adjustments**

If composite scores are close (±0.2), rank higher the name with:
1. clearer **dated catalyst**
2. better **liquidity / tighter spreads**
3. cleaner **technical structure** (breakout level + volume confirmation)
Add a **Risk Adjustment Note** (not altering score unless extreme):
* binary event risk (FDA/earnings)
* high short interest / borrow rate
* sector correlation (e.g., biotech, small-cap, crypto proxies)
* market regime risk (SPY trend, VIX)

---
## **4) Required Output (STRICT JSON ONLY)**

Return a single JSON object with:
* run_timestamp_utc
* universe_note
* method_version
* top5 (array of 5 objects sorted by rank)

Each stock object must match:

```json
{
  "rank": 1,
  "ticker": "ABC",
  "name": "Company Name",
  "exchange": "NASDAQ",
  "sector": "Technology",
  "current_price": 123.45,
  "market_cap_usd": 1234567890,
  "avg_dollar_volume_20d": 75000000,
  "asof_price_utc": "2025-12-14T00:00:00Z",

  "target": {
    "horizon_trading_days": 7,
    "upside_threshold_pct": 10,
    "target_price_for_10pct": 135.80,
    "base_case_upside_pct_range": [10, 18],
    "bear_case_note": "What invalidates the setup"
  },

  "primary_catalyst": {
    "title": "Earnings",
    "key_date_local": "2025-12-18",
    "timing": "BMO",
    "why_it_matters": "1-2 sentences",
    "sources": [
      { "title": "Source title", "publisher": "Publisher", "url": "https://...", "published_at": "2025-12-10" }
    ]
  },

  "scores": {
    "technical": 8.5,
    "catalyst": 9.0,
    "options": 7.2,
    "sentiment": 8.0
  },
  "composite_score": 8.18,

  "evidence": {
    "technical": {
      "within_5pct_52w_high": true,
      "resistance_level": 130.0,
      "volume_ratio_3d_to_20d": 1.8,
      "rsi14": 63.2,
      "above_ma10_ma20_ma50": true,
      "realized_vol_5d_ann_pct": 28.0
    },
    "options": {
      "call_put_ratio": null,
      "unusual_volume_multiple": null,
      "largest_bullish_premium_usd": null,
      "iv_rank": null,
      "notable_contracts": []
    },
    "sentiment": {
      "twitter": { "mention_velocity_vs_7d": null, "quality_accounts_est": null, "bullish_pct_est": null },
      "reddit": { "mention_velocity": null, "upvote_ratio_est": null },
      "stocktwits": { "bull_bear_ratio_est": null },
      "news_tone": "positive"
    }
  },

  "risk_factors": [
    "Top risk 1",
    "Top risk 2"
  ],
  "confidence": "HIGH",
  "data_gaps": []
}
```

### **Confidence labels**
* HIGH: all four factor scores present and ≥7, minimal data gaps
* MEDIUM: 3 factors ≥7 and the fourth ≥5, or minor data gaps
* SPECULATIVE: missing options/social data OR only 2 factors ≥7

---
## **5) Final instruction**

Now rank the provided packets and return **only the JSON** response with the **Top 5** ranked results.

**CRITICAL REMINDERS:**
- Technical scores are LOCKED - use them as-is from the packet
- Options data: Use `options_score` if available, otherwise cap at 3.0
- Sentiment data: Use `sentiment_score` if available, otherwise cap at 4.0
- You must still return exactly 5 candidates, even if some have lower scores
- All catalysts must be source-cited from the headlines in the packet
- Output must be valid JSON only, no markdown wrapper

---
## **Packets to Rank**

"""
    
    # Add packets
    for i, packet in enumerate(packets, 1):
        prompt += f"\n=== PACKET {i} ===\n"
        prompt += json.dumps(packet, indent=2, default=str)
        prompt += "\n\n"
    
    prompt += "\n\nNow return the Top 5 ranked results as JSON only (no markdown, no commentary).\n"
    
    return prompt


def call_openai(prompt: str, model: str = "gpt-4o", api_key: Optional[str] = None) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai library not installed")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a quantitative trading analyst. Return only valid JSON. Never include markdown code blocks or commentary outside the JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"} if model.startswith("gpt-4") or "o1" not in model else None,
    )
    
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic library not installed")
    
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=0.3,
        system="You are a quantitative trading analyst. Return only valid JSON. Never include markdown code blocks or commentary outside the JSON.",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.content[0].text


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}\nResponse preview: {text[:500]}")


def rank_weekly_candidates(
    packets: list[dict],
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Rank weekly scanner candidates using LLM.
    
    Args:
        packets: List of candidate packets
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (defaults to provider default)
        api_key: API key (overrides env var)
    
    Returns:
        Dict with top5 ranking results
    """
    import datetime as dt
    
    # Set model defaults
    if not model:
        if provider == "openai":
            model = "gpt-5.2"  # Will fallback if not available
        else:
            model = "claude-3-5-sonnet-20241022"
    
    # Build prompt
    prompt = build_weekly_scanner_prompt(packets)
    
    # Call LLM
    if provider == "openai":
        response_text = call_openai(prompt, model, api_key)
    else:
        response_text = call_anthropic(prompt, model, api_key)
    
    # Parse JSON
    result = extract_json_from_response(response_text)
    
    # Add metadata
    result["run_timestamp_utc"] = dt.datetime.utcnow().isoformat() + "Z"
    result["method_version"] = "v3.0"
    if "universe_note" not in result:
        result["universe_note"] = "SP500 + NASDAQ100 + R2000 (liquid proxy universe)"
    
    return result


def rank_with_debate(
    packets: list[dict],
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    debate_rounds: int = 1,
    debate_top_n: int = 10,
    use_memory: bool = True,
) -> dict:
    """
    Advanced ranking with bull/bear debate and memory integration.
    
    This function:
    1. Enriches packets with historical memory insights
    2. Runs initial LLM ranking
    3. Conducts bull/bear debate on top candidates
    4. Re-ranks based on debate outcomes
    
    Args:
        packets: List of candidate packets
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (defaults to gpt-5.2)
        api_key: API key
        debate_rounds: Number of debate rounds per candidate (1-3)
        debate_top_n: Number of top candidates to debate
        use_memory: Whether to use memory system for context
    
    Returns:
        Dict with enhanced ranking results including debate analysis
    """
    import datetime as dt
    
    model = model or "gpt-5.2"
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    logger.info("=" * 60)
    logger.info("ADVANCED RANKING WITH DEBATE (GPT-5.2)")
    logger.info("=" * 60)
    
    # Step 1: Enrich packets with memory insights
    enriched_packets = packets
    if use_memory and MEMORY_AVAILABLE:
        logger.info("\n[1/4] Enriching with historical memory...")
        memory = get_trading_memory()
        if memory.is_available:
            enriched_packets = []
            for packet in packets:
                enriched = enrich_packet_with_memory(packet, memory)
                enriched_packets.append(enriched)
                
                # Log memory insights
                insights = enriched.get("memory_insights", {})
                if insights.get("n_similar", 0) > 0:
                    logger.info(f"  {packet.get('ticker')}: {insights.get('n_similar')} similar past situations, "
                               f"hit rate: {(insights.get('hit_rate') or 0)*100:.0f}%")
            logger.info(f"  ✓ Enriched {len(enriched_packets)} packets with memory")
        else:
            logger.info("  ⚠ Memory system not available")
    else:
        logger.info("\n[1/4] Memory enrichment skipped")
    
    # Step 2: Initial LLM ranking
    logger.info("\n[2/4] Initial LLM ranking...")
    initial_result = rank_weekly_candidates(
        packets=enriched_packets,
        provider=provider,
        model=model,
        api_key=api_key,
    )
    initial_top5 = initial_result.get("top5", [])
    logger.info(f"  ✓ Initial Top 5: {[t.get('ticker') for t in initial_top5]}")
    
    # Step 3: Bull/Bear Debate on top candidates
    debate_results = {}
    if DEBATE_AVAILABLE and debate_top_n > 0:
        logger.info(f"\n[3/4] Bull/Bear Debate on top {debate_top_n} candidates...")
        
        # Get packets for debate (match by ticker from initial results)
        top_tickers = [t.get("ticker") for t in initial_top5[:debate_top_n]]
        debate_packets = [p for p in enriched_packets if p.get("ticker") in top_tickers]
        
        # Also include next best candidates not in top 5
        remaining_packets = [p for p in enriched_packets if p.get("ticker") not in top_tickers]
        remaining_packets = sorted(
            remaining_packets,
            key=lambda p: p.get("technical_score", 0),
            reverse=True
        )[:debate_top_n - len(debate_packets)]
        debate_packets.extend(remaining_packets)
        
        if debate_packets:
            debate_results = run_batch_debate(
                packets=debate_packets,
                max_rounds=debate_rounds,
                top_n=debate_top_n,
                api_key=api_key,
            )
            
            # Print debate summary
            summary = format_debate_summary(debate_results)
            logger.info(summary)
    else:
        logger.info("\n[3/4] Debate skipped (not available or disabled)")
    
    # Step 4: Re-rank based on debate
    logger.info("\n[4/4] Final ranking with debate adjustments...")
    final_top5 = _rerank_with_debate(initial_top5, debate_results)
    
    # Build enhanced result
    result = {
        "run_timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "method_version": "v4.0-debate",
        "universe_note": "SP500 + NASDAQ100 + R2000 (with debate analysis)",
        "model_used": model,
        "debate_enabled": bool(debate_results),
        "debate_rounds": debate_rounds if debate_results else 0,
        "memory_enabled": use_memory and MEMORY_AVAILABLE,
        "top5": final_top5,
        "debate_analysis": {
            ticker: debate_result_to_dict(dr)
            for ticker, dr in debate_results.items()
        } if debate_results else {},
        "initial_ranking": initial_top5,  # For comparison
    }
    
    # Log final results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RANKING (Post-Debate)")
    logger.info("=" * 60)
    for i, pick in enumerate(final_top5, 1):
        ticker = pick.get("ticker")
        score = pick.get("composite_score", 0)
        verdict = pick.get("debate_verdict", "N/A")
        confidence = pick.get("confidence", "?")
        logger.info(f"  {i}. {ticker}: Score={score:.2f}, Verdict={verdict}, Conf={confidence}")
    
    return result


def _rerank_with_debate(initial_top5: list[dict], debate_results: dict) -> list[dict]:
    """
    Re-rank candidates based on debate results.
    
    Adjusts scores based on:
    - Debate verdict (STRONG_BUY boosts, AVOID penalizes)
    - Conviction delta from debate
    - Risk-adjusted score from debate analysis
    """
    if not debate_results:
        return initial_top5
    
    verdict_adjustments = {
        "STRONG_BUY": 1.5,
        "BUY": 0.5,
        "HOLD": 0.0,
        "AVOID": -1.0,
        "STRONG_AVOID": -2.0,
    }
    
    adjusted = []
    for pick in initial_top5:
        ticker = pick.get("ticker")
        debate = debate_results.get(ticker)
        
        if debate:
            # Get adjustment
            verdict_adj = verdict_adjustments.get(debate.final_verdict, 0)
            conviction_adj = debate.conviction_delta
            
            # Calculate new score
            original_score = pick.get("composite_score", 5.0)
            adjusted_score = original_score + verdict_adj + conviction_adj
            adjusted_score = max(0, min(10, adjusted_score))  # Clamp to 0-10
            
            # Update pick
            pick = pick.copy()
            pick["composite_score"] = adjusted_score
            pick["original_score"] = original_score
            pick["debate_verdict"] = debate.final_verdict
            pick["debate_reasoning"] = debate.verdict_reasoning
            pick["debate_bull_score"] = debate.bull_score
            pick["debate_bear_score"] = debate.bear_score
            pick["key_risks_from_debate"] = debate.key_risks
            pick["key_catalysts_from_debate"] = debate.key_catalysts
            
            # Upgrade/downgrade confidence based on verdict
            if debate.final_verdict in ["STRONG_BUY", "BUY"]:
                if pick.get("confidence") == "SPECULATIVE":
                    pick["confidence"] = "MEDIUM"
                elif pick.get("confidence") == "MEDIUM" and debate.final_verdict == "STRONG_BUY":
                    pick["confidence"] = "HIGH"
            elif debate.final_verdict in ["AVOID", "STRONG_AVOID"]:
                pick["confidence"] = "SPECULATIVE"
        else:
            pick = pick.copy()
            pick["debate_verdict"] = "NOT_DEBATED"
        
        adjusted.append(pick)
    
    # Re-sort by adjusted score
    adjusted.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    
    # Re-assign ranks
    for i, pick in enumerate(adjusted, 1):
        pick["rank"] = i
    
    return adjusted[:5]  # Return top 5

