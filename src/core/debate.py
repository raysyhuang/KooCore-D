"""
Bull/Bear Debate System (Inspired by TradingAgents)

Implements adversarial debate between bull and bear perspectives to improve
trading analysis quality. Each candidate is debated before final ranking.

Uses GPT-5.2 for high-quality reasoning.
"""

from __future__ import annotations
import os
import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from src.utils.time import utc_now_iso_z

# Module availability flag
DEBATE_AVAILABLE = True


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, handling None and invalid types."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


@dataclass
class DebateArgument:
    """A single argument in the bull/bear debate."""
    position: str  # "bull" or "bear"
    argument: str
    key_points: list[str]
    confidence: float  # 0-10
    evidence_cited: list[str]


@dataclass
class DebateResult:
    """Result of a bull/bear debate on a candidate."""
    ticker: str
    bull_arguments: list[DebateArgument]
    bear_arguments: list[DebateArgument]
    final_verdict: str  # "STRONG_BUY", "BUY", "HOLD", "AVOID", "STRONG_AVOID"
    verdict_reasoning: str
    bull_score: float  # 0-10
    bear_score: float  # 0-10
    conviction_delta: float  # How much debate changed original assessment
    risk_adjusted_score: float
    key_risks: list[str]
    key_catalysts: list[str]
    debate_rounds: int
    timestamp: str


def build_bull_prompt(packet: dict, bear_argument: str = None, round_num: int = 1) -> str:
    """Build prompt for bull researcher."""
    ticker = packet.get("ticker", "UNKNOWN")
    
    base_prompt = f"""You are a BULL ANALYST advocating for investing in {ticker}.

Your role is to build a strong, evidence-based case for why this stock will achieve a ≥10% gain in the next 7 trading days.

CANDIDATE DATA:
{json.dumps(packet, indent=2, default=str)}

KEY FOCUS AREAS:
1. **Growth Catalysts**: What near-term events could drive the stock higher?
2. **Technical Strength**: Why does the chart support upside?
3. **Sentiment Momentum**: What's driving positive sentiment?
4. **Undervaluation**: Why might the market be underpricing this opportunity?
5. **Risk/Reward**: Why is the upside potential worth the risk?

RULES:
- Use ONLY data from the packet - do not hallucinate facts
- Cite specific evidence (numbers, dates, sources) from the packet
- Be persuasive but honest about data gaps
- Address any weaknesses preemptively
- Confidence must be justified by evidence quality
"""

    if bear_argument and round_num > 1:
        base_prompt += f"""

BEAR ANALYST'S PREVIOUS ARGUMENT (you must counter this):
{bear_argument}

IMPORTANT: Directly address the bear's concerns with specific evidence. Don't just dismiss them - explain why the bull case is stronger despite these concerns.
"""

    base_prompt += """

Respond with JSON only:
{
    "position": "bull",
    "argument": "Your main argument (2-3 paragraphs)",
    "key_points": ["Point 1 with evidence", "Point 2 with evidence", "Point 3"],
    "confidence": 7.5,  // 0-10 based on evidence quality
    "evidence_cited": ["Source/metric 1", "Source/metric 2"],
    "counterpoints_to_bear": ["Counter 1", "Counter 2"]  // Only if responding to bear
}
"""
    return base_prompt


def build_bear_prompt(packet: dict, bull_argument: str = None, round_num: int = 1) -> str:
    """Build prompt for bear researcher."""
    ticker = packet.get("ticker", "UNKNOWN")
    
    base_prompt = f"""You are a BEAR ANALYST arguing AGAINST investing in {ticker}.

Your role is to identify risks, weaknesses, and reasons why this stock may NOT achieve a ≥10% gain in the next 7 trading days.

CANDIDATE DATA:
{json.dumps(packet, indent=2, default=str)}

KEY FOCUS AREAS:
1. **Downside Risks**: What could cause the stock to decline?
2. **Technical Weakness**: Any concerning chart patterns?
3. **Sentiment Risks**: Is the stock overbought? Crowded trade?
4. **Valuation Concerns**: Is the upside already priced in?
5. **Catalyst Risks**: What could go wrong with expected catalysts?
6. **Macro/Sector Risks**: External factors that could hurt the stock?

RULES:
- Use ONLY data from the packet - do not hallucinate facts
- Be specific about what could go wrong and when
- Cite data gaps as risk factors
- Be rigorous but fair - not just contrarian for the sake of it
- Consider probability-weighted scenarios
"""

    if bull_argument and round_num > 1:
        base_prompt += f"""

BULL ANALYST'S PREVIOUS ARGUMENT (you must counter this):
{bull_argument}

IMPORTANT: Directly challenge the bull's optimism with specific concerns. Point out weaknesses in their evidence and assumptions.
"""

    base_prompt += """

Respond with JSON only:
{
    "position": "bear",
    "argument": "Your main argument (2-3 paragraphs)",
    "key_points": ["Risk 1 with evidence", "Risk 2 with evidence", "Risk 3"],
    "confidence": 6.0,  // 0-10 based on evidence quality
    "evidence_cited": ["Source/metric 1", "Source/metric 2"],
    "counterpoints_to_bull": ["Counter 1", "Counter 2"]  // Only if responding to bull
}
"""
    return base_prompt


def build_judge_prompt(
    ticker: str,
    bull_arguments: list[DebateArgument],
    bear_arguments: list[DebateArgument],
    original_score: float,
) -> str:
    """Build prompt for the judge to make final verdict."""
    
    bull_summary = "\n\n".join([
        f"**Bull Round {i+1}** (Confidence: {arg.confidence}/10)\n{arg.argument}\nKey Points: {', '.join(arg.key_points)}"
        for i, arg in enumerate(bull_arguments)
    ])
    
    bear_summary = "\n\n".join([
        f"**Bear Round {i+1}** (Confidence: {arg.confidence}/10)\n{arg.argument}\nKey Points: {', '.join(arg.key_points)}"
        for i, arg in enumerate(bear_arguments)
    ])
    
    return f"""You are the RESEARCH MANAGER making the final investment decision on {ticker}.

You have heard arguments from both Bull and Bear analysts. Your job is to weigh the evidence objectively and deliver a verdict.

ORIGINAL COMPOSITE SCORE: {original_score}/10

=== BULL CASE ===
{bull_summary}

=== BEAR CASE ===
{bear_summary}

DECISION FRAMEWORK:
1. Which side presented stronger EVIDENCE (not just louder arguments)?
2. What is the probability-weighted expected return?
3. What is the risk/reward ratio?
4. How does debate change the original assessment?

VERDICT OPTIONS:
- STRONG_BUY: Bull case is compelling, bear concerns addressed (>80% confidence)
- BUY: Bull case outweighs bear, acceptable risks (60-80% confidence)
- HOLD: Arguments balanced, unclear edge (<60% confidence either way)
- AVOID: Bear concerns valid, risk/reward unfavorable (60-80% bear confidence)
- STRONG_AVOID: Serious red flags identified (>80% bear confidence)

Respond with JSON only:
{{
    "verdict": "BUY",  // One of the options above
    "verdict_reasoning": "2-3 sentence explanation of your decision",
    "bull_score": 7.5,  // How convincing was the bull case (0-10)
    "bear_score": 5.0,  // How convincing was the bear case (0-10)
    "conviction_delta": 0.5,  // How much to adjust original score (-3 to +3)
    "risk_adjusted_score": 7.2,  // Final score after debate
    "key_risks": ["Risk 1", "Risk 2"],
    "key_catalysts": ["Catalyst 1", "Catalyst 2"]
}}
"""


def call_gpt52(prompt: str, api_key: Optional[str] = None) -> str:
    """Call GPT-5.2 (or fallback to gpt-4o if unavailable)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai library not installed")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Try GPT-5.2 first, fallback to gpt-4o
    models_to_try = ["gpt-5.2", "gpt-4o"]
    
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a quantitative trading analyst. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Slightly higher for creative arguments
                response_format={"type": "json_object"} if "gpt-4" in model or "gpt-5" in model else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            if model == models_to_try[-1]:
                raise
            logger.warning(f"Model {model} failed, trying next: {e}")
    
    raise RuntimeError("All models failed")


def extract_json_safe(text: str) -> dict:
    """Safely extract JSON from response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed: {text[:200]}")
        return {}


def run_debate(
    packet: dict,
    max_rounds: int = 2,
    api_key: Optional[str] = None,
) -> DebateResult:
    """
    Run a bull/bear debate on a single candidate.
    
    Args:
        packet: Candidate data packet
        max_rounds: Number of debate rounds (1-3 recommended)
        api_key: OpenAI API key
    
    Returns:
        DebateResult with verdict and scores
    """
    ticker = packet.get("ticker", "UNKNOWN")
    original_score = packet.get("technical_score", 5.0)
    
    bull_arguments = []
    bear_arguments = []
    
    last_bull = None
    last_bear = None
    
    for round_num in range(1, max_rounds + 1):
        logger.info(f"  Debate round {round_num}/{max_rounds} for {ticker}...")
        
        # Bull argues first
        bull_prompt = build_bull_prompt(packet, last_bear, round_num)
        bull_response = call_gpt52(bull_prompt, api_key)
        bull_data = extract_json_safe(bull_response)
        
        if bull_data:
            bull_arg = DebateArgument(
                position="bull",
                argument=bull_data.get("argument", ""),
                key_points=bull_data.get("key_points", []),
                confidence=_safe_float(bull_data.get("confidence"), 5.0),
                evidence_cited=bull_data.get("evidence_cited", []),
            )
            bull_arguments.append(bull_arg)
            last_bull = bull_arg.argument
        
        # Bear responds
        bear_prompt = build_bear_prompt(packet, last_bull, round_num)
        bear_response = call_gpt52(bear_prompt, api_key)
        bear_data = extract_json_safe(bear_response)
        
        if bear_data:
            bear_arg = DebateArgument(
                position="bear",
                argument=bear_data.get("argument", ""),
                key_points=bear_data.get("key_points", []),
                confidence=_safe_float(bear_data.get("confidence"), 5.0),
                evidence_cited=bear_data.get("evidence_cited", []),
            )
            bear_arguments.append(bear_arg)
            last_bear = bear_arg.argument
    
    # Judge makes final decision
    logger.info(f"  Judge deliberating on {ticker}...")
    judge_prompt = build_judge_prompt(ticker, bull_arguments, bear_arguments, original_score)
    judge_response = call_gpt52(judge_prompt, api_key)
    judge_data = extract_json_safe(judge_response)
    
    return DebateResult(
        ticker=ticker,
        bull_arguments=bull_arguments,
        bear_arguments=bear_arguments,
        final_verdict=judge_data.get("verdict", "HOLD"),
        verdict_reasoning=judge_data.get("verdict_reasoning", ""),
        bull_score=_safe_float(judge_data.get("bull_score"), 5.0),
        bear_score=_safe_float(judge_data.get("bear_score"), 5.0),
        conviction_delta=_safe_float(judge_data.get("conviction_delta"), 0.0),
        risk_adjusted_score=_safe_float(judge_data.get("risk_adjusted_score"), original_score),
        key_risks=judge_data.get("key_risks", []),
        key_catalysts=judge_data.get("key_catalysts", []),
        debate_rounds=max_rounds,
        timestamp=utc_now_iso_z(),
    )


def run_batch_debate(
    packets: list[dict],
    max_rounds: int = 1,  # Default to 1 round for speed
    top_n: int = 10,  # Only debate top N candidates
    api_key: Optional[str] = None,
) -> dict[str, DebateResult]:
    """
    Run debates on a batch of candidates.
    
    Args:
        packets: List of candidate packets
        max_rounds: Debate rounds per candidate
        top_n: Only debate top N candidates by technical_score
        api_key: OpenAI API key
    
    Returns:
        Dict mapping ticker to DebateResult
    """
    # Sort by technical_score and take top N
    sorted_packets = sorted(
        packets,
        key=lambda p: p.get("technical_score", 0),
        reverse=True
    )[:top_n]
    
    results = {}
    total = len(sorted_packets)
    
    logger.info(f"Running bull/bear debate on top {total} candidates...")
    
    for i, packet in enumerate(sorted_packets, 1):
        ticker = packet.get("ticker", f"UNKNOWN_{i}")
        logger.info(f"[{i}/{total}] Debating {ticker}...")
        
        try:
            result = run_debate(packet, max_rounds=max_rounds, api_key=api_key)
            results[ticker] = result
        except Exception as e:
            logger.error(f"  Debate failed for {ticker}: {e}")
            # Create minimal result on failure
            results[ticker] = DebateResult(
                ticker=ticker,
                bull_arguments=[],
                bear_arguments=[],
                final_verdict="HOLD",
                verdict_reasoning=f"Debate failed: {e}",
                bull_score=5.0,
                bear_score=5.0,
                conviction_delta=0.0,
                risk_adjusted_score=packet.get("technical_score", 5.0),
                key_risks=["Debate analysis unavailable"],
                key_catalysts=[],
                debate_rounds=0,
                timestamp=utc_now_iso_z(),
            )
    
    return results


def format_debate_summary(debate_results: dict[str, DebateResult]) -> str:
    """Format debate results for display."""
    lines = [
        "",
        "=" * 60,
        "BULL/BEAR DEBATE RESULTS",
        "=" * 60,
        "",
    ]
    
    # Sort by risk_adjusted_score
    sorted_results = sorted(
        debate_results.values(),
        key=lambda r: r.risk_adjusted_score,
        reverse=True
    )
    
    verdict_emoji = {
        "STRONG_BUY": "🟢🟢",
        "BUY": "🟢",
        "HOLD": "🟡",
        "AVOID": "🔴",
        "STRONG_AVOID": "🔴🔴",
    }
    
    for i, result in enumerate(sorted_results, 1):
        emoji = verdict_emoji.get(result.final_verdict, "⚪")
        lines.append(
            f"{i}. {result.ticker}: {emoji} {result.final_verdict} "
            f"(Score: {result.risk_adjusted_score:.1f}, "
            f"Bull: {result.bull_score:.1f}, Bear: {result.bear_score:.1f})"
        )
        lines.append(f"   └─ {result.verdict_reasoning[:80]}...")
        if result.key_risks:
            lines.append(f"   └─ Risks: {', '.join(result.key_risks[:2])}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def debate_result_to_dict(result: DebateResult) -> dict:
    """Convert DebateResult to serializable dict."""
    return {
        "ticker": result.ticker,
        "final_verdict": result.final_verdict,
        "verdict_reasoning": result.verdict_reasoning,
        "bull_score": result.bull_score,
        "bear_score": result.bear_score,
        "conviction_delta": result.conviction_delta,
        "risk_adjusted_score": result.risk_adjusted_score,
        "key_risks": result.key_risks,
        "key_catalysts": result.key_catalysts,
        "debate_rounds": result.debate_rounds,
        "timestamp": result.timestamp,
        "bull_arguments": [
            {
                "argument": arg.argument[:500],  # Truncate for storage
                "key_points": arg.key_points,
                "confidence": arg.confidence,
            }
            for arg in result.bull_arguments
        ],
        "bear_arguments": [
            {
                "argument": arg.argument[:500],
                "key_points": arg.key_points,
                "confidence": arg.confidence,
            }
            for arg in result.bear_arguments
        ],
    }
