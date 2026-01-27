"""
Additional Artifact Generators

Markdown summary, CSV exports, and JSON run cards.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import pandas as pd
from .formatters import fmt_num, fmt_pct, fmt_currency, fmt_date


def generate_summary_md(report_data: Any, output_path: Path) -> Path:
    """
    Generate Obsidian-friendly Markdown summary.
    
    Args:
        report_data: ReportData instance
        output_path: Path to save the markdown file
    
    Returns:
        Path to saved file
    """
    rd = report_data
    
    # Extract key data
    top5_list = rd.top5_json.get("top5") or []
    primary_label = rd.top5_json.get("primary_label") or ("Swing" if rd.swing_top5_json else "Weekly")
    method_version = rd.run_metadata.get("method_version", "v3.0")
    primary_label = rd.top5_json.get("primary_label") or ("Swing" if rd.swing_top5_json else "Weekly")
    
    # Regime
    regime = rd.run_metadata.get("regime_info") or {}
    regime_ok = regime.get("ok", False)
    
    # Overlaps
    pro30_tickers = set()
    for df in (rd.pro30_candidates, rd.pro30_reversal):
        if df is not None and not df.empty and "Ticker" in df.columns:
            pro30_tickers |= set(df["Ticker"].astype(str).tolist())
    
    weekly_top5_tickers = {str(x.get("ticker")) for x in top5_list if x.get("ticker")}
    movers_tickers = set(rd.hybrid_json.get("movers_tickers", []))
    
    overlap_all_three = sorted(list(weekly_top5_tickers & pro30_tickers & movers_tickers))
    overlap_weekly_pro30 = sorted(list((weekly_top5_tickers & pro30_tickers) - set(overlap_all_three)))
    
    md_lines = [
        f"# Momentum Scanner Report — {rd.date_str}",
        "",
        f"**Generated:** {rd.run_metadata.get('run_timestamp_utc', 'Unknown')}",
        f"**Method:** {method_version}",
        f"**Regime Gate:** {'✅ OK' if regime_ok else '❌ NOT OK'}",
        "",
        "## Executive Summary",
        "",
        f"- **Tickers Screened:** {rd.run_metadata.get('tickers_screened', '—')}",
        f"- **Swing Candidates:** {len(rd.swing_candidates) if rd.swing_candidates is not None else 0}",
        f"- **PRO30 Candidates:** {len(rd.pro30_candidates) if rd.pro30_candidates is not None else 0}",
        f"- **{primary_label} Top 5:** {len(top5_list)}",
        f"- **Daily Movers:** {len(movers_tickers)}",
        "",
        "## Overlaps (Higher Conviction)",
        "",
    ]
    
    if overlap_all_three:
        md_lines.append(f"### ⭐ ALL THREE ({len(overlap_all_three)})")
        for t in overlap_all_three:
            md_lines.append(f"- {t}")
        md_lines.append("")
    
    if overlap_weekly_pro30:
        md_lines.append(f"### 🔥 {primary_label} + 30-Day ({len(overlap_weekly_pro30)})")
        for t in overlap_weekly_pro30:
            md_lines.append(f"- {t}")
        md_lines.append("")
    
    # Hybrid Top 3 (best across all models)
    hybrid_top3 = rd.hybrid_json.get("hybrid_top3", [])
    if hybrid_top3:
        md_lines.extend([
            "## Hybrid Top 3 (Best Across All Models)",
            "",
            "*Based on weighted scoring across Primary + Pro30 + Movers (weights configurable)*",
            "",
        ])
        
        for item in hybrid_top3[:3]:
            ticker = item.get("ticker", "UNKNOWN")
            name = item.get("name", "") or "(Pro30/Movers)"
            hybrid_score = item.get("hybrid_score", 0)
            sources = ", ".join(item.get("sources", []))
            confidence = item.get("confidence", "SPECULATIVE")
            catalyst = item.get("primary_catalyst", {}).get("title", "N/A") if isinstance(item.get("primary_catalyst"), dict) else "N/A"
            
            md_lines.extend([
                f"### {ticker} — {name}",
                "",
                f"- **Hybrid Score:** {fmt_num(hybrid_score, decimals=1)} pts",
                f"- **Sources:** {sources}",
                f"- **Confidence:** {confidence}",
                f"- **Primary Catalyst:** {catalyst}",
                "",
            ])
    
    # Also show Primary Top 5 for reference
    md_lines.extend([
        f"## {primary_label} Top 5 (Reference)",
        "",
    ])
    
    for item in top5_list:
        ticker = item.get("ticker", "UNKNOWN")
        name = item.get("name", "")
        composite = item.get("composite_score", 0)
        verdict = item.get("verdict") or item.get("confidence", "UNKNOWN")
        catalyst = item.get("primary_catalyst", {}).get("title", "Unknown")
        
        md_lines.extend([
            f"### {ticker} — {name}",
            "",
            f"- **Composite Score:** {fmt_num(composite, decimals=2)}",
            f"- **Verdict:** {verdict}",
            f"- **Primary Catalyst:** {catalyst}",
            "",
        ])
    
    md_lines.extend([
        "## Notes",
        "",
        "- Validate catalysts with real-time data",
        "- Entry discipline: wait for setup triggers",
        "- Risk control: 1–2% per trade",
        "",
    ])
    
    md_content = "\n".join(md_lines)
    output_path.write_text(md_content, encoding="utf-8")
    return output_path


def generate_top5_csv(report_data: Any, output_path: Path) -> Path:
    """
    Generate human-ordered Top picks CSV with key columns.
    Prioritizes Hybrid Top 3 (best across all models) over Weekly Top 5.
    
    Args:
        report_data: ReportData instance
        output_path: Path to save the CSV file
    
    Returns:
        Path to saved file
    """
    rd = report_data
    
    # Prioritize Hybrid Top 3 (best across all models)
    hybrid_top3 = rd.hybrid_json.get("hybrid_top3", [])
    top5_list = rd.top5_json.get("top5") or []
    
    # Use Hybrid Top 3 as primary, fall back to Weekly Top 5
    primary_list = hybrid_top3 if hybrid_top3 else top5_list
    
    if not primary_list:
        # Create empty CSV with expected columns
        df = pd.DataFrame(columns=[
            "rank", "ticker", "name", "sector", "hybrid_score", "sources",
            "current_price", "composite_score", "confidence",
            "technical_score", "catalyst_score", "options_score", "sentiment_score",
            "primary_catalyst", "target_price"
        ])
        df.to_csv(output_path, index=False)
        return output_path
    
    # Build DataFrame with human-ordered columns
    rows = []
    for item in primary_list:
        target = item.get("target", {}) or {}
        catalyst = item.get("primary_catalyst", {}) or {}
        scores = item.get("scores", {}) or {}
        
        row = {
            "rank": item.get("rank", 0),
            "ticker": item.get("ticker", ""),
            "name": item.get("name", ""),
            "sector": item.get("sector", ""),
            "hybrid_score": item.get("hybrid_score", 0),
            "sources": ", ".join(item.get("sources", [])) if item.get("sources") else "",
            "current_price": item.get("current_price", 0),
            "composite_score": item.get("composite_score", 0),
            "confidence": item.get("confidence", ""),
            "technical_score": scores.get("technical", 0),
            "catalyst_score": scores.get("catalyst", 0),
            "options_score": scores.get("options", 0),
            "sentiment_score": scores.get("sentiment", 0),
            "primary_catalyst": catalyst.get("title", "") if isinstance(catalyst, dict) else "",
            "target_price": target.get("target_price_for_10pct", 0) if isinstance(target, dict) else 0,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def generate_run_card_json(report_data: Any, output_path: Path) -> Path:
    """
    Generate essential run metadata as JSON card.
    
    Args:
        report_data: ReportData instance
        output_path: Path to save the JSON file
    
    Returns:
        Path to saved file
    """
    rd = report_data
    
    top5_list = rd.top5_json.get("top5") or []
    top5_tickers = [item.get("ticker") for item in top5_list if item.get("ticker")]
    primary_label = rd.top5_json.get("primary_label") or ("Swing" if rd.swing_top5_json else "Weekly")
    
    # Compute overlaps
    pro30_tickers = set()
    for df in (rd.pro30_candidates, rd.pro30_reversal):
        if df is not None and not df.empty and "Ticker" in df.columns:
            pro30_tickers |= set(df["Ticker"].astype(str).tolist())
    
    weekly_top5_tickers = set(top5_tickers)
    movers_tickers = set(rd.hybrid_json.get("movers_tickers", []))
    
    overlap_all_three = sorted(list(weekly_top5_tickers & pro30_tickers & movers_tickers))
    
    card = {
        "date": rd.date_str,
        "run_timestamp_utc": rd.run_metadata.get("run_timestamp_utc"),
        "method_version": rd.run_metadata.get("method_version", "v3.0"),
        "summary": {
            "tickers_screened": rd.run_metadata.get("tickers_screened"),
            "swing_candidates": len(rd.swing_candidates) if rd.swing_candidates is not None else 0,
            "pro30_candidates": len(rd.pro30_candidates) if rd.pro30_candidates is not None else 0,
            "weekly_top5": len(top5_list),
            "movers": len(movers_tickers),
        },
        "top5_tickers": top5_tickers,
        "overlap_all_three": overlap_all_three,
        "primary_label": primary_label,
        "regime": {
            "ok": rd.run_metadata.get("regime_info", {}).get("ok", False),
            "message": rd.run_metadata.get("regime_info", {}).get("message", ""),
        },
    }
    
    output_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return output_path

