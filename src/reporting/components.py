"""
HTML Component Builders

Reusable components for building the HTML report: KPI cards, ticker cards, TOC, tables.
"""

from __future__ import annotations
from typing import Any, Optional
import pandas as pd
from .formatters import escape_html, fmt_num, fmt_pct, fmt_currency, fmt_verdict_badge, fmt_confidence_badge


def build_kpi_card(label: str, value: Any, hint: str = "") -> str:
    """Build a KPI card HTML."""
    value_str = str(value) if value is not None else "—"
    hint_html = f'<div class="hint">{escape_html(hint)}</div>' if hint else ""
    return f"""
    <div class="kpi">
      <div class="label">{escape_html(label)}</div>
      <div class="value">{escape_html(value_str)}</div>
      {hint_html}
    </div>"""


def build_ticker_card(ticker_data: dict[str, Any], rank: int) -> str:
    """
    Build a ticker card section with verdict, scores, metrics, and risk flags.
    
    Args:
        ticker_data: Dict with ticker information from top5 JSON
        rank: Rank (1-5)
    
    Returns:
        HTML string for the ticker card
    """
    ticker = ticker_data.get("ticker", "UNKNOWN")
    name = ticker_data.get("name", "")
    sector = ticker_data.get("sector", "")
    exchange = ticker_data.get("exchange", "")
    
    # Verdict and confidence
    # Try to extract verdict from LLM response format
    verdict = ticker_data.get("verdict")
    if not verdict:
        # Fallback: use confidence as verdict indicator
        confidence_val = ticker_data.get("confidence", "")
        if "HIGH" in str(confidence_val).upper():
            verdict = "BUY"
        elif "SPECULATIVE" in str(confidence_val).upper() or "LOW" in str(confidence_val).upper():
            verdict = "WATCH"
        else:
            verdict = "WATCH"  # Default
    
    verdict_class, verdict_text = fmt_verdict_badge(verdict)
    confidence = ticker_data.get("confidence", "")
    conf_class, conf_text = fmt_confidence_badge(confidence)
    
    # Scores
    scores = ticker_data.get("scores", {})
    technical_score = scores.get("technical", 0)
    catalyst_score = scores.get("catalyst", 0)
    options_score = scores.get("options", 0)
    sentiment_score = scores.get("sentiment", 0)
    composite_score = ticker_data.get("composite_score", 0)
    
    # Current price and target
    current_price = ticker_data.get("current_price", 0)
    target = ticker_data.get("target", {})
    target_price = target.get("target_price_for_10pct", 0)
    upside_pct = target.get("base_case_upside_pct_range", [0, 0])
    
    # Primary catalyst
    catalyst = ticker_data.get("primary_catalyst", {})
    catalyst_title = catalyst.get("title", "Unknown catalyst")
    catalyst_timing = catalyst.get("timing", "Unknown")
    
    # Evidence
    evidence = ticker_data.get("evidence", {})
    tech_evidence = evidence.get("technical", {})
    rsi14 = tech_evidence.get("rsi14")
    vol_ratio = tech_evidence.get("volume_ratio_3d_to_20d")
    realized_vol = tech_evidence.get("realized_vol_5d_ann_pct")
    above_mas = tech_evidence.get("above_ma10_ma20_ma50", False)
    
    # Risk factors
    risk_factors = ticker_data.get("risk_factors", [])
    risk_html = ""
    if risk_factors:
        risk_items = "".join([f'<li>{escape_html(rf)}</li>' for rf in risk_factors])
        risk_html = f"""
        <div class="risk-panel">
          <h4>⚠️ Risk Factors</h4>
          <ul class="risk-list">
            {risk_items}
          </ul>
        </div>"""
    
    # Data gaps
    data_gaps = ticker_data.get("data_gaps", [])
    gaps_html = ""
    if data_gaps:
        gap_items = "".join([f'<li>{escape_html(gap)}</li>' for gap in data_gaps])
        gaps_html = f"""
        <div class="gaps-panel">
          <h4>📊 Data Gaps</h4>
          <ul class="gaps-list">
            {gap_items}
          </ul>
        </div>"""
    
    # Why included (1-line summary)
    why_included = f"{catalyst_title} ({catalyst_timing})"
    
    return f"""
    <div id="ticker-{ticker.lower()}" class="ticker-card">
      <div class="ticker-header">
        <div class="ticker-title">
          <h3>
            <span class="rank">#{rank}</span>
            <span class="ticker-symbol">{escape_html(ticker)}</span>
            <span class="ticker-name">{escape_html(name)}</span>
          </h3>
          <div class="ticker-meta">
            {escape_html(sector)} • {escape_html(exchange)}
          </div>
        </div>
        <div class="ticker-badges">
          <span class="{verdict_class}">{escape_html(verdict_text)}</span>
          <span class="{conf_class}">{escape_html(conf_text)}</span>
        </div>
      </div>
      
      <div class="ticker-body">
        <div class="ticker-summary">
          <p class="why-included"><strong>Why included:</strong> {escape_html(why_included)}</p>
        </div>
        
        <div class="ticker-scores">
          <div class="score-grid">
            <div class="score-item">
              <div class="score-label">Technical</div>
              <div class="score-value">{fmt_num(technical_score, decimals=1)}</div>
            </div>
            <div class="score-item">
              <div class="score-label">Catalyst</div>
              <div class="score-value">{fmt_num(catalyst_score, decimals=1)}</div>
            </div>
            <div class="score-item">
              <div class="score-label">Options</div>
              <div class="score-value">{fmt_num(options_score, decimals=1)}</div>
            </div>
            <div class="score-item">
              <div class="score-label">Sentiment</div>
              <div class="score-value">{fmt_num(sentiment_score, decimals=1)}</div>
            </div>
            <div class="score-item score-composite">
              <div class="score-label">Composite</div>
              <div class="score-value">{fmt_num(composite_score, decimals=2)}</div>
            </div>
          </div>
        </div>
        
        <div class="ticker-metrics">
          <table class="metrics-table">
            <tbody>
              <tr>
                <td>Current Price</td>
                <td><strong>{fmt_currency(current_price)}</strong></td>
              </tr>
              <tr>
                <td>Target (10% move)</td>
                <td><strong>{fmt_currency(target_price)}</strong></td>
              </tr>
              <tr>
                <td>Upside Range</td>
                <td><strong>{fmt_pct(upside_pct[0] if upside_pct else 0)} - {fmt_pct(upside_pct[1] if len(upside_pct) > 1 else 0)}</strong></td>
              </tr>
              <tr>
                <td>RSI(14)</td>
                <td>{fmt_num(rsi14, decimals=1)}</td>
              </tr>
              <tr>
                <td>Volume Ratio (3d/20d)</td>
                <td>{fmt_num(vol_ratio, decimals=2)}</td>
              </tr>
              <tr>
                <td>Realized Vol (5d ann.)</td>
                <td>{fmt_pct(realized_vol, decimals=1)}</td>
              </tr>
              <tr>
                <td>Above All MAs</td>
                <td>{"✓" if above_mas else "✗"}</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        {risk_html}
        {gaps_html}
      </div>
    </div>"""


def build_toc(sections: list[tuple[str, str]]) -> str:
    """
    Build table of contents HTML.
    
    Args:
        sections: List of (anchor_id, section_title) tuples
    
    Returns:
        HTML string for sticky TOC sidebar
    """
    toc_items = []
    for anchor_id, title in sections:
        toc_items.append(f'<li><a href="#{anchor_id}">{escape_html(title)}</a></li>')
    
    toc_list = "\n".join(toc_items)
    
    return f"""
    <nav class="toc-sidebar" id="toc-sidebar">
      <div class="toc-header">
        <h4>Contents</h4>
      </div>
      <ul class="toc-list">
        {toc_list}
      </ul>
    </nav>"""


def build_table_with_search(df: pd.DataFrame, table_id: str, title: str = "") -> str:
    """
    Build a searchable, exportable table.
    
    Args:
        df: DataFrame to render
        table_id: Unique ID for the table
        title: Optional table title
    
    Returns:
        HTML string with table, search box, and export button
    """
    if df is None or df.empty:
        return f'<div class="empty">No rows.</div>'
    
    # Check if DataFrame contains HTML (for links)
    contains_html = False
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample, str) and ("<a" in sample or "<span" in sample):
                contains_html = True
                break
    
    table_html = df.to_html(index=False, classes=["dataframe", "table"], table_id=table_id, border=0, escape=not contains_html)
    
    title_html = f'<h3>{escape_html(title)}</h3>' if title else ""
    
    return f"""
    {title_html}
    <div class="toolbar">
      <div class="search">
        <span class="badge">Search</span>
        <input type="search" placeholder="Filter rows by any text…" data-table="{table_id}" id="search-{table_id}"/>
      </div>
      <div class="btns">
        <button data-export="{table_id}">Export CSV (visible rows)</button>
        <button data-reset="{table_id}">Reset</button>
      </div>
    </div>
    <div class="tablewrap">
      {table_html}
    </div>"""


def build_overlap_heat_row(overlaps: dict[str, list[str]]) -> str:
    """
    Build overlap heat row showing which tickers appear in multiple screeners.
    
    Args:
        overlaps: Dict with keys like "all_three", "weekly_pro30", etc. and ticker lists
    
    Returns:
        HTML string for overlap visualization
    """
    all_three = overlaps.get("all_three", [])
    weekly_pro30 = overlaps.get("weekly_pro30", [])
    weekly_movers = overlaps.get("weekly_movers", [])
    pro30_movers = overlaps.get("pro30_movers", [])
    
    items = []
    
    if all_three:
        tickers_str = ", ".join([escape_html(t) for t in all_three])
        items.append(f'<div class="overlap-item overlap-all"><span class="overlap-icon">⭐</span><strong>ALL THREE</strong> ({len(all_three)}): {tickers_str}</div>')
    
    if weekly_pro30:
        tickers_str = ", ".join([escape_html(t) for t in weekly_pro30])
        items.append(f'<div class="overlap-item overlap-strong"><span class="overlap-icon">🔥</span><strong>Weekly + 30-Day</strong> ({len(weekly_pro30)}): {tickers_str}</div>')
    
    if weekly_movers:
        tickers_str = ", ".join([escape_html(t) for t in weekly_movers])
        items.append(f'<div class="overlap-item overlap-medium"><span class="overlap-icon">📈</span><strong>Weekly + Movers</strong> ({len(weekly_movers)}): {tickers_str}</div>')
    
    if pro30_movers:
        tickers_str = ", ".join([escape_html(t) for t in pro30_movers])
        items.append(f'<div class="overlap-item overlap-medium"><span class="overlap-icon">💎</span><strong>30-Day + Movers</strong> ({len(pro30_movers)}): {tickers_str}</div>')
    
    if not items:
        return '<div class="overlap-item">No overlaps found.</div>'
    
    return f'<div class="overlap-heat-row">{"".join(items)}</div>'

