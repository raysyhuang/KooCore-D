"""
Main HTML Report Builder

Orchestrates the construction of the comprehensive HTML report with TOC, executive summary,
ticker cards, and collapsible deep dives.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from .components import build_kpi_card, build_ticker_card, build_toc, build_table_with_search, build_overlap_heat_row
from .formatters import escape_html, fmt_num, fmt_pct
from .assets import CSS, JAVASCRIPT


def build_html_report(report_data: Any) -> str:
    """
    Build the comprehensive HTML report.
    
    Args:
        report_data: ReportData dataclass instance with all report data
    
    Returns:
        Complete HTML string
    """
    # Extract data
    rd = report_data
    date_str = rd.date_str
    
    # Timestamps
    import pandas as pd
    now_local = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S ET")
    
    method_version = (
        str(rd.run_metadata.get("method_version") or rd.top5_json.get("method_version") or "v3.0")
    )
    
    # Regime info
    regime = rd.run_metadata.get("regime_info") or {}
    regime_ok = bool(regime.get("ok")) if regime else False
    regime_label = "OK" if regime_ok else "NOT OK"
    regime_cls = "ok" if regime_ok else "bad"
    spy_last = regime.get("spy_last")
    spy_ma = regime.get("spy_ma")
    vix_last = regime.get("vix_last")
    regime_msg = regime.get("message") or ""
    
    # Primary Top 5 (Swing preferred)
    top5_list = rd.top5_json.get("top5") or []
    weekly_top5_df = pd.DataFrame(top5_list) if top5_list else pd.DataFrame()
    primary_label = rd.top5_json.get("primary_label")
    if not primary_label:
        primary_label = "Swing" if rd.swing_top5_json else "Weekly"
    
    # As-of date
    asof_price_utc = None
    try:
        if not weekly_top5_df.empty and "asof_price_utc" in weekly_top5_df.columns:
            asof_price_utc = str(weekly_top5_df["asof_price_utc"].dropna().iloc[0])
        elif rd.swing_candidates is not None and not rd.swing_candidates.empty and "asof_price_utc" in rd.swing_candidates.columns:
            asof_price_utc = str(rd.swing_candidates["asof_price_utc"].dropna().iloc[0])
        elif not rd.weekly_candidates.empty and "asof_price_utc" in rd.weekly_candidates.columns:
            asof_price_utc = str(rd.weekly_candidates["asof_price_utc"].dropna().iloc[0])
        elif rd.weekly_packets_json.get("packets"):
            asof_price_utc = str(rd.weekly_packets_json["packets"][0].get("asof_price_utc") or "")
    except Exception:
        asof_price_utc = None
    asof_date = (asof_price_utc or "").split("T")[0] if asof_price_utc else "Unknown"
    
    # KPIs
    tickers_screened = rd.run_metadata.get("tickers_screened")
    pro30_candidates_count = int(len(rd.pro30_candidates)) if rd.pro30_candidates is not None else 0
    weekly_top5_count = int(len(top5_list))
    swing_candidates_count = int(len(rd.swing_candidates)) if rd.swing_candidates is not None else 0
    
    movers_list = rd.hybrid_json.get("movers_tickers") or []
    movers_count = int(len(movers_list))
    
    # Compute overlaps
    pro30_tickers = set()
    for df in (rd.pro30_candidates, rd.pro30_reversal):
        if df is not None and not df.empty and "Ticker" in df.columns:
            pro30_tickers |= set(df["Ticker"].astype(str).tolist())
    
    weekly_top5_tickers = set()
    if top5_list:
        weekly_top5_tickers = {str(x.get("ticker")) for x in top5_list if x.get("ticker")}
    
    movers_tickers = set(map(str, movers_list)) if movers_list else set()
    
    overlap_all_three = sorted(list(weekly_top5_tickers & pro30_tickers & movers_tickers))
    overlap_weekly_pro30 = sorted(list((weekly_top5_tickers & pro30_tickers) - set(overlap_all_three)))
    overlap_weekly_movers = sorted(list((weekly_top5_tickers & movers_tickers) - set(overlap_all_three)))
    overlap_pro30_movers = sorted(list((pro30_tickers & movers_tickers) - set(overlap_all_three)))
    
    overlaps = {
        "all_three": overlap_all_three,
        "weekly_pro30": overlap_weekly_pro30,
        "weekly_movers": overlap_weekly_movers,
        "pro30_movers": overlap_pro30_movers,
    }
    
    # Build TOC sections
    toc_sections = [
        ("exec-summary", "Executive Summary"),
        ("top5", "Top 5 Candidates"),
    ]
    
    # Add ticker sections (will be under ticker-cards section)
    if top5_list:
        toc_sections.append(("ticker-cards", "Detailed Analysis"))
        for item in top5_list:
            ticker = item.get("ticker", "UNKNOWN")
            name = item.get("name", "")
            name_short = name[:20] + "..." if len(name) > 20 else name
            toc_sections.append((f"ticker-{ticker.lower()}", f"  {ticker} — {name_short}"))
    
    toc_sections.extend([
        ("swing-candidates", "Swing Candidates"),
        ("pro30-candidates", "30-Day Candidates"),
        ("reversal-candidates", "Reversal Candidates"),
        ("weekly-pool", "Weekly Candidate Pool"),
        ("deep-dive", "Deep Dive — LLM Packets"),
        ("appendix", "Appendix"),
    ])
    
    toc_html = build_toc(toc_sections)
    
    # Build Executive Summary
    kpi_cards = [
        build_kpi_card("Tickers Screened", tickers_screened if tickers_screened is not None else "—", "Universe size for PRO30 run"),
        build_kpi_card("Swing Candidates", swing_candidates_count, "Swing strategy candidate pool"),
        build_kpi_card("PRO30 Candidates", pro30_candidates_count, "Momentum + reversal screen output"),
        build_kpi_card(f"{primary_label} Top 5", weekly_top5_count, f"{primary_label} Top 5 JSON"),
        build_kpi_card("Daily Movers", movers_count, "Daily gainers/losers/reversals (if enabled)"),
    ]
    
    overlap_heat = build_overlap_heat_row(overlaps)
    
    exec_summary_html = f"""
    <section id="exec-summary" class="exec-summary">
      <h2>Executive Summary</h2>
      <div class="kpis">
        {"".join(kpi_cards)}
      </div>
      {overlap_heat}
      <div class="note" style="margin-top: 16px;">
        <strong>Regime Gate:</strong> {escape_html(regime_msg)}<br/>
        <strong>Data as-of:</strong> {escape_html(asof_date)} (UTC in packets)
      </div>
    </section>"""
    
    # Build Top 5 table with links
    top5_table_html = ""
    if not weekly_top5_df.empty:
        # Create simplified display table with links
        display_rows = []
        for _, row in weekly_top5_df.iterrows():
            ticker = str(row.get("ticker", ""))
            display_rows.append({
                "Rank": row.get("rank", ""),
                "Ticker": f'<a href="#ticker-{ticker.lower()}">{escape_html(ticker)}</a>',
                "Name": row.get("name", ""),
                "Composite Score": fmt_num(row.get("composite_score", 0), decimals=2),
                "Verdict": row.get("verdict") or row.get("confidence", ""),
                "Technical": fmt_num(row.get("scores", {}).get("technical", 0) if isinstance(row.get("scores"), dict) else 0, decimals=1),
                "Catalyst": fmt_num(row.get("scores", {}).get("catalyst", 0) if isinstance(row.get("scores"), dict) else 0, decimals=1),
                "Price": fmt_num(row.get("current_price", 0), decimals=2),
            })
        
        top5_display_df = pd.DataFrame(display_rows)
        top5_table_html = f"""
    <section id="top5" class="card">
      <h2>Top 5 Candidates ({primary_label} strategy)</h2>
      <p class="note">Click ticker symbol to jump to detailed analysis.</p>
      <div class="toolbar">
        <div class="search">
          <span class="badge">Search</span>
          <input type="search" placeholder="Filter rows by any text…" data-table="weeklyTop5" id="search-weeklyTop5"/>
        </div>
        <div class="btns">
          <button data-export="weeklyTop5">Export CSV (visible rows)</button>
          <button data-reset="weeklyTop5">Reset</button>
        </div>
      </div>
      <div class="tablewrap">
        {top5_display_df.to_html(index=False, classes=["dataframe", "table"], table_id="weeklyTop5", border=0, escape=False)}
      </div>
    </section>"""
    else:
        top5_table_html = """
    <section id="top5" class="card">
      <h2>Top 5 Candidates ({primary_label} strategy)</h2>
      <div class="empty">No Top 5 candidates available.</div>
    </section>"""
    
    # Build Ticker Cards (each as its own section, not wrapped in a card)
    ticker_cards_content = ""
    for item in top5_list:
        rank = item.get("rank", 0)
        ticker_cards_content += build_ticker_card(item, rank)
    
    if not ticker_cards_content:
        ticker_cards_html = '<div class="empty">No ticker cards to display.</div>'
    else:
        # Wrap in a section for TOC
        ticker_cards_html = f"""
    <section id="ticker-cards">
      <h2 style="margin-bottom: 24px; font-size: 20px;">Detailed Analysis — {primary_label} Top 5</h2>
      {ticker_cards_content}
    </section>"""
    
    # Build other tables
    swing_table_html = build_table_with_search(rd.swing_candidates, "swingCandidates", "Swing Candidates (primary strategy pool)")
    pro30_table_html = build_table_with_search(rd.pro30_candidates, "pro30Candidates", "PRO30 Candidates (30-day screen output)")
    reversal_table_html = build_table_with_search(rd.pro30_reversal, "reversalCandidates", "Reversal Candidates (30-day reversal subset)")
    weekly_pool_html = build_table_with_search(rd.weekly_candidates, "weeklyCandidates", "Weekly Scanner — Candidate Pool")
    
    # Deep dive packets (collapsible)
    packet_text = rd.llm_packets_txt.strip()
    if not packet_text:
        packet_text = "(No llm_packets text file found for this run.)"
    
    # Split packets by ticker (if possible) or show as one block
    deep_dive_html = f"""
    <section id="deep-dive" class="card">
      <h2>Deep Dive — LLM Packets</h2>
      <p class="note">Full packet text for detailed review of penalties, gaps, and trade-plan templates.</p>
      <details>
        <summary>Show/Hide All Packets</summary>
        <pre>{escape_html(packet_text)}</pre>
      </details>
    </section>"""
    
    # Appendix
    files = sorted([p.name for p in rd.output_dir.glob("*") if p.is_file()])
    files_li = "\n".join([f'<li><span class="mono">{escape_html(name)}</span></li>' for name in files])
    
    appendix_html = f"""
    <section id="appendix" class="card">
      <h2>Appendix</h2>
      <h3>Timestamps</h3>
      <p class="note">
        UTC: <span class="mono">{escape_html(rd.run_metadata.get("run_timestamp_utc","—"))}</span><br/>
        ET: <span class="mono">{escape_html(rd.run_metadata.get("run_timestamp_et","—"))}</span>
      </p>
      <h3>Files in this report</h3>
      <ul class="note">
        {files_li}
      </ul>
      <h3>What to do next (practical checklist)</h3>
      <ol class="note">
        <li><b>Validate catalysts:</b> confirm earnings date and scan 14–30d headlines from a second source for your final decision.</li>
        <li><b>Entry discipline:</b> only trade when your setup trigger actually happens (e.g., reclaim MA20/MA50 + volume/RSI confirmation).</li>
        <li><b>Risk control:</b> size positions off ATR and risk 1–2% per trade (your packets already include a template).</li>
      </ol>
    </section>"""
    
    # Assemble full HTML
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Momentum Scanner Report — {escape_html(date_str)}</title>
  <style>
{CSS}
  </style>
</head>
<body>
<div class="wrap">
  {toc_html}
  
  <div class="main-content">
    <header>
      <div>
        <h1 class="h-title">Momentum Scanner Report — {escape_html(date_str)}</h1>
        <p class="h-sub">Generated locally at <span class="mono">{escape_html(now_local)}</span></p>
      </div>
      <div class="pillrow">
        <div class="pill"><b>Method</b>: {escape_html(method_version)}</div>
        <div class="pill"><b>Regime Gate</b>: <span class="{regime_cls}">{escape_html(regime_label)}</span></div>
        <div class="pill"><b>SPY</b>: {fmt_num(spy_last)} vs MA20 {fmt_num(spy_ma)}</div>
        <div class="pill"><b>VIX</b>: {fmt_num(vix_last)}</div>
      </div>
    </header>

    {exec_summary_html}
    
    {top5_table_html}
    
    {ticker_cards_html}
    
    <section id="swing-candidates" class="card">
      <h2>Swing Candidates (Primary Strategy Pool)</h2>
      {swing_table_html}
    </section>
    
    <section id="pro30-candidates" class="card">
      <h2>30-Day Candidates</h2>
      {pro30_table_html}
      <p class="note">If this table shows only a few tickers, your filters were strict (or market regime + liquidity gates removed most names).</p>
    </section>
    
    <section id="reversal-candidates" class="card">
      <h2>Reversal Candidates</h2>
      {reversal_table_html}
    </section>
    
    <section id="weekly-pool" class="card">
      <h2>Weekly Scanner — Candidate Pool</h2>
      {weekly_pool_html}
    </section>
    
    {deep_dive_html}
    
    {appendix_html}
    
    <footer>
      <div class="note">
        Disclaimer: This report is a screening + organization artifact. Not financial advice. Always validate with real-time data, earnings calendar, SEC filings, and liquidity/borrow constraints.
      </div>
    </footer>
  </div>
</div>

<script>
{JAVASCRIPT}
</script>
</body>
</html>"""
    
    return html

