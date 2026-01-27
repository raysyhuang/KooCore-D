"""
HTML Report Generator (Hybrid-style)

Generates a single, comprehensive, easy-to-read HTML report in the style of
`hybrid_report_YYYY-MM-DD.html` (dark theme, searchable/exportable tables, packet deep dives).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class ReportData:
    date_str: str
    output_dir: Path
    top5_json: dict[str, Any]
    swing_candidates: pd.DataFrame
    swing_top5_json: dict[str, Any]
    weekly_candidates: pd.DataFrame
    weekly_packets_json: dict[str, Any]
    pro30_candidates: pd.DataFrame
    pro30_reversal: pd.DataFrame
    hybrid_json: dict[str, Any]
    run_metadata: dict[str, Any]
    llm_packets_txt: str


def generate_html_report(output_dir: Path, date_str: str) -> Path:
    """
    Generate the hybrid-style HTML report into the run's output folder.

    Writes **two files** (same content) for backwards compatibility:
    - outputs/YYYY-MM-DD/report_YYYY-MM-DD.html
    - outputs/YYYY-MM-DD/hybrid_report_YYYY-MM-DD.html

    Also generates new artifacts:
    - outputs/YYYY-MM-DD/summary_YYYY-MM-DD.md
    - outputs/YYYY-MM-DD/top5_YYYY-MM-DD.csv
    - outputs/YYYY-MM-DD/run_card_YYYY-MM-DD.json

    Returns:
        Path to `report_YYYY-MM-DD.html` (stable name used by the CLI).
    """
    out = Path(output_dir)
    rd = _load_report_data(out, date_str)
    
    # Use new reporting package
    try:
        from ..reporting import build_html_report as build_new_report
        from ..reporting import generate_summary_md, generate_top5_csv, generate_run_card_json
        
        html = build_new_report(rd)
        
        # Generate new artifacts
        summary_md = out / f"summary_{date_str}.md"
        top5_csv = out / f"top5_{date_str}.csv"
        run_card_json = out / f"run_card_{date_str}.json"
        
        generate_summary_md(rd, summary_md)
        generate_top5_csv(rd, top5_csv)
        generate_run_card_json(rd, run_card_json)
    except ImportError:
        # Fallback to old renderer if new package not available
        html = _render_hybrid_style_html(rd)

    report_file = out / f"report_{date_str}.html"
    hybrid_file = out / f"hybrid_report_{date_str}.html"

    report_file.write_text(html, encoding="utf-8")
    hybrid_file.write_text(html, encoding="utf-8")

    return report_file


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_report_data(output_dir: Path, date_str: str) -> ReportData:
    swing_top5_json = _safe_read_json(output_dir / f"swing_top5_{date_str}.json")
    weekly_top5_json = _safe_read_json(output_dir / f"weekly_scanner_top5_{date_str}.json")
    top5_json = swing_top5_json or weekly_top5_json
    swing_candidates = _safe_read_csv(output_dir / f"swing_candidates_{date_str}.csv")
    weekly_candidates = _safe_read_csv(output_dir / f"weekly_scanner_candidates_{date_str}.csv")
    weekly_packets_json = _safe_read_json(output_dir / f"weekly_scanner_packets_{date_str}.json")

    # PRO30: we treat the "momentum candidates" file as the main candidate table
    pro30_candidates = _safe_read_csv(output_dir / f"30d_momentum_candidates_{date_str}.csv")
    pro30_reversal = _safe_read_csv(output_dir / f"30d_reversal_candidates_{date_str}.csv")

    hybrid_json = _safe_read_json(output_dir / f"hybrid_analysis_{date_str}.json")
    run_metadata = _safe_read_json(output_dir / "run_metadata.json")
    llm_packets_txt = _safe_read_text(output_dir / f"llm_packets_{date_str}.txt")

    return ReportData(
        date_str=date_str,
        output_dir=output_dir,
        top5_json=top5_json,
        swing_candidates=swing_candidates,
        swing_top5_json=swing_top5_json,
        weekly_candidates=weekly_candidates,
        weekly_packets_json=weekly_packets_json,
        pro30_candidates=pro30_candidates,
        pro30_reversal=pro30_reversal,
        hybrid_json=hybrid_json,
        run_metadata=run_metadata,
        llm_packets_txt=llm_packets_txt,
    )


def _df_to_table(df: pd.DataFrame, table_id: str) -> str:
    if df is None or df.empty:
        return '<div class="empty">No rows.</div>'
    # Keep the same class names used by the provided HTML (so CSS applies)
    return df.to_html(index=False, classes=["dataframe", "table"], table_id=table_id, border=0, escape=True)


def _fmt_num(x: Any, default: str = "—") -> str:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return "1" if x else "0"
        return f"{float(x):,.2f}".rstrip("0").rstrip(".")
    except Exception:
        return default


def _render_hybrid_style_html(rd: ReportData) -> str:
    # Use NY timezone for display
    import pandas as pd
    now_local = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S ET")

    method_version = (
        str(rd.run_metadata.get("method_version") or rd.top5_json.get("method_version") or "v3.0")
    )

    regime = rd.run_metadata.get("regime_info") or {}
    regime_ok = bool(regime.get("ok")) if regime else False
    regime_label = "OK" if regime_ok else "NOT OK"
    regime_cls = "ok" if regime_ok else "bad"
    spy_last = regime.get("spy_last")
    spy_ma = regime.get("spy_ma")
    vix_last = regime.get("vix_last")
    regime_msg = regime.get("message") or ""

    # Weekly Top 5
    top5_list = rd.top5_json.get("top5") or []
    weekly_top5_df = pd.DataFrame(top5_list) if top5_list else pd.DataFrame()

    # Derive as-of date (best-effort)
    asof_price_utc = None
    try:
        if not weekly_top5_df.empty and "asof_price_utc" in weekly_top5_df.columns:
            asof_price_utc = str(weekly_top5_df["asof_price_utc"].dropna().iloc[0])
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

    # Movers: we don’t currently persist movers tickers to disk, so default to hybrid_json value
    movers_list = rd.hybrid_json.get("movers_tickers") or []
    movers_count = int(len(movers_list))

    # Overlaps computed from actual available sets (don’t trust stale hybrid_analysis.json)
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

    # Tables
    weekly_top5_table = _df_to_table(weekly_top5_df, "weeklyTop5")
    pro30_candidates_table = _df_to_table(rd.pro30_candidates, "pro30Candidates")
    reversal_candidates_table = _df_to_table(rd.pro30_reversal, "reversalCandidates")
    weekly_candidates_table = _df_to_table(rd.weekly_candidates, "weeklyCandidates")

    # Deep dive packet text: show full llm packet dump (already formatted)
    packet_title = "LLM Packets (verbatim)"
    packet_text = rd.llm_packets_txt.strip()
    if not packet_text:
        packet_text = "(No llm_packets text file found for this run.)"

    # File list
    files = sorted([p.name for p in rd.output_dir.glob("*") if p.is_file()])
    files_li = "\n".join([f'            <li><span class="mono">{name}</span></li>' for name in files])

    # NOTE: inline JS (filter/export) is copied from your preferred HTML.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Hybrid Momentum & Reversal Scanner Report — {rd.date_str}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --card: rgba(255,255,255,0.06);
      --card2: rgba(255,255,255,0.08);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --muted2: rgba(255,255,255,0.5);
      --accent: #7dd3fc;
      --accent2: #a78bfa;
      --good: #86efac;
      --warn: #fde047;
      --bad: #fb7185;
      --border: rgba(255,255,255,0.12);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }}
    html, body {{ background: radial-gradient(1200px 800px at 20% 0%, rgba(167,139,250,0.18), transparent 60%),
                          radial-gradient(1000px 700px at 80% 20%, rgba(125,211,252,0.18), transparent 55%),
                          var(--bg);
                 color: var(--text); font-family: var(--sans); margin: 0; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px 70px; }}
    header {{ display:flex; gap:16px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; }}
    .h-title {{ font-size: 26px; letter-spacing: 0.2px; margin: 0; }}
    .h-sub {{ margin: 6px 0 0; color: var(--muted); font-size: 14px; }}
    .pillrow {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .pill {{ padding: 8px 10px; border: 1px solid var(--border); background: rgba(0,0,0,0.12);
            border-radius: 999px; font-size: 12px; color: var(--muted); }}
    .pill b {{ color: var(--text); font-weight: 600; }}
    .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap: 14px; margin-top: 18px; }}
    .card {{ grid-column: span 12; background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 16px 16px 14px; box-shadow: 0 12px 30px rgba(0,0,0,0.25); }}
    .card h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .card h3 {{ margin: 14px 0 8px; font-size: 14px; color: var(--text);}}
    .kpis {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .kpi {{ background: var(--card2); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
    .kpi .label {{ color: var(--muted); font-size: 12px; }}
    .kpi .value {{ font-size: 20px; margin-top: 4px; }}
    .kpi .hint {{ margin-top: 6px; color: var(--muted2); font-size: 12px; }}
    .split {{ display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 14px; }}
    @media (max-width: 980px) {{ .split {{ grid-template-columns: 1fr; }} .kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; margin: 10px 0 10px; }}
    .search {{ display:flex; gap:8px; align-items:center; }}
    input[type="search"] {{ width: min(520px, 72vw); padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border); background: rgba(0,0,0,0.2); color: var(--text); outline: none; }}
    .btns {{ display:flex; gap:8px; flex-wrap:wrap; }}
    button {{ cursor:pointer; padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border); background: rgba(0,0,0,0.15); color: var(--text); }}
    button:hover {{ background: rgba(255,255,255,0.08); }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .mono {{ font-family: var(--mono); }}
    .badge {{ font-size: 12px; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); background: rgba(0,0,0,0.2); color: var(--muted); }}
    .ok {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    .tablewrap {{ overflow:auto; border-radius: 14px; border: 1px solid var(--border); }}
    table.table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    table.table thead th {{ position: sticky; top: 0; background: rgba(10,16,32,0.92); border-bottom: 1px solid var(--border); padding: 10px 10px; text-align:left; white-space: nowrap; }}
    table.table td {{ border-bottom: 1px solid rgba(255,255,255,0.08); padding: 8px 10px; white-space: nowrap; }}
    table.table tr:hover td {{ background: rgba(255,255,255,0.04); }}
    .empty {{ padding: 12px; border: 1px dashed rgba(255,255,255,0.22); border-radius: 14px; color: var(--muted); }}
    details {{ border: 1px solid var(--border); border-radius: 14px; padding: 12px; background: rgba(0,0,0,0.14); }}
    details > summary {{ cursor: pointer; color: var(--text); font-weight: 600; }}
    pre {{ margin: 10px 0 0; white-space: pre-wrap; word-break: break-word; font-family: var(--mono); font-size: 12px; color: rgba(255,255,255,0.86); }}
    footer {{ margin-top: 22px; color: var(--muted2); font-size: 12px; }}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 class="h-title">Hybrid Momentum & Reversal Scanner Report — {rd.date_str}</h1>
      <p class="h-sub">Generated locally at <span class="mono">{now_local}</span>. Data as-of prices: {asof_date} (UTC in packets).</p>
    </div>
    <div class="pillrow">
      <div class="pill"><b>Method</b>: {method_version}</div>
      <div class="pill"><b>Regime Gate</b>: <span class="{regime_cls}">{regime_label}</span></div>
      <div class="pill"><b>SPY</b>: {_fmt_num(spy_last)} vs MA20 {_fmt_num(spy_ma)}</div>
      <div class="pill"><b>VIX</b>: {_fmt_num(vix_last)}</div>
    </div>
  </header>

  <div class="grid">
    <section class="card">
      <div class="split">
        <div>
          <h2>Run Overview</h2>
          <p class="note">{regime_msg}</p>
          <div class="kpis">
            <div class="kpi">
              <div class="label">Tickers screened</div>
              <div class="value">{tickers_screened if tickers_screened is not None else "—"}</div>
              <div class="hint">Universe size for PRO30 run</div>
            </div>
            <div class="kpi">
              <div class="label">PRO30 candidates</div>
              <div class="value">{pro30_candidates_count}</div>
              <div class="hint">Momentum + reversal screen output</div>
            </div>
            <div class="kpi">
              <div class="label">Weekly Top 5</div>
              <div class="value">{weekly_top5_count}</div>
              <div class="hint">Weekly scanner Top 5 JSON</div>
            </div>
            <div class="kpi">
              <div class="label">Movers</div>
              <div class="value">{movers_count}</div>
              <div class="hint">Daily gainers/losers/reversals (if enabled)</div>
            </div>
          </div>

          <h3>Overlaps</h3>
          <p class="note">
            All three: <span class="mono">{", ".join(overlap_all_three) if overlap_all_three else "—"}</span><br/>
            Weekly ∩ PRO30: <span class="mono">{", ".join(overlap_weekly_pro30) if overlap_weekly_pro30 else "—"}</span><br/>
            Weekly ∩ Movers: <span class="mono">{", ".join(overlap_weekly_movers) if overlap_weekly_movers else "—"}</span><br/>
            PRO30 ∩ Movers: <span class="mono">{", ".join(overlap_pro30_movers) if overlap_pro30_movers else "—"}</span>
          </p>
        </div>

        <div>
          <h2>Timestamps</h2>
          <p class="note">
            UTC: <span class="mono">{rd.run_metadata.get("run_timestamp_utc","—")}</span><br/>
            ET: <span class="mono">{rd.run_metadata.get("run_timestamp_et","—")}</span><br/>
            Local note: these timestamps come from the run metadata JSON.
          </p>

          <h3>Files in this report</h3>
          <ul class="note">
{files_li}
          </ul>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Weekly Scanner — Top 5 (7 trading days, ≥10% target)</h2>
      <p class="note">
        Note: this table is generated from <span class="mono">weekly_scanner_top5_{rd.date_str}.json</span> (not from <span class="mono">hybrid_analysis</span>).
      </p>
      <div class="tablewrap">{weekly_top5_table}</div>
    </section>

    <section class="card">
      <h2>PRO30 Candidates (30-day screen output)</h2>
      <div class="toolbar">
        <div class="search">
          <span class="badge">Search</span>
          <input type="search" placeholder="Filter rows by any text… (e.g., AAP, Reversal, RSI)" data-table="pro30Candidates"/>
        </div>
        <div class="btns">
          <button data-export="pro30Candidates">Export CSV (visible rows)</button>
          <button data-reset="pro30Candidates">Reset</button>
        </div>
      </div>
      <div class="tablewrap">{pro30_candidates_table}</div>
      <p class="note">If this table shows only a few tickers, your filters were strict (or market regime + liquidity gates removed most names).</p>
    </section>

    <section class="card">
      <h2>Reversal Candidates (30-day reversal subset)</h2>
      <div class="toolbar">
        <div class="search">
          <span class="badge">Search</span>
          <input type="search" placeholder="Filter rows by any text…" data-table="reversalCandidates"/>
        </div>
        <div class="btns">
          <button data-export="reversalCandidates">Export CSV (visible rows)</button>
          <button data-reset="reversalCandidates">Reset</button>
        </div>
      </div>
      <div class="tablewrap">{reversal_candidates_table}</div>
    </section>

    <section class="card">
      <h2>Weekly Scanner — Candidate Pool</h2>
      <div class="toolbar">
        <div class="search">
          <span class="badge">Search</span>
          <input type="search" placeholder="Filter weekly candidates…" data-table="weeklyCandidates"/>
        </div>
        <div class="btns">
          <button data-export="weeklyCandidates">Export CSV (visible rows)</button>
          <button data-reset="weeklyCandidates">Reset</button>
        </div>
      </div>
      <div class="tablewrap">{weekly_candidates_table}</div>
    </section>

    <section class="card">
      <h2>Deep Dive — {packet_title}</h2>
      <p class="note">
        Below is the full packet text so you can review penalties, gaps, and the trade-plan template.
      </p>
      <details open>
        <summary>Show/Hide packets</summary>
        <pre>{packet_text}</pre>
      </details>
    </section>

    <section class="card">
      <h2>What to do next (practical checklist)</h2>
      <ol class="note">
        <li><b>Validate catalysts:</b> confirm earnings date and scan 14–30d headlines from a second source for your final decision.</li>
        <li><b>Entry discipline:</b> only trade when your setup trigger actually happens (e.g., reclaim MA20/MA50 + volume/RSI confirmation).</li>
        <li><b>Risk control:</b> size positions off ATR and risk 1–2% per trade (your packets already include a template).</li>
      </ol>
    </section>

    <footer>
      <div class="note">
        Disclaimer: This report is a screening + organization artifact. Not financial advice. Always validate with real-time data, earnings calendar, SEC filings, and liquidity/borrow constraints.
      </div>
    </footer>
  </div>
</div>

<script>
/* Tiny table filter + export (no external dependencies) */
function getTable(id) {{ return document.getElementById(id); }}

function filterTable(tableId, query) {{
  const table = getTable(tableId);
  if (!table) return;
  const q = query.trim().toLowerCase();
  const rows = Array.from(table.tBodies[0].rows);
  rows.forEach(r => {{
    const txt = r.innerText.toLowerCase();
    r.style.display = (q === \"\" || txt.includes(q)) ? \"\" : \"none\";
  }});
}}

function resetTable(tableId) {{
  const table = getTable(tableId);
  if (!table) return;
  Array.from(table.tBodies[0].rows).forEach(r => r.style.display = \"\");
  document.querySelectorAll('input[data-table=\"'+tableId+'\"]').forEach(inp => inp.value = \"\");
}}

function exportVisibleRowsToCSV(tableId) {{
  const table = getTable(tableId);
  if (!table) return;
  const rows = Array.from(table.querySelectorAll(\"tr\"))
    .filter(r => r.style.display !== \"none\");
  const csv = rows.map(r => Array.from(r.children).map(c => {{
      const t = (c.innerText ?? \"\").replaceAll('\"','\"\"');
      return '\"' + t + '\"';
    }}).join(\",\")).join(\"\\n\");
  const blob = new Blob([csv], {{type: \"text/csv;charset=utf-8;\"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement(\"a\");
  a.href = url;
  a.download = tableId + \"_visible.csv\";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

document.querySelectorAll('input[type=\"search\"][data-table]').forEach(inp => {{
  inp.addEventListener(\"input\", () => {{
    filterTable(inp.getAttribute(\"data-table\"), inp.value);
  }});
}});

document.querySelectorAll('button[data-export]').forEach(btn => {{
  btn.addEventListener(\"click\", () => exportVisibleRowsToCSV(btn.getAttribute(\"data-export\")));
}});

document.querySelectorAll('button[data-reset]').forEach(btn => {{
  btn.addEventListener(\"click\", () => resetTable(btn.getAttribute(\"data-reset\")));
}});
</script>
</body>
</html>
"""

