from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CalibrationSuggestion:
    key: str
    value: Any
    rationale: str


def _hit_rate(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "hit10" not in df.columns:
        return None
    s = df["hit10"].dropna()
    if s.empty:
        return None
    return float(s.mean())


def _best_threshold_by_numeric(
    df: pd.DataFrame,
    score_col: str,
    *,
    min_rows: int = 15,
    higher_is_better: bool = True,
) -> tuple[float | None, pd.DataFrame]:
    """
    Pick a threshold that maximizes hit-rate while retaining at least `min_rows`.

    Returns:
      (best_threshold, table)
    table columns: threshold, n, hit_rate
    """
    if df is None or df.empty or score_col not in df.columns or "hit10" not in df.columns:
        return None, pd.DataFrame(columns=["threshold", "n", "hit_rate"])

    x = df[[score_col, "hit10"]].copy()
    x = x.dropna(subset=[score_col])
    # keep only rows with known hit outcome
    x = x.dropna(subset=["hit10"])
    if x.empty:
        return None, pd.DataFrame(columns=["threshold", "n", "hit_rate"])

    x[score_col] = x[score_col].astype(float)
    # candidate thresholds: unique scores (sorted)
    thresholds = sorted(set(x[score_col].tolist()))
    rows = []
    best_thr = None
    best_hr = -1.0
    best_n = 0

    for thr in thresholds:
        if higher_is_better:
            g = x[x[score_col] >= thr]
        else:
            g = x[x[score_col] <= thr]
        n = int(len(g))
        if n < min_rows:
            continue
        hr = float(g["hit10"].mean())
        rows.append({"threshold": float(thr), "n": n, "hit_rate": hr})
        if hr > best_hr or (hr == best_hr and n > best_n):
            best_hr, best_thr, best_n = hr, float(thr), n

    return best_thr, pd.DataFrame(rows)


def build_calibration_suggestions(
    perf_detail: pd.DataFrame,
    *,
    outputs_root: str | Path = "outputs",
    min_rows: int = 15,
) -> tuple[list[CalibrationSuggestion], dict[str, pd.DataFrame]]:
    """
    Generate simple, backtest-driven suggestions that map cleanly to config keys:
    - quality_filters_weekly.min_technical_score (from weekly top5 historicals)
    - quality_filters_30d.min_score (from pro30 historicals)
    """
    artifacts: dict[str, pd.DataFrame] = {}
    sugg: list[CalibrationSuggestion] = []

    # --- Weekly (uses top5_{date}.csv -> technical_score) ---
    weekly = perf_detail[perf_detail["in_weekly_top5"]].copy() if not perf_detail.empty else pd.DataFrame()
    # enrich weekly with technical_score from top5 artifacts (if present)
    if not weekly.empty:
        rows = []
        for d in sorted(set(weekly["baseline_date"].astype(str).tolist())):
            p = Path(outputs_root) / d / f"top5_{d}.csv"
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty or "ticker" not in df.columns:
                continue
            df = df.copy()
            df["baseline_date"] = d
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            if "technical_score" in df.columns:
                rows.append(df[["baseline_date", "ticker", "technical_score"]])
        if rows:
            tech = pd.concat(rows, ignore_index=True)
            weekly = weekly.merge(tech, on=["baseline_date", "ticker"], how="left")

    if not weekly.empty and "technical_score" in weekly.columns:
        best_thr, table = _best_threshold_by_numeric(weekly, "technical_score", min_rows=min_rows)
        artifacts["weekly_technical_thresholds"] = table
        if best_thr is not None:
            sugg.append(
                CalibrationSuggestion(
                    key="quality_filters_weekly.min_technical_score",
                    value=best_thr,
                    rationale=f"Maximizes Hit10 among weekly picks while keeping ≥{min_rows} rows (historical).",
                )
            )

    # --- PRO30 (uses pro30 CSV Score column) ---
    pro30 = perf_detail[perf_detail["in_pro30"]].copy() if not perf_detail.empty else pd.DataFrame()
    if not pro30.empty:
        # enrich with Score from pro30 outputs
        score_rows = []
        for d in sorted(set(pro30["baseline_date"].astype(str).tolist())):
            for fn in [
                f"30d_momentum_candidates_{d}.csv",
                f"30d_breakout_candidates_{d}.csv",
                f"30d_reversal_candidates_{d}.csv",
            ]:
                p = Path(outputs_root) / d / fn
                if not p.exists():
                    continue
                try:
                    df = pd.read_csv(p)
                except Exception:
                    continue
                if df.empty or "Ticker" not in df.columns or "Score" not in df.columns:
                    continue
                df = df.copy()
                df["baseline_date"] = d
                df["ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
                score_rows.append(df[["baseline_date", "ticker", "Score"]].rename(columns={"Score": "pro30_score"}))
        if score_rows:
            scores = pd.concat(score_rows, ignore_index=True)
            scores = scores.drop_duplicates(subset=["baseline_date", "ticker"], keep="first")
            pro30 = pro30.merge(scores, on=["baseline_date", "ticker"], how="left")

    if not pro30.empty and "pro30_score" in pro30.columns:
        best_thr, table = _best_threshold_by_numeric(pro30, "pro30_score", min_rows=min_rows)
        artifacts["pro30_score_thresholds"] = table
        if best_thr is not None:
            sugg.append(
                CalibrationSuggestion(
                    key="quality_filters_30d.min_score",
                    value=best_thr,
                    rationale=f"Maximizes Hit10 among pro30 picks while keeping ≥{min_rows} rows (historical).",
                )
            )

    return sugg, artifacts


def write_calibration_report(
    perf_by_component: pd.DataFrame,
    perf_by_date: pd.DataFrame,
    suggestions: list[CalibrationSuggestion],
    *,
    output_dir: str | Path = "outputs/performance",
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Markdown report
    lines = []
    lines.append("# Backtest calibration recommendations")
    lines.append("")
    lines.append("KPI: **Hit +10% within 7 trading days** (entry = baseline close; max-forward uses High by default).")
    lines.append("")

    if perf_by_component is not None and not perf_by_component.empty:
        lines.append("## Current hit-rate by component")
        lines.append("")
        for _, r in perf_by_component.iterrows():
            comp = str(r.get("component"))
            n = r.get("n")
            hr = r.get("hit_rate")
            hr_s = "—" if pd.isna(hr) else f"{float(hr):.3f}"
            lines.append(f"- {comp}: n={int(n) if pd.notna(n) else '—'} hit_rate={hr_s}")
        lines.append("")

    if perf_by_date is not None and not perf_by_date.empty:
        lines.append("## Hit-rate by date (all combined picks)")
        lines.append("")
        # show last 10 rows
        tail = perf_by_date.tail(10)
        for _, r in tail.iterrows():
            d = str(r.get("baseline_date"))
            n = r.get("n_tickers")
            hr = r.get("hit_rate_all")
            hr_s = "—" if pd.isna(hr) else f"{float(hr):.3f}"
            lines.append(f"- {d}: n={int(n) if pd.notna(n) else '—'} hit_rate_all={hr_s}")
        lines.append("")

    lines.append("## Suggested config changes (to test)")
    lines.append("")
    if not suggestions:
        lines.append("- (No statistically usable suggestions found yet; try widening the date range.)")
    else:
        for s in suggestions:
            lines.append(f"- `{s.key}`: `{s.value}`  \n  {s.rationale}")
    lines.append("")

    report_path = out / "recommendations.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # YAML snippet (not a patch tool, just values to copy)
    yaml_lines = []
    yaml_lines.append("# Suggested YAML snippet (copy into config/default.yaml)")
    for s in suggestions:
        yaml_lines.append(f"{s.key}: {s.value}")
    yaml_path = out / "suggested_config_snippet.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {
        "recommendations_md": str(report_path),
        "suggested_config_snippet_yaml": str(yaml_path),
    }

