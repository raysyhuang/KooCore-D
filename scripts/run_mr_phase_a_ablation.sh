#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Run MR-only Phase A ablation backtests.

Usage:
  scripts/run_mr_phase_a_ablation.sh 1y
  scripts/run_mr_phase_a_ablation.sh 3y
  scripts/run_mr_phase_a_ablation.sh 5y
  scripts/run_mr_phase_a_ablation.sh 1y mr_a4_mr_score_floor_up_plus_bull

Periods:
  1y  -> 2025-03-14 to 2026-03-13
  3y  -> 2023-03-14 to 2026-03-13
  5y  -> 2021-03-14 to 2026-03-13
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

period="$1"
single_variant="${2:-}"

case "$period" in
  1y)
    start="2025-03-14"
    end="2026-03-13"
    ;;
  3y)
    start="2023-03-14"
    end="2026-03-13"
    ;;
  5y)
    start="2021-03-14"
    end="2026-03-13"
    ;;
  *)
    echo "Unknown period: $period" >&2
    usage
    exit 1
    ;;
esac

variants=(
  mr_a0_baseline
  mr_a1_bull_tight_1
  mr_a2_bull_tight_2
  mr_a3_mr_score_floor_up
  mr_a4_mr_score_floor_up_plus_bull
  mr_a5_rsi_deeper
  mr_a6_rsi_deeper_plus_score
  mr_a7_liquidity_up
  mr_a8_damage_filter
  mr_a9_acceptance_tighter
)

if [[ -n "$single_variant" ]]; then
  variants=("$single_variant")
fi

for variant in "${variants[@]}"; do
  config_path="config/experiments/${variant}.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 1
  fi

  label="${variant}_${period}"
  echo
  echo "=== ${label} ($(date '+%Y-%m-%d %H:%M:%S')) ==="
  python scripts/backtest_1yr.py \
    --start "$start" \
    --end "$end" \
    --config "$config_path" \
    --acceptance-mode live_equivalent \
    --engines mr_only \
    --label "$label"
done
