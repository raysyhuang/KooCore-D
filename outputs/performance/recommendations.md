# Backtest calibration recommendations

KPI: **Hit +10% within 7 trading days** (entry = baseline close; max-forward uses High by default).

## Current hit-rate by component

- all: n=120 hit_rate=0.367
- weekly_top5: n=90 hit_rate=0.300
- pro30: n=31 hit_rate=0.548
- movers: n=0 hit_rate=—

## Hit-rate by date (all combined picks)

- 2025-12-01: n=6 hit_rate_all=0.333
- 2025-12-02: n=6 hit_rate_all=0.333
- 2025-12-03: n=7 hit_rate_all=0.429
- 2025-12-04: n=7 hit_rate_all=0.429
- 2025-12-05: n=9 hit_rate_all=0.444
- 2025-12-08: n=8 hit_rate_all=0.375
- 2025-12-09: n=7 hit_rate_all=0.286
- 2025-12-10: n=5 hit_rate_all=0.400
- 2025-12-30: n=6 hit_rate_all=0.500
- 2025-12-31: n=6 hit_rate_all=0.833

## Suggested config changes (to test)

- `quality_filters_weekly.min_technical_score`: `6.0`  
  Maximizes Hit10 among weekly picks while keeping ≥15 rows (historical).
- `quality_filters_30d.min_score`: `27.41`  
  Maximizes Hit10 among pro30 picks while keeping ≥15 rows (historical).
