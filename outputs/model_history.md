# Model History & Analysis Log

This file tracks model runs, observations, and improvement experiments.

---

## How to Use This File

After running scans for a few days/weeks, use Cursor to:
1. Analyze patterns in winners vs losers
2. Identify filter/scoring improvements
3. Track experiments and their results

---

## Quick Analysis Commands

Ask Cursor:
- "Analyze the last 10 runs in outputs/ - which picks hit +10%? What do they have in common?"
- "Look at dropped_pro30 files - are we filtering good stocks?"
- "Compare hit rates by composite_score bucket - should we raise thresholds?"

---

## Run History

<!-- New runs will be appended below -->


### 📊 Validation Check (2026-01-14)

**KPI:** Hit +10% within T+7 days
- Hit Rate: 12.1%
- Win Rate: 62.1%
- Avg Return: 2.3%
- Model Health: 🔴 Poor

**Strategy Performance:**
- pro30: 21.1% hit rate (n=19)
- weekly_top5: 8.9% hit rate (n=45)
- movers: 0.0% hit rate (n=2)

---

### 📊 Validation Check (2026-01-14)

**KPI:** Hit +7% within T+7 days
- Hit Rate: 0.0%
- Win Rate: 68.2%
- Avg Return: 3.9%
- Model Health: 🔴 Poor

**Strategy Performance:**
- weekly_top5: 0.0% hit rate (n=45)
- pro30: 0.0% hit rate (n=19)
- movers: 0.0% hit rate (n=2)

---

### 📊 Validation Check (2026-01-14)

**KPI:** Hit +7% within T+7 days
- Hit Rate: 43.9%
- Win Rate: 68.2%
- Avg Return: 3.9%
- Model Health: 🟢 Excellent

**Strategy Performance:**
- pro30: 57.9% hit rate (n=19)
- weekly_top5: 40.0% hit rate (n=45)
- movers: 0.0% hit rate (n=2)

**Improvement Suggestions:**
- [High] Pro30 outperforms Weekly (57.9% vs 40.0%). Increase Pro30 weight.

---

### 2026-01-15 (run: 2026-01-15 05:59 UTC)

**Picks:**
- Weekly Top 5: (none)
- Pro30: (none)
- Movers: (none)

**Overlaps:**
- All Three: (none)
- Weekly+Pro30: (none)

---

### 2026-01-15 (run: 2026-01-15 09:10 UTC)

**Picks:**
- Weekly Top 5: (none)
- Pro30: (none)
- Movers: (none)

**Overlaps:**
- All Three: (none)
- Weekly+Pro30: (none)

---

### 2026-01-15 (run: 2026-01-15 14:36 UTC)

**Picks:**
- Weekly Top 5: (none)
- Pro30: (none)
- Movers: (none)

**Overlaps:**
- All Three: (none)
- Weekly+Pro30: (none)

---

### 2026-01-15 (run: 2026-01-15 23:35 UTC)

**Picks:**
- Weekly Top 5: (none)
- Pro30: (none)
- Movers: WGS, CRM, IRTC, STNG, TVTX, FIG, PATH

**Overlaps:**
- All Three: (none)
- Weekly+Pro30: (none)

---
