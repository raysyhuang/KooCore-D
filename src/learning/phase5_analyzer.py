def summarize_learning(records):
    summary = {}

    for r in records or []:
        key = f"{r.get('source')}|{r.get('regime')}"
        summary.setdefault(key, {"count": 0, "hits": 0})

        summary[key]["count"] += 1
        if r.get("hit_7pct"):
            summary[key]["hits"] += 1

    for _, v in summary.items():
        v["hit_rate"] = round(v["hits"] / max(v["count"], 1), 2)

    return summary
