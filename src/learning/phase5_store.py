import json
from pathlib import Path

from src.utils.time import utc_now


def persist_learning(records, base_path="learning/phase5"):
    if not records:
        return

    Path(base_path).mkdir(parents=True, exist_ok=True)

    fname = f"{utc_now().strftime('%Y-%m-%d')}.json"
    path = Path(base_path) / fname

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
