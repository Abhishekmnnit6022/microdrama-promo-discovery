"""Editor feedback store (append-only JSONL) used by the review screen and by eval/evaluate.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

DECISIONS = ("approved", "partially_correct", "rejected")
REJECTION_REASONS = (
    "weak opening hook", "no real cliffhanger", "needs prior context", "wrong timestamps",
    "duplicate of another promo", "segments do not fit together", "spoiler / reveals too much", "other",
)


def record(path: Path, promo: dict, decision: str, reason: str = "", note: str = "", editor: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "editor": editor,
        "promo_id": promo.get("promo_id"),
        "rank": promo.get("rank"),
        "show": promo.get("show"),
        "episodes": promo.get("episodes"),
        "segments": [{k: s[k] for k in ("episode_id", "start_seconds", "end_seconds")} for s in promo.get("segments", [])],
        "priority_score": promo.get("priority_score"),
        "decision": decision,
        "reason": reason,
        "note": note,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def latest_by_promo(entries: list) -> dict:
    """Most recent decision per promo_id."""
    out = {}
    for e in entries:
        out[e.get("promo_id")] = e
    return out


def summary(entries: list) -> dict:
    latest = latest_by_promo(entries)
    n = len(latest)
    counts = {d: sum(1 for e in latest.values() if e["decision"] == d) for d in DECISIONS}
    reasons = {}
    for e in latest.values():
        if e["decision"] != "approved" and e.get("reason"):
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    return {
        "reviewed_promos": n,
        **counts,
        "acceptance_rate": round(counts["approved"] / n, 4) if n else None,
        "acceptance_rate_incl_partial": round((counts["approved"] + counts["partially_correct"]) / n, 4) if n else None,
        "rejection_reasons": reasons,
    }
