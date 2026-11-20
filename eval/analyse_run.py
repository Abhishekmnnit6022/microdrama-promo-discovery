"""Print the numbers used in raw_outputs/OUTPUT_ANALYSIS.md, computed from a run folder (no re-scoring).

Usage: python -m eval.analyse_run --run raw_outputs > raw_outputs/6_evaluation_and_tests/run_breakdown.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.batch import output_path, reference_results_dir  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(reference_results_dir(ROOT)))
    a = ap.parse_args(argv)
    d = Path(a.run)
    man = json.loads(output_path(d, "run_manifest.json").read_text(encoding="utf-8"))
    rec = json.loads(output_path(d, "recommendations.json").read_text(encoding="utf-8"))["promos"]
    rows = list(csv.DictReader(output_path(d, "candidates_all.csv").open(encoding="utf-8")))

    print("== summary ==")
    for k, v in man["stats"]["summary"].items():
        print(f"{k:34s} {v}")
    print("\n== per structure (all candidates) ==")
    for st in ("continuous", "multi_segment", "cross_episode"):
        r = [x for x in rows if x["structure"] == st]
        if not r:
            continue
        p = [x for x in r if x["passed"] == "True"]
        print(f"{st:15s} n={len(r):6d} passed={len(p):6d} ({len(p) / len(r):.1%}) "
              f"mean_score={statistics.mean(float(x['priority_score']) for x in r):.3f} "
              f"max={max(float(x['priority_score']) for x in r):.3f}")
    print("\n== stage-2 shortlist ==")
    s2 = [x for x in rows if x["stage"] == "2"]
    print(f"shortlisted={len(s2)} passed={sum(1 for x in s2 if x['passed'] == 'True')}")
    print("rejection reasons (stage 2):", Counter(r.split(' (')[0] for x in s2 for r in x["rejection_reasons"].split(" | ") if r).most_common())
    print("\n== rejection reasons (all) ==")
    for k, v in man["stats"]["rejection_reason_counts"].items():
        print(f"{v:7d}  {k}")
    print("\n== component score distribution (all candidates) ==")
    for c in ("hook", "cliffhanger", "coherence", "emotion"):
        vals = sorted(float(x[c]) for x in rows)
        q = statistics.quantiles(vals, n=4)
        print(f"{c:12s} p25={q[0]:.2f} median={q[1]:.2f} p75={q[2]:.2f}")
    print("\n== recommended ==")
    for p in rec:
        segs = " + ".join(f"ep{s['episode_number']} {s['start']}-{s['end']}" for s in p["segments"])
        llm = p.get("llm_review") or {}
        v = (f"LLM hook={llm.get('hook_works')} cliff={llm.get('cliffhanger_works')} standalone={llm.get('standalone')}"
             if llm.get("status") == "ok" else f"LLM {llm.get('status')}")
        print(f"#{p['rank']:2d} {p['priority_score']:.3f} {p['total_duration_seconds']:5.1f}s {p['structure']:13s} {segs}")
        print(f"     hook: {p['opening_hook']['evidence'][0]} | cliff: {p['ending_cliffhanger']['evidence'][0]} | coh {p['standalone_coherence']['score']:.2f} | {v}")
        if p["risk_flags"]:
            print(f"     flags: {'; '.join(p['risk_flags'])}")
    print("\nhook reasons:", Counter(p["opening_hook"]["evidence"][0] for p in rec).most_common())
    print("cliff reasons:", Counter(p["ending_cliffhanger"]["evidence"][0].split(" (")[0] for p in rec).most_common())
    print("episodes covered:", sorted({s["episode_number"] for p in rec for s in p["segments"]}))
    print("\n== episodes without a valid single-episode promo ==")
    for e in man["episode_status"]:
        if e["status"] != "ok":
            print(f"ep {e['episode_number']}: {e['status']} {e['error']} {e['no_promo_reason']}")


if __name__ == "__main__":
    main()
