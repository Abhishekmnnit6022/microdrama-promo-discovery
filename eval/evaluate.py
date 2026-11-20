"""Evaluation of a finished run.

Three clearly separated kinds of evidence:

1. AUTOMATED CHECKS (always available) — computed from the run outputs: valid-duration rate, timestamp
   validity, duplicate rate, hook/cliffhanger/coherence gate rates, processing time per episode.
2. SILVER PROXY CHECKS (always available, NOT human judgement) — use structure of the scripts themselves:
   (a) cliffhanger proxy: micro-drama writers end each episode on a hook, so the detector should rank an
       episode's final line above its other possible cut points;
   (b) emotion-model sanity: the model never sees the writer's emotion tags (e.g. "(Furious)"), so its arousal
       should correlate with tag intensity.
3. HUMAN EVALUATION (only if data exists) — editor decisions from feedback/editor_feedback.jsonl and labelled
   promos in eval/labels/*.json (schema: eval/labels_schema.json). If none exist, the report says so.

Usage:  python -m eval.evaluate --run raw_outputs --out raw_outputs/6_evaluation_and_tests
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import time_iou  # noqa: E402
from src import feedback as fb  # noqa: E402
from src.batch import output_path, reference_results_dir  # noqa: E402
from src import ml  # noqa: E402
from src.candidates import make_segment  # noqa: E402
from src.cliffhanger import detect_cliffhanger  # noqa: E402
from src.context import ShowContext  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingest import discover_shows, parse_episode_file  # noqa: E402


def automated(run_dir: Path) -> dict:
    man = json.loads(output_path(run_dir, "run_manifest.json").read_text(encoding="utf-8"))
    rec = json.loads(output_path(run_dir, "recommendations.json").read_text(encoding="utf-8"))["promos"]
    val = json.loads(output_path(run_dir, "validation.json").read_text(encoding="utf-8"))
    perf = json.loads(output_path(run_dir, "performance.json").read_text(encoding="utf-8"))
    s = man["stats"]["summary"]
    n = s["candidates_total"] or 1
    k = len(rec) or 1
    pairs = k * (k - 1) / 2 or 1
    return {
        "episodes_discovered": s["episodes_discovered"],
        "episodes_failed": s["episodes_failed"],
        "candidates": s["candidates_total"],
        "candidate_valid_duration_rate": round(s["passed_duration"] / n, 4),
        "candidate_valid_hook_rate": round(s["valid_hook"] / n, 4),
        "candidate_valid_cliffhanger_rate": round(s["valid_cliffhanger"] / n, 4),
        "candidate_poor_coherence_rate": round(s["rejected_poor_coherence"] / n, 4),
        "candidate_pass_all_gates_rate": round(s["passed_all_gates"] / n, 4),
        "recommended": len(rec),
        "recommended_valid_duration_rate": round(sum(1 for p in rec if 20 <= p["total_duration_seconds"] <= 60) / k, 4),
        "recommended_timestamp_validity_rate": round(val["promos_valid"] / max(1, val["promos_checked"]), 4),
        "recommended_duplicate_pair_rate": round(val["duplicate_or_overlapping_pairs"] / pairs, 4),
        "recommended_hook_gate_rate": round(sum(1 for p in rec if p["opening_hook"]["score"] >= 0.45) / k, 4),
        "recommended_cliffhanger_gate_rate": round(sum(1 for p in rec if p["ending_cliffhanger"]["score"] >= 0.5) / k, 4),
        "recommended_mean_coherence": round(statistics.mean(p["standalone_coherence"]["score"] for p in rec), 4) if rec else None,
        "processing_seconds_total": perf["timings_seconds"].get("total"),
        "processing_seconds_per_episode": perf["seconds_per_episode_total"],
        "timings_seconds": perf["timings_seconds"],
    }


def _cut_score(ctx, eid, j):
    lines = ctx.content[eid]
    i = j
    while i > 0 and lines[j].end - lines[i - 1].start <= 6.0:
        i -= 1
    return detect_cliffhanger(make_segment(ctx, eid, i, j), ctx).score


def _proxy(pairs: list) -> dict:
    """pairs: [(final_score, [other_scores])] per episode."""
    aucs = [sum(1.0 if f > o else 0.5 if f == o else 0.0 for o in rest) / len(rest) for f, rest in pairs if rest]
    top = [sum(1 for o in rest if o > f) / len(rest) <= 0.2 for f, rest in pairs if rest]
    return {
        "episodes": len(aucs),
        "mean_auc_final_line_vs_other_cuts": round(statistics.mean(aucs), 4) if aucs else None,
        "final_line_in_top_20pct_of_cuts": round(sum(top) / len(top), 4) if top else None,
        "mean_score_final_lines": round(statistics.mean(f for f, _ in pairs), 4) if pairs else None,
        "mean_score_other_lines": round(statistics.mean(o for _, r in pairs for o in r), 4) if pairs else None,
    }


def silver(dataset: Path, cache_dir: Path, use_zeroshot: bool = True, sample_per_episode: int = 8) -> dict:
    from src.features import add_zeroshot
    shows = discover_shows(dataset)
    stage1, stage2 = [], []
    arousal, tags = [], []
    for show, files in shows.items():
        eps = [parse_episode_file(f, show) for f in files]
        eps = [e for e in eps if e.content]
        feats = build_features(eps, emotion_cache=ml.DiskCache(cache_dir / "emotion_cache.json"), use_emotion=True)
        ctx = ShowContext(show, eps, feats)
        sel = {}
        for e in eps:
            lines = ctx.content[e.episode_id]
            dl = [j for j, u in enumerate(lines) if u.kind == "dialogue"]
            if len(dl) < 2:
                continue
            fin = dl[-1]  # final DIALOGUE line = the writers' episode-ending beat
            others = [j for j in dl[:-1]]
            step = max(1, len(others) // sample_per_episode)
            sel[e.episode_id] = (fin, others[::step][:sample_per_episode])  # fixed, evenly spaced sample
            for u in lines:
                f = ctx.f(u)
                if u.kind == "dialogue" and f.arousal is not None and u.emotion_tag:
                    arousal.append(f.arousal)
                    tags.append(f.tag_intensity)
        for eid, (fin, others) in sel.items():
            stage1.append((_cut_score(ctx, eid, fin), [_cut_score(ctx, eid, j) for j in others]))
        if use_zeroshot:
            need = []
            for eid, (fin, others) in sel.items():
                lines = ctx.content[eid]
                for j in [fin] + others:
                    need += [u for u in lines if lines[j].end - 4.0 < u.end <= lines[j].end]
            add_zeroshot(feats, need, cache=ml.DiskCache(cache_dir / "zeroshot_cache.json"))
            for eid, (fin, others) in sel.items():
                stage2.append((_cut_score(ctx, eid, fin), [_cut_score(ctx, eid, j) for j in others]))
    corr = None
    if len(arousal) > 2 and statistics.pstdev(arousal) > 0 and statistics.pstdev(tags) > 0:
        corr = round(statistics.correlation(arousal, tags), 4)
    return {
        "note": "Silver proxies derived from script structure; NOT human evaluation.",
        "cliffhanger_proxy": {
            "method": f"per episode: final dialogue line vs {sample_per_episode} evenly spaced other dialogue lines as cut points",
            "interpretation": "AUC 0.5 = no better than chance at preferring the writers' episode-ending line.",
            "stage1_lexical_emotion": _proxy(stage1),
            "stage2_with_zeroshot": _proxy(stage2) if use_zeroshot else "not run",
        },
        "emotion_model_vs_script_tags": {
            "dialogue_lines": len(arousal),
            "pearson_r_arousal_vs_tag_intensity": corr,
        },
    }


def human(run_dir: Path, feedback_file: Path, labels_dir: Path) -> dict:
    out = {}
    entries = fb.load(feedback_file)
    out["editor_feedback"] = fb.summary(entries) if entries else "no editor decisions recorded"
    labels = []
    for f in sorted(labels_dir.glob("*.json")) if labels_dir.exists() else []:
        try:
            labels += json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not labels:
        out["labelled_promos"] = "no human-labelled promos found in eval/labels/ — recall@K, hook/cliffhanger precision not measured"
        return out
    rec = json.loads(output_path(run_dir, "recommendations.json").read_text(encoding="utf-8"))["promos"]
    good = [l for l in labels if l.get("grade", 0) >= 1]

    def match(p, l):
        return any(s["episode_id"] == l["episode_id"] and time_iou(s["start_seconds"], s["end_seconds"], l["start"], l["end"]) >= 0.5
                   for s in p["segments"])
    res = {}
    for k in (5, 10, len(rec)):
        top = rec[:k]
        res[f"recall@{k}"] = round(sum(1 for l in good if any(match(p, l) for p in top)) / max(1, len(good)), 4)
    hook_lab = [l for l in labels if "hook_ok" in l]
    cliff_lab = [l for l in labels if "cliffhanger_ok" in l]
    res["hook_precision"] = round(sum(1 for l in hook_lab if l["hook_ok"]) / len(hook_lab), 4) if hook_lab else None
    res["cliffhanger_precision"] = round(sum(1 for l in cliff_lab if l["cliffhanger_ok"]) / len(cliff_lab), 4) if cliff_lab else None
    res["labels"] = len(labels)
    out["labelled_promos"] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(reference_results_dir(ROOT)))
    ap.add_argument("--dataset", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / ".cache"))
    ap.add_argument("--feedback", default=str(ROOT / "feedback" / "editor_feedback.jsonl"))
    ap.add_argument("--labels", default=str(ROOT / "eval" / "labels"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_dir = Path(a.run)
    report = {"automated_checks": automated(run_dir), "silver_proxy_checks": silver(Path(a.dataset), Path(a.cache)),
              "human_evaluation": human(run_dir, Path(a.feedback), Path(a.labels))}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "evaluation_summary.json").write_text(text, encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
