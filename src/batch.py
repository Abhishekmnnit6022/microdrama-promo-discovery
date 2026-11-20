"""Show-level batch processing: discover → parse (cached) → per-line ML features (cached, model loaded once)
→ candidate generation → stage-1 scoring → stage-2 zero-shot re-scoring of a shortlist → diverse selection
→ optional Gemini review of the final list → outputs.

Failures are isolated per episode: a broken file is recorded with its traceback and the run continues.
Inference caches are flushed after every episode, so a crashed or interrupted run resumes without
re-running the models on episodes already processed.
"""

from __future__ import annotations

import json
import logging
import pickle
import platform
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from src import ml
from src.candidates import GenConfig, continuous_candidates, ending_lines, hook_beats, multi_candidates
from src.context import ShowContext
from src.export import promos_to_csv, promos_to_json, promos_to_markdown
from src.features import add_zeroshot, build_features
from src.ingest import PARSER_VERSION, discover_shows, parse_episode_file
from src.llm_review import review_promos
from src.models import fmt_ts
from src.ranking import Gates, Scorer, Weights, select_diverse

logger = logging.getLogger("promo")
PIPELINE_VERSION = "3.0.0"


@dataclass
class RunConfig:
    dataset_dir: str = "data"
    output_dir: Optional[str] = None
    cache_dir: str = ".cache"
    top_k: int = 15
    max_per_episode: int = 2
    max_overlap: float = 0.3
    use_emotion: bool = True
    use_zeroshot: bool = True
    use_llm: bool = False
    api_key: str = ""
    shortlist_per_episode: int = 25
    shortlist_multi: int = 300
    weights: Weights = field(default_factory=Weights)
    gates: Gates = field(default_factory=Gates)
    gen: GenConfig = field(default_factory=GenConfig)
    shows: Optional[list] = None      # restrict to these show names
    max_episodes: Optional[int] = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("api_key", None)
        d["llm_key_configured"] = bool(self.api_key)
        return d


@dataclass
class EpisodeStatus:
    show: str
    episode_id: str
    episode_number: Optional[int]
    source_path: str
    status: str = "pending"         # ok | failed | no_valid_promo
    error: str = ""
    traceback: str = ""
    warnings: list = field(default_factory=list)
    lines_total: int = 0
    dialogue_lines: int = 0
    action_lines: int = 0
    untimed_lines: int = 0
    duration_seconds: float = 0.0
    candidates: int = 0
    candidates_passed: int = 0
    best_promo_id: str = ""
    best_score: Optional[float] = None
    no_promo_reason: str = ""
    seconds_features: float = 0.0
    seconds_candidates: float = 0.0
    logline: str = ""


@dataclass
class RunResult:
    config: dict
    shows: dict = field(default_factory=dict)            # show -> [episode ids]
    episodes: list = field(default_factory=list)         # [EpisodeStatus]
    candidates: list = field(default_factory=list)       # all scored PromoCandidates
    recommended: list = field(default_factory=list)      # global ranked list
    recommended_by_show: dict = field(default_factory=dict)
    episode_best: dict = field(default_factory=dict)     # episode_id -> PromoCandidate | None
    alternatives: dict = field(default_factory=dict)     # structure -> diverse list (best multi-segment / cross-episode)
    near_misses: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    llm_summary: dict = field(default_factory=dict)
    feats: dict = field(default_factory=dict)
    contexts: dict = field(default_factory=dict)
    generated_at: str = ""


def _parse_cached(path: Path, show: str, cache_dir: Path):
    from src.ingest import file_hash
    h = file_hash(path)
    cp = cache_dir / "parsed" / f"{h}-v{PARSER_VERSION}.pkl"
    if cp.exists():
        try:
            ep = pickle.loads(cp.read_bytes())
            if ep.source_path == str(path) and ep.show == show:
                return ep, True
        except Exception:
            pass
    ep = parse_episode_file(path, show)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps(ep))
    return ep, False


def run(cfg: RunConfig, progress: Optional[Callable[[float, str], None]] = None) -> RunResult:
    t_run = time.perf_counter()
    prog = progress or (lambda p, m: None)
    res = RunResult(config=cfg.public(), generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    cache_dir = Path(cfg.cache_dir)
    emo_cache = ml.DiskCache(cache_dir / "emotion_cache.json")
    zs_cache = ml.DiskCache(cache_dir / "zeroshot_cache.json")

    shows = discover_shows(Path(cfg.dataset_dir))
    if cfg.shows:
        shows = {k: v for k, v in shows.items() if k in cfg.shows}
    logger.info("Discovered %d show(s) in %s", len(shows), cfg.dataset_dir)
    if not shows:
        res.stats = {"error": f"No transcripts (.txt/.srt/.json) found under {cfg.dataset_dir}"}
        logger.error(res.stats["error"])
        return res

    timings = Counter()
    all_cands = []
    total_files = sum(len(v) if cfg.max_episodes is None else min(cfg.max_episodes, len(v)) for v in shows.values())
    done = 0
    for show, files in shows.items():
        if cfg.max_episodes:
            files = files[: cfg.max_episodes]
        # ── 1. parse ──────────────────────────────────────────────────────
        t = time.perf_counter()
        episodes, statuses = [], {}
        for f in files:
            st = EpisodeStatus(show=show, episode_id=f.stem, episode_number=None, source_path=str(f))
            try:
                ep, cached = _parse_cached(f, show, cache_dir)
                st.episode_number = ep.number
                st.warnings = list(ep.warnings)
                st.lines_total = len(ep.utterances)
                st.dialogue_lines = sum(1 for u in ep.utterances if u.kind == "dialogue")
                st.action_lines = sum(1 for u in ep.utterances if u.kind == "action")
                st.untimed_lines = sum(1 for u in ep.utterances if not u.timed)
                st.duration_seconds = round(ep.duration, 2)
                st.logline = ep.logline
                if not ep.content:
                    raise ValueError("no timed dialogue/action lines could be parsed")
                episodes.append(ep)
                st.status = "parsed"
                logger.info("[%s] parsed %s: %d lines (%d dialogue, %d untimed)%s", show, f.name, st.lines_total,
                            st.dialogue_lines, st.untimed_lines, " [cache]" if cached else "")
            except Exception as exc:
                st.status = "failed"
                st.error = f"parse: {type(exc).__name__}: {exc}"
                st.traceback = traceback.format_exc()
                logger.error("[%s] FAILED to parse %s: %s", show, f.name, exc)
            statuses[st.episode_id] = st
        timings["parse"] += time.perf_counter() - t
        res.shows[show] = [e.episode_id for e in episodes]

        # ── 2. per-line features (emotion model once, cached per episode) ─
        feats = {}
        for ep in episodes:
            done += 1
            prog(0.05 + 0.55 * done / max(1, total_files), f"{show}: emotion features for episode {ep.number or ep.episode_id}")
            t = time.perf_counter()
            try:
                feats.update(build_features([ep], emotion_cache=emo_cache, use_emotion=cfg.use_emotion))
                emo_cache.save()
            except Exception as exc:
                st = statuses[ep.episode_id]
                st.status, st.error, st.traceback = "failed", f"features: {type(exc).__name__}: {exc}", traceback.format_exc()
                logger.error("[%s] feature extraction failed for %s: %s", show, ep.episode_id, exc)
            dt = time.perf_counter() - t
            statuses[ep.episode_id].seconds_features = round(dt, 3)
            timings["features_emotion"] += dt
        episodes = [e for e in episodes if statuses[e.episode_id].status != "failed"]
        ctx = ShowContext(show, episodes, feats)
        scorer = Scorer(ctx, cfg.weights, cfg.gates)
        res.contexts[show] = ctx

        # ── 3. candidates + stage-1 scoring per episode ───────────────────
        prog(0.62, f"{show}: generating and scoring candidates")
        beats, endings, by_ep = {}, {}, {}
        gen_stats: dict = {}
        for ep in episodes:
            st = statuses[ep.episode_id]
            t = time.perf_counter()
            try:
                cands = continuous_candidates(ctx, ep.episode_id, cfg.gen)
                for c in cands:
                    scorer.score(c)
                beats[ep.episode_id] = hook_beats(ctx, ep.episode_id, cfg.gen)
                endings[ep.episode_id] = ending_lines(ctx, ep.episode_id, cfg.gen)
                by_ep[ep.episode_id] = cands
                st.status = "ok"
            except Exception as exc:
                st.status, st.error, st.traceback = "failed", f"candidates: {type(exc).__name__}: {exc}", traceback.format_exc()
                logger.error("[%s] candidate generation failed for %s: %s", show, ep.episode_id, exc)
                by_ep[ep.episode_id] = []
            dt = time.perf_counter() - t
            st.seconds_candidates = round(dt, 3)
            timings["candidates_continuous"] += dt
        t = time.perf_counter()
        multi = multi_candidates(ctx, beats, endings, cfg.gen, gen_stats)
        for c in multi:
            scorer.score(c)
        timings["candidates_multi"] += time.perf_counter() - t
        show_cands = [c for v in by_ep.values() for c in v] + multi
        logger.info("[%s] stage 1: %d candidates (%d continuous, %d multi-segment, %d cross-episode); %d unrelated pairs discarded",
                    show, len(show_cands), sum(len(v) for v in by_ep.values()),
                    sum(1 for c in multi if c.structure == "multi_segment"),
                    sum(1 for c in multi if c.structure == "cross_episode"), gen_stats.get("rejected_unrelated_pairs", 0))

        # ── 4. stage 2: zero-shot NLI on the shortlist's boundary lines ───
        t = time.perf_counter()
        shortlist = []
        for eid, cands in by_ep.items():
            shortlist += sorted(cands, key=lambda c: -c.score)[: cfg.shortlist_per_episode]
        shortlist += sorted(multi, key=lambda c: -c.score)[: cfg.shortlist_multi]
        boundary = []
        for c in shortlist:
            s0, s1 = c.segments[0], c.segments[-1]
            boundary += [u for u in s0.lines if u.start < s0.start + 3.0]
            boundary += [u for u in s1.lines if u.end > s1.end - 4.0]
        boundary = list({(u.episode_id, u.index): u for u in boundary}.values())
        prog(0.75, f"{show}: zero-shot NLI on {len(boundary)} boundary lines")
        n_zs = add_zeroshot(feats, boundary, cache=zs_cache, enabled=cfg.use_zeroshot) if cfg.use_zeroshot else 0
        zs_cache.save()
        scorer.clear_cache()
        for c in shortlist:
            scorer.score(c)
            c.stage = 2  # type: ignore[attr-defined]
        timings["stage2_zeroshot"] += time.perf_counter() - t
        logger.info("[%s] stage 2: re-scored %d shortlisted candidates; zero-shot scores for %d/%d boundary lines",
                    show, len(shortlist), n_zs, len(boundary))

        # stage-2 scores are only comparable among stage-2 candidates -> recommend only from the shortlist
        for c in show_cands:
            if not hasattr(c, "stage"):
                c.stage = 1  # type: ignore[attr-defined]
        res.recommended_by_show[show] = select_diverse(shortlist, cfg.top_k, cfg.max_per_episode, cfg.max_overlap)
        for structure in ("continuous", "multi_segment", "cross_episode"):
            alt = select_diverse([c for c in shortlist if c.structure == structure], 5, 1, cfg.max_overlap, assign_rank=False)
            res.alternatives.setdefault(structure, []).extend(alt)
        for eid in by_ep:
            st = statuses[eid]
            own = [c for c in show_cands if eid in c.episodes]
            st.candidates = len(own)
            st.candidates_passed = sum(1 for c in own if c.passed)
            ranked = sorted((c for c in shortlist if c.passed and c.episodes == [eid]), key=lambda c: -c.score)
            best = ranked[0] if ranked else None
            res.episode_best[eid] = best
            if best is not None:
                st.best_promo_id, st.best_score = best.promo_id, round(best.score, 4)
            elif st.status == "ok":
                st.status = "no_valid_promo"
                reasons = Counter(r.split(" (")[0] for c in own for r in c.rejection_reasons)
                st.no_promo_reason = "; ".join(f"{k}: {v}/{len(own)} candidates" for k, v in reasons.most_common(3)) or "no candidates generated"
                logger.warning("[%s] %s: no valid promo (%s)", show, eid, st.no_promo_reason)
        all_cands += show_cands
        res.episodes += list(statuses.values())
        res.feats.update(feats)

    # ── 5. global ranking across shows, near misses ─────────────────────
    pool = [c for v in res.recommended_by_show.values() for c in v]
    res.recommended = sorted(pool, key=lambda c: -c.score)[: cfg.top_k]
    for i, c in enumerate(res.recommended, 1):
        c.rank = i
    chosen_ids = {c.promo_id for c in res.recommended}
    stage2 = [c for c in all_cands if getattr(c, "stage", 1) == 2]
    res.near_misses = sorted((c for c in stage2 if not c.passed and c.promo_id not in chosen_ids), key=lambda c: -c.score)[:15]
    res.candidates = all_cands

    # ── 6. optional LLM second opinion on the final list only ───────────
    if cfg.use_llm:
        prog(0.92, f"Gemini review of {len(res.recommended)} final promos")
        t = time.perf_counter()
        res.llm_summary = review_promos(res.recommended, cfg.api_key)
        timings["llm_review"] += time.perf_counter() - t
        logger.info("Gemini review: %s", res.llm_summary)
    else:
        res.llm_summary = {"status": "disabled"}

    timings["total"] = time.perf_counter() - t_run
    res.timings = {k: round(v, 3) for k, v in timings.items()}
    res.models = ml.model_status()
    res.stats = compute_stats(res)
    prog(1.0, "Done")
    logger.info("Run finished in %.1fs: %s", timings["total"], json.dumps(res.stats["summary"]))
    return res


def compute_stats(res: RunResult) -> dict:
    cands = res.candidates
    g = Gates(**{k: v for k, v in res.config["gates"].items()})
    n = len(cands)
    by_struct = Counter(c.structure for c in cands)
    dur_ok = sum(1 for c in cands if g.min_duration - 1e-6 <= c.total_duration <= g.max_duration + 1e-6)
    hook_ok = sum(1 for c in cands if c.hook.score >= g.hook)
    cliff_ok = sum(1 for c in cands if c.cliffhanger.score >= g.cliffhanger and getattr(c.cliffhanger, "open_signal", 0) >= g.cliff_open_signal)
    coh_bad = sum(1 for c in cands if c.coherence.score < g.coherence)
    passed = sum(1 for c in cands if c.passed)
    reasons = Counter(r.split(" (")[0] for c in cands for r in c.rejection_reasons)
    eps = res.episodes
    summary = {
        "shows": len(res.shows),
        "episodes_discovered": len(eps),
        "episodes_processed_ok": sum(1 for e in eps if e.status in ("ok", "no_valid_promo")),
        "episodes_failed": sum(1 for e in eps if e.status == "failed"),
        "episodes_without_valid_promo": sum(1 for e in eps if e.status == "no_valid_promo"),
        "candidates_total": n,
        "candidates_continuous": by_struct.get("continuous", 0),
        "candidates_multi_segment": by_struct.get("multi_segment", 0),
        "candidates_cross_episode": by_struct.get("cross_episode", 0),
        "passed_duration": dur_ok,
        "valid_hook": hook_ok,
        "valid_cliffhanger": cliff_ok,
        "rejected_poor_coherence": coh_bad,
        "passed_all_gates": passed,
        "stage2_shortlisted": sum(1 for c in cands if getattr(c, "stage", 1) == 2),
        "stage2_passed": sum(1 for c in cands if getattr(c, "stage", 1) == 2 and c.passed),
        "recommended": len(res.recommended),
    }
    rec = res.recommended
    validation = validate_promos(rec, res)
    return {
        "summary": summary,
        "rejection_reason_counts": dict(reasons.most_common()),
        "recommended_structure_counts": dict(Counter(c.structure for c in rec)),
        "recommended_duration": {
            "min": min((c.total_duration for c in rec), default=None),
            "max": max((c.total_duration for c in rec), default=None),
            "mean": round(sum(c.total_duration for c in rec) / len(rec), 2) if rec else None,
        },
        "validation": validation,
    }


def validate_promos(promos: list, res: RunResult) -> dict:
    """Automated checks on final promos: durations, timestamps traceable to source lines, story order, overlap."""
    checks = []
    for c in promos:
        problems = []
        if not (20.0 - 1e-6 <= c.total_duration <= 60.0 + 1e-6):
            problems.append(f"duration {c.total_duration}")
        if abs(sum(s.duration for s in c.segments) - c.total_duration) > 0.02:
            problems.append("duration sum mismatch")
        ctx = res.contexts.get(c.show)
        prev_key = None
        for s in c.segments:
            if not s.lines or s.start != s.lines[0].start or abs(s.end - max(u.end for u in s.lines)) > 1e-6:
                problems.append(f"segment {s.episode_id} bounds not equal to source line timestamps")
            if any(u.episode_id != s.episode_id for u in s.lines):
                problems.append("segment mixes episodes")
            if any(not u.timed for u in s.lines):
                problems.append("untimed line in segment")
            key = (ctx.order[s.episode_id] if ctx else 0, s.start)
            if prev_key is not None and key <= prev_key:
                problems.append("segments not in story order")
            prev_key = key
        if c.hook.timestamp is None or not (c.segments[0].start <= c.hook.timestamp <= c.segments[0].start + 3.0):
            problems.append("hook timestamp outside first 3 s")
        if c.cliffhanger.timestamp is None or abs(c.cliffhanger.timestamp - c.segments[-1].end) > 1e-6:
            problems.append("cliffhanger timestamp is not the cut point")
        checks.append({"promo_id": c.promo_id, "rank": c.rank, "ok": not problems, "problems": problems})
    from src.ranking import conflicts
    dup_pairs = sum(1 for i, a in enumerate(promos) for b in promos[i + 1:] if conflicts(a, b))
    return {
        "promos_checked": len(checks),
        "promos_valid": sum(1 for x in checks if x["ok"]),
        "duplicate_or_overlapping_pairs": dup_pairs,
        "details": checks,
    }


# ── output folder layout ──────────────────────────────────────────────────
# Every output file has a fixed place, grouped so a reviewer can read the folder top to bottom.
OUTPUT_LAYOUT = {
    "recommendations.json": "1_recommendations/recommendations.json",
    "recommendations.csv": "1_recommendations/recommendations.csv",
    "recommendations_report.md": "1_recommendations/recommendations_report.md",
    "alternatives_by_structure.json": "2_other_candidates/alternatives_by_structure.json",
    "episode_best.json": "2_other_candidates/episode_best.json",
    "near_misses.json": "2_other_candidates/near_misses.json",
    "candidates_all.csv": "2_other_candidates/candidates_all.csv",
    "episodes/": "3_per_episode/",
    "model_outputs/": "4_model_outputs/",
    "run_log.txt": "5_run_logs/run_log.txt",
    "run_manifest.json": "5_run_logs/run_manifest.json",
    "performance.json": "5_run_logs/performance.json",
    "errors.jsonl": "5_run_logs/errors.jsonl",
    "validation.json": "6_evaluation_and_tests/validation.json",
}


def reference_results_dir(project_root) -> Path:
    """The submitted full-dataset run: raw_outputs/ inside the project folder (or next to it)."""
    project_root = Path(project_root)
    for p in (project_root / "raw_outputs", project_root.parent / "raw_outputs"):
        if p.exists():
            return p
    return project_root / "raw_outputs"


def output_path(out_dir, name: str) -> Path:
    """Where a named output file lives inside a results folder (e.g. 'run_manifest.json')."""
    for key, rel in OUTPUT_LAYOUT.items():
        if key.endswith("/") and name.startswith(key):
            return Path(out_dir) / (rel + name[len(key):])
        if name == key:
            return Path(out_dir) / rel
    return Path(out_dir) / name


# ── writing outputs ───────────────────────────────────────────────────────
def write_outputs(res: RunResult, out_dir: Path) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def w(name: str, text: str):
        p = output_path(out, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        written.append(str(p))

    meta = {"shows": list(res.shows), "episodes_processed": res.stats["summary"]["episodes_processed_ok"],
            "generated_at": res.generated_at, "pipeline_version": PIPELINE_VERSION,
            "weights": res.config["weights"], "gates": res.config["gates"],
            "note": "priority_score is a heuristic promo-priority score, not a prediction of views/CTR/virality"}
    w("recommendations.json", promos_to_json(res.recommended, meta))
    w("recommendations.csv", promos_to_csv(res.recommended))
    w("recommendations_report.md", promos_to_markdown(res.recommended, near_misses=res.near_misses, meta=meta))
    w("alternatives_by_structure.json", json.dumps(
        {k: [c.to_dict() for c in sorted(v, key=lambda c: -c.score)] for k, v in res.alternatives.items()},
        indent=2, ensure_ascii=False))
    w("near_misses.json", promos_to_json(res.near_misses, {"description": "highest-scoring stage-2 candidates that failed a quality gate"}))
    best = [c for c in res.episode_best.values() if c is not None]
    w("episode_best.json", promos_to_json(sorted(best, key=lambda c: (c.segments[0].episode_number or 0)),
                                          {"description": "best valid single-episode promo per episode (stage-2 scores)"}))
    # all candidates (compact)
    import csv
    import io
    buf = io.StringIO()
    cols = ["promo_id", "show", "structure", "stage", "episodes", "segments", "total_duration", "priority_score",
            "hook", "cliffhanger", "cliff_open_signal", "coherence", "emotion", "continuity", "dialogue_quality",
            "passed", "rejection_reasons", "hook_reason", "cliffhanger_reason"]
    wr = csv.writer(buf)
    wr.writerow(cols)
    for c in res.candidates:
        comp = c.components()
        wr.writerow([c.promo_id, c.show, c.structure, getattr(c, "stage", 1), " + ".join(c.episodes),
                     " + ".join(f"{s.episode_number}:{fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in c.segments),
                     c.total_duration, round(c.score, 4), round(comp["hook"], 3), round(comp["cliffhanger"], 3),
                     round(getattr(c.cliffhanger, "open_signal", 0), 3), round(comp["coherence"], 3),
                     round(comp["emotion"], 3), round(comp["continuity"], 3), round(comp["dialogue_quality"], 3),
                     c.passed, " | ".join(c.rejection_reasons), c.hook.evidence[0], c.cliffhanger.evidence[0]])
    w("candidates_all.csv", buf.getvalue())
    # per-episode
    for st in res.episodes:
        d = asdict(st)
        b = res.episode_best.get(st.episode_id)
        d["best_promo"] = b.to_dict() if b is not None else None
        w(f"episodes/{st.episode_id}.json", json.dumps(d, indent=2, ensure_ascii=False))
    # model outputs per line
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["episode_id", "line_index", "start", "end", "character", "kind", "script_tag", "text", "emotion_dominant",
                 "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise", "arousal", "lexical_cues",
                 "zs_threat", "zs_revelation", "zs_betrayal", "zs_violence", "zs_suspense", "zs_calm"])
    for show, ctx in res.contexts.items():
        for ep in ctx.episodes:
            for u in ep.content:
                f = ctx.f(u)
                e = f.emotions or {}
                z = f.zeroshot or {}
                wr.writerow([u.episode_id, u.index, fmt_ts(u.start), fmt_ts(u.end), u.character, u.kind, u.emotion_tag, u.text,
                             ml.dominant(f.emotions) if f.emotions else "",
                             *[e.get(k, "") for k in ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")],
                             "" if f.arousal is None else round(f.arousal, 4),
                             "; ".join(f"{k}:{','.join(v)}" for k, v in f.cues.items()),
                             *[z.get(k, "") for k in ("threat", "revelation", "betrayal", "violence", "suspense", "calm")]])
    w("model_outputs/line_features.csv", buf.getvalue())
    w("model_outputs/llm_review.json", json.dumps({"summary": res.llm_summary,
                                                   "reviews": [{"rank": c.rank, "promo_id": c.promo_id, **(c.llm_review or {})} for c in res.recommended]},
                                                  indent=2, ensure_ascii=False))
    errs = [asdict(e) for e in res.episodes if e.status == "failed" or e.warnings]
    w("errors.jsonl", "\n".join(json.dumps({k: e[k] for k in ("episode_id", "status", "error", "traceback", "warnings")}, ensure_ascii=False) for e in errs) + ("\n" if errs else ""))
    w("validation.json", json.dumps(res.stats["validation"], indent=2))
    perf = {
        "timings_seconds": res.timings,
        "episodes": len(res.episodes),
        "seconds_per_episode_total": round(res.timings.get("total", 0) / max(1, len(res.episodes)), 3),
        "per_episode": [{"episode_id": e.episode_id, "features_s": e.seconds_features, "candidates_s": e.seconds_candidates}
                        for e in res.episodes],
        "machine": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()},
    }
    w("performance.json", json.dumps(perf, indent=2))
    manifest = {
        "pipeline_version": PIPELINE_VERSION, "generated_at": res.generated_at, "config": res.config,
        "models": res.models, "llm_review": res.llm_summary, "shows": res.shows, "stats": {k: v for k, v in res.stats.items() if k != "validation"},
        "recommended_by_show": {s: [c.promo_id for c in v] for s, v in res.recommended_by_show.items()},
        "episode_status": [{k: v for k, v in asdict(e).items() if k not in ("traceback", "logline")} for e in res.episodes],
    }
    w("run_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return written


def load_outputs(out_dir: Path) -> dict:
    """Load a previous run's JSON outputs (used by the UI)."""
    out = Path(out_dir)
    data = {}
    for name in ("run_manifest", "recommendations", "near_misses", "episode_best", "validation", "alternatives_by_structure"):
        p = output_path(out, f"{name}.json")
        if p.exists():
            data[name] = json.loads(p.read_text(encoding="utf-8"))
    return data
