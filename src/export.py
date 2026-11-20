"""Exports: JSON (full detail), CSV (one row per segment) and a readable Markdown report."""

from __future__ import annotations

import csv
import io
import json

from src.models import PromoCandidate, fmt_ts

SEGMENT_COLUMNS = [
    "rank", "promo_id", "show", "structure", "priority_score", "total_duration_seconds", "episodes",
    "segment_order", "episode_id", "episode_number", "segment_start", "segment_end", "segment_start_seconds",
    "segment_end_seconds", "segment_duration_seconds", "is_hook_segment", "is_cliffhanger_segment",
    "hook_timestamp", "hook_score", "hook_reason", "cliffhanger_timestamp", "cliffhanger_score",
    "cliffhanger_reason", "coherence_score", "coherence_summary", "emotion_score", "continuity_score",
    "dialogue_quality_score", "why_it_works", "risk_flags", "llm_verdict", "segment_dialogue",
]


def promos_to_json(promos: list, meta: dict = None) -> str:
    return json.dumps({"meta": meta or {}, "promos": [p.to_dict() for p in promos]}, indent=2, ensure_ascii=False)


def promo_rows(p: PromoCandidate) -> list:
    rows = []
    llm = p.llm_review or {}
    verdict = ""
    if llm.get("status") == "ok":
        verdict = (f"hook={llm.get('hook_works')} cliff={llm.get('cliffhanger_works')} "
                   f"standalone={llm.get('standalone')} | {llm.get('one_line', '')}")
    for k, s in enumerate(p.segments, 1):
        rows.append({
            "rank": p.rank, "promo_id": p.promo_id, "show": p.show, "structure": p.structure,
            "priority_score": round(p.score, 4), "total_duration_seconds": p.total_duration,
            "episodes": " + ".join(p.episodes), "segment_order": k, "episode_id": s.episode_id,
            "episode_number": s.episode_number, "segment_start": fmt_ts(s.start), "segment_end": fmt_ts(s.end),
            "segment_start_seconds": round(s.start, 2), "segment_end_seconds": round(s.end, 2),
            "segment_duration_seconds": s.duration, "is_hook_segment": k == 1,
            "is_cliffhanger_segment": k == len(p.segments),
            "hook_timestamp": fmt_ts(p.hook.timestamp), "hook_score": round(p.hook.score, 4),
            "hook_reason": p.hook.evidence[0], "cliffhanger_timestamp": fmt_ts(p.cliffhanger.timestamp),
            "cliffhanger_score": round(p.cliffhanger.score, 4), "cliffhanger_reason": p.cliffhanger.evidence[0],
            "coherence_score": round(p.coherence.score, 4), "coherence_summary": p.coherence.evidence[0],
            "emotion_score": round(p.emotion.score, 4), "continuity_score": round(p.continuity.score, 4),
            "dialogue_quality_score": round(p.dialogue_quality.score, 4), "why_it_works": p.explanation,
            "risk_flags": " | ".join(p.risk_flags), "llm_verdict": verdict,
            "segment_dialogue": " / ".join(f"{u.character}: {u.text}" for u in s.lines),
        })
    return rows


def promos_to_csv(promos: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SEGMENT_COLUMNS)
    w.writeheader()
    for p in promos:
        for r in promo_rows(p):
            w.writerow(r)
    return buf.getvalue()


def promos_to_markdown(promos: list, title: str = "Ranked promo recommendations", near_misses: list = None,
                       meta: dict = None) -> str:
    out = [f"# {title}", ""]
    if meta:
        out.append(f"Show(s): {', '.join(meta.get('shows', []))} · episodes processed: {meta.get('episodes_processed', '?')} · "
                   f"generated: {meta.get('generated_at', '')}")
        out.append("")
    out.append("Scores are heuristic promo-priority scores (0–1), not predictions of views, CTR or virality.")
    out.append("")
    for p in promos:
        out.append(f"## #{p.rank} · score {p.score:.3f} · {p.total_duration:.1f}s · {p.structure.replace('_', ' ')}")
        out.append("")
        out.append(f"**Episodes:** {', '.join(p.episodes)}  ")
        out.append(f"**Opening hook:** {fmt_ts(p.hook.timestamp)} — {p.hook.evidence[0]} — “{p.hook.text}”  ")
        out.append(f"**Ending cliffhanger:** {fmt_ts(p.cliffhanger.timestamp)} — {p.cliffhanger.evidence[0]} — “{p.cliffhanger.text}”  ")
        out.append(f"**Standalone coherence:** {p.coherence.score:.2f} — {p.coherence.evidence[0]}  ")
        out.append(f"**Why it works:** {p.explanation}")
        out.append("")
        out.append("| # | Episode | Start | End | Duration | Dialogue |")
        out.append("|---|---|---|---|---|---|")
        for k, s in enumerate(p.segments, 1):
            tag = " (hook)" if k == 1 else ""
            tag += " (cliffhanger)" if k == len(p.segments) else ""
            dia = " / ".join(f"{u.character}: {u.text}" for u in s.lines).replace("|", "/")
            out.append(f"| {k}{tag} | {s.episode_number} | {fmt_ts(s.start)} | {fmt_ts(s.end)} | {s.duration:.1f}s | {dia} |")
        b = p.components()
        out.append("")
        out.append("Score breakdown: " + ", ".join(f"{k} {v:.2f}" for k, v in b.items()))
        if p.risk_flags:
            out.append("")
            out.append("Risk flags: " + "; ".join(p.risk_flags))
        if p.llm_review and p.llm_review.get("status") == "ok":
            r = p.llm_review
            out.append("")
            out.append(f"Gemini review ({r.get('model')}): hook={r.get('hook_works')}, cliffhanger={r.get('cliffhanger_works')}, "
                       f"standalone={r.get('standalone')} — {r.get('one_line')} Concern: {r.get('concerns') or 'none'}")
        out.append("")
    if near_misses:
        out.append("## Near misses (high score, failed a quality gate)")
        out.append("")
        for p in near_misses:
            out.append(f"- {p.promo_id} · {', '.join(p.episodes)} · {fmt_ts(p.segments[0].start)}→{fmt_ts(p.segments[-1].end)} · "
                       f"score {p.score:.3f} · rejected: {'; '.join(p.rejection_reasons)}")
        out.append("")
    return "\n".join(out)
