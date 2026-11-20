"""Opening-hook detector: does the first 3 seconds of the promo grab attention?

Evidence combined per opening line (lines that begin inside the first 3 s of the first segment):
  * ML  — emotion model arousal (anger + fear + surprise + disgust); zero-shot NLI when available (stage 2)
  * NLP — whole-word cue lexicon (physical / threat / betrayal / revelation / shock / arrival / discovery),
          exclamations and charged questions
  * Script — the writer's own emotion annotation, e.g. ``(Furious, Betrayed)``
Penalties: opening that starts mid-thought (pronoun/conjunction), filler opener, dead air before first line.
"""

from __future__ import annotations

from src.context import ShowContext
from src.models import ComponentResult, Segment, fmt_ts

HOOK_WINDOW = 3.0
ZS_HOOK = ("threat", "revelation", "betrayal", "violence")
CUE_LABEL = {
    "physical": "physical confrontation", "threat": "threat", "betrayal": "betrayal/accusation",
    "revelation": "revelation", "secret_setup": "secret about to come out", "shock": "shock reaction",
    "arrival": "sudden arrival", "discovery": "discovery",
}


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def score_line(u, f, offset: float) -> tuple:
    """Hook strength of one opening line -> (score, evidence list, dominant reason)."""
    cue = f.cue_strength
    reason = None
    if f.cues:
        cats = [c for c in f.cues if c != "stakes"]
        if cats:
            top = max(cats, key=lambda c: {"physical": 9, "threat": 8, "betrayal": 7, "secret_setup": 6,
                                           "revelation": 6, "shock": 5, "arrival": 4, "discovery": 3}.get(c, 0))
            reason = CUE_LABEL.get(top, top)
    arousal = f.ml_arousal()
    if u.kind == "dialogue" and f.question and cue < 0.55:
        charged = arousal > 0.4 or f.tag_intensity >= 0.66
        cue = max(cue, 0.55 if charged else 0.35)
        reason = reason or ("charged question" if charged else "question")
    if f.exclamation and cue < 0.45:
        cue = 0.45
        reason = reason or "exclamation"
    zs = None
    if f.zeroshot:
        zs = max(f.zeroshot.get(k, 0.0) for k in ZS_HOOK)
    if u.kind == "action":
        # no speech -> no emotion-model signal; the physical/visual cue carries the hook
        # zero-shot (stage 2) tempers ambiguous verbs, e.g. "strikes flint" vs "strikes the guard"
        s = 0.55 * cue + 0.45 * zs if zs is not None else 0.8 * cue
    elif zs is None:
        s = 0.40 * arousal + 0.40 * cue + 0.20 * f.tag_intensity
    else:
        s = 0.30 * arousal + 0.30 * cue + 0.15 * f.tag_intensity + 0.25 * zs
    ev = []
    if u.kind == "dialogue" and f.emotions:
        top_emo = max(((k, v) for k, v in f.emotions.items() if k != "neutral"), key=lambda kv: kv[1], default=("neutral", 0))
        ev.append(f"model: {top_emo[0]} {top_emo[1]:.2f}, arousal {arousal:.2f}")
    if f.cues:
        ev.append("cues: " + "; ".join(f"{c} ({', '.join(repr(h) for h in hs[:2])})" for c, hs in f.cues.items() if c != "stakes"))
    if f.tag_intensity:
        ev.append(f"script tag '{u.emotion_tag}' ({f.tag_intensity:.2f})")
    if zs is not None:
        ev.append(f"zero-shot conflict/reveal {zs:.2f}")
    penalties = []
    if f.dangling_start and offset < 0.5:
        s *= 0.75
        penalties.append("opens mid-thought (pronoun/conjunction)")
    if u.kind == "dialogue" and f.words <= 2 and f.cue_strength == 0 and not f.exclamation:
        s *= 0.8
        penalties.append("filler opener")
    if offset > 1.5:
        s *= 0.85
        penalties.append(f"first line starts {offset:.1f}s in")
    return max(0.0, min(1.0, s)), ev, reason, penalties


def detect_hook(seg: Segment, ctx: ShowContext) -> ComponentResult:
    t0 = seg.start
    opening = [u for u in seg.lines if u.start < t0 + HOOK_WINDOW] or seg.lines[:1]
    scored = []
    for u in opening:
        s, ev, reason, pen = score_line(u, ctx.f(u), u.start - t0)
        scored.append((s, u, ev, reason, pen))
    scored.sort(key=lambda x: -x[0])
    best_s, best_u, ev, reason, pen = scored[0]
    # a second strong beat inside the 3 s (e.g. slap + line) reinforces the hook
    bonus = 0.1 * scored[1][0] if len(scored) > 1 and scored[1][0] > 0.4 else 0.0
    score = min(1.0, best_s + bonus)
    evidence = [f"{fmt_ts(best_u.start)} {best_u.character}: “{_short(best_u.text)}”"] + ev
    if bonus:
        evidence.append(f"second strong beat within 3 s (+{bonus:.2f})")
    evidence += [f"penalty: {p}" for p in pen]
    label = reason or ("high-arousal line" if ctx.f(best_u).ml_arousal() > 0.5 else "weak opening")
    return ComponentResult(score=round(score, 4), evidence=[label] + evidence, timestamp=best_u.start,
                           episode_id=best_u.episode_id, text=best_u.text)
