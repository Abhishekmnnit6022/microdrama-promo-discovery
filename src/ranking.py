"""Explainable promo ranking.

priority = 0.25 hook + 0.25 cliffhanger + 0.20 coherence + 0.15 emotion + 0.10 continuity + 0.05 dialogue quality

The weights are starting hypotheses (not fitted to labels). Every component is already in [0, 1] by
construction, so the weighted sum is in [0, 1]. It is a heuristic *promo priority* score — NOT a
prediction of views, CTR or virality.

Quality gates (a candidate must pass all to be recommended):
  duration within [20, 60] s, all timestamps available, hook >= 0.45,
  cliffhanger >= 0.50 with a real open-loop signal at the cut, coherence >= 0.55.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from src.cliffhanger import detect_cliffhanger
from src.coherence import analyse_coherence
from src.context import ShowContext
from src.hook import detect_hook
from src.models import ComponentResult, PromoCandidate, fmt_ts


@dataclass
class Weights:
    hook: float = 0.25
    cliffhanger: float = 0.25
    coherence: float = 0.20
    emotion: float = 0.15
    continuity: float = 0.10
    dialogue_quality: float = 0.05

    def normalised(self) -> "Weights":
        tot = sum(asdict(self).values()) or 1.0
        return Weights(**{k: v / tot for k, v in asdict(self).items()})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Gates:
    min_duration: float = 20.0
    max_duration: float = 60.0
    hook: float = 0.45
    cliffhanger: float = 0.50
    cliff_open_signal: float = 0.50
    coherence: float = 0.55

    def to_dict(self) -> dict:
        return asdict(self)


class Scorer:
    """Scores candidates; caches hook/cliffhanger results shared by many candidates."""

    def __init__(self, ctx: ShowContext, weights: Weights = None, gates: Gates = None):
        self.ctx = ctx
        self.w = (weights or Weights()).normalised()
        self.g = gates or Gates()
        self._hook: dict = {}
        self._cliff: dict = {}

    def clear_cache(self):
        self._hook.clear()
        self._cliff.clear()

    def hook(self, seg) -> ComponentResult:
        k = (seg.episode_id, seg.lines[0].index)
        if k not in self._hook:
            self._hook[k] = detect_hook(seg, self.ctx)
        return self._hook[k]

    def cliff(self, seg) -> ComponentResult:
        final = seg.lines[-1]
        k = (seg.episode_id, final.index, seg.lines[0].index if seg.start > final.end - 6.0 else -1)
        if k not in self._cliff:
            self._cliff[k] = detect_cliffhanger(seg, self.ctx)
        return self._cliff[k]

    # ── secondary components ──────────────────────────────────────────────
    def emotion(self, c: PromoCandidate) -> ComponentResult:
        dl = [u for s in c.segments for u in s.lines if u.kind == "dialogue"]
        if not dl:
            return ComponentResult(0.0, ["no dialogue"])
        vals = [(self.ctx.f(u).ml_intensity(), max(0.3, u.duration), u) for u in dl]
        mean = sum(v * d for v, d, _ in vals) / sum(d for _, d, _ in vals)
        peak_v, _, peak_u = max(vals, key=lambda x: x[0])
        score = 0.5 * peak_v + 0.5 * mean
        src = "emotion model" if any(self.ctx.f(u).intensity is not None for u in dl) else "script tags (model unavailable)"
        return ComponentResult(round(score, 4), [f"peak {peak_v:.2f} at {fmt_ts(peak_u.start)}, mean {mean:.2f} ({src})"],
                               timestamp=peak_u.start, episode_id=peak_u.episode_id)

    def continuity(self, c: PromoCandidate) -> ComponentResult:
        if c.structure == "continuous":
            scenes = {u.scene for u in c.segments[0].lines}
            s = 1.0 - 0.1 * max(0, len(scenes) - 1)
            return ComponentResult(round(s, 4), [f"one unbroken stretch, {len(scenes)} scene(s)"])
        a, b = c.segments[0], c.segments[-1]
        if c.structure == "multi_segment":
            gap = b.start - a.end
            s = 0.8 - (0.1 if gap > 45 else 0.0)
            note = f"same episode, {gap:.1f}s skipped between segments"
        else:
            jump = abs(self.ctx.order[b.episode_id] - self.ctx.order[a.episode_id])
            s = 0.6 if jump == 1 else 0.45
            note = f"jumps {jump} episode(s) forward, story order kept"
        return ComponentResult(round(s, 4), [note])

    def dialogue_quality(self, c: PromoCandidate) -> ComponentResult:
        lines = [u for s in c.segments for u in s.lines]
        dl = [u for u in lines if u.kind == "dialogue"]
        if not dl:
            return ComponentResult(0.0, ["no spoken lines"])
        spoken = sum(u.duration for u in dl)
        coverage = min(1.0, spoken / max(1e-6, c.total_duration))
        filler = sum(1 for u in dl if len(u.text.split()) <= 2) / len(dl)
        avg = sum(len(u.text.split()) for u in dl) / len(dl)
        avg_ok = 1.0 if 3 <= avg <= 25 else 0.5
        s = 0.5 * min(1.0, coverage / 0.6) + 0.3 * (1 - filler) + 0.2 * avg_ok
        return ComponentResult(round(s, 4), [f"speech covers {coverage:.0%} of runtime, {filler:.0%} filler lines, {avg:.1f} words/line"])

    # ── full scoring ──────────────────────────────────────────────────────
    def score(self, c: PromoCandidate) -> PromoCandidate:
        if any(not u.timed for s in c.segments for u in s.lines):
            # cannot cut without real timestamps: reject without guessing
            empty = ComponentResult(0.0, ["timestamps unavailable"])
            c.hook = c.cliffhanger = c.coherence = c.emotion = c.continuity = c.dialogue_quality = empty
            c.score, c.passed = 0.0, False
            c.rejection_reasons = ["contains lines without timestamps"]
            c.explanation = "Not scored: at least one line has no usable source timestamp."
            return c
        c.hook = self.hook(c.segments[0])
        c.cliffhanger = self.cliff(c.segments[-1])
        c.coherence = analyse_coherence(c.segments, self.ctx, c.structure)
        c.emotion = self.emotion(c)
        c.continuity = self.continuity(c)
        c.dialogue_quality = self.dialogue_quality(c)
        comp = c.components()
        w = self.w
        c.score = (w.hook * comp["hook"] + w.cliffhanger * comp["cliffhanger"] + w.coherence * comp["coherence"]
                   + w.emotion * comp["emotion"] + w.continuity * comp["continuity"]
                   + w.dialogue_quality * comp["dialogue_quality"])
        self._gate(c)
        self._flags(c)
        c.explanation = explain(c)
        return c

    def _gate(self, c: PromoCandidate):
        g = self.g
        r = []
        d = c.total_duration
        if d < g.min_duration - 1e-6:
            r.append(f"too short ({d:.1f}s < {g.min_duration:.0f}s)")
        if d > g.max_duration + 1e-6:
            r.append(f"too long ({d:.1f}s > {g.max_duration:.0f}s)")
        if any(not u.timed for s in c.segments for u in s.lines):
            r.append("contains lines without timestamps")
        if c.hook.score < g.hook:
            r.append(f"weak opening hook ({c.hook.score:.2f} < {g.hook})")
        if c.cliffhanger.score < g.cliffhanger or getattr(c.cliffhanger, "open_signal", 1.0) < g.cliff_open_signal:
            r.append(f"no genuine cliffhanger ({c.cliffhanger.score:.2f}; open-loop {getattr(c.cliffhanger, 'open_signal', 0):.2f})")
        if c.coherence.score < g.coherence:
            r.append(f"poor standalone coherence ({c.coherence.score:.2f} < {g.coherence})")
        c.rejection_reasons = r
        c.passed = not r

    def _flags(self, c: PromoCandidate):
        f = []
        hook_line_feats = None
        for s in c.segments[:1]:
            for u in s.lines:
                if u.start == c.hook.timestamp:
                    hook_line_feats = self.ctx.f(u)
                    break
        if hook_line_feats is not None and hook_line_feats.cue_strength >= 0.8 and hook_line_feats.ml_arousal() < 0.3 and \
                hook_line_feats.arousal is not None:
            f.append("hook rests on a keyword match; emotion model reads the line as calm")
        ev0 = c.cliffhanger.evidence[0] if c.cliffhanger.evidence else ""
        if ev0.startswith("unanswered question") and "(charged)" not in ev0:
            f.append("cliffhanger rests on a single uncharged question")
        if any("answers it at once" in e or "same speaker continues" in e for e in c.cliffhanger.evidence):
            f.append("the source resolves the cut almost immediately")
        if any(e.startswith("penalty:") for e in c.coherence.evidence):
            f.append("coherence: " + "; ".join(e[9:] for e in c.coherence.evidence if e.startswith("penalty:")))
        if c.structure == "cross_episode":
            f.append(f"spans episodes {', '.join(str(s.episode_number) for s in c.segments)} — verify continuity of costumes/locations")
        if c.dialogue_quality.score < 0.5:
            f.append("little spoken dialogue — relies on visuals the transcript cannot verify")
        c.risk_flags = f


def explain(c: PromoCandidate) -> str:
    h, k = c.hook, c.cliffhanger
    h_ep = next((s.episode_number for s in c.segments if s.episode_id == h.episode_id), None)
    k_ep = next((s.episode_number for s in c.segments if s.episode_id == k.episode_id), None)
    quote_h = h.text if len(h.text) <= 60 else h.text[:57] + "…"
    quote_k = k.text if len(k.text) <= 60 else k.text[:57] + "…"
    return (f"Opens on {h.evidence[0]} (ep {h_ep} {fmt_ts(h.timestamp)}: “{quote_h}”) and cuts on "
            f"{k.evidence[0]} (ep {k_ep} {fmt_ts(k.timestamp)}: “{quote_k}”); {c.coherence.evidence[0]}.")


def _overlap(a, b) -> float:
    if a.episode_id != b.episode_id:
        return 0.0
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, min(a.duration, b.duration))


def conflicts(c: PromoCandidate, kept: PromoCandidate, max_overlap: float = 0.3) -> bool:
    if (c.hook.episode_id, c.hook.timestamp) == (kept.hook.episode_id, kept.hook.timestamp):
        return True
    if (c.cliffhanger.episode_id, c.cliffhanger.timestamp) == (kept.cliffhanger.episode_id, kept.cliffhanger.timestamp):
        return True
    return any(_overlap(a, b) > max_overlap for a in c.segments for b in kept.segments)


def select_diverse(cands: list, top_k: int = 15, max_per_episode: int = 2, max_overlap: float = 0.3,
                   assign_rank: bool = True, preselected: list = None) -> list:
    """Greedy: best-first, skipping candidates that reuse a hook/cut line or overlap a kept promo by > 30 %.

    ``preselected`` promos are kept first and count towards the per-episode limit and the overlap checks.
    """
    chosen, per_ep = list(preselected or []), {}
    for c in chosen:
        for e in c.episodes:
            per_ep[e] = per_ep.get(e, 0) + 1
    for c in sorted(cands, key=lambda x: -x.score):
        if c in chosen:
            continue
        if not c.passed:
            continue
        if any(per_ep.get(e, 0) >= max_per_episode for e in c.episodes):
            continue
        if any(conflicts(c, k, max_overlap) for k in chosen):
            continue
        chosen.append(c)
        for e in c.episodes:
            per_ep[e] = per_ep.get(e, 0) + 1
        if len(chosen) >= top_k:
            break
    if assign_rank:
        for i, c in enumerate(chosen, 1):
            c.rank = i
    return chosen


# ── promo mix (which kinds of promos fill the final list) ─────────────────
MIX_OPTIONS = {
    "best": "Best overall (ranked purely by priority score)",
    "include_multi_episode": "Include multi-episode promos (reserve slots for promos that join different episodes)",
    "multi_episode_only": "Multi-episode promos only (hook from one episode, ending from a later one)",
}


def select_with_mix(cands: list, top_k: int = 15, max_per_episode: int = 2, max_overlap: float = 0.3,
                    mix: str = "best", multi_episode_slots: int = 5) -> list:
    """Final selection with an optional quota for cross-episode promos. Scores are never changed;
    only which candidates fill the list. The list is ranked by priority score."""
    if mix not in MIX_OPTIONS:
        raise ValueError(f"mix must be one of {list(MIX_OPTIONS)}")
    cross = [c for c in cands if c.structure == "cross_episode"]
    if mix == "multi_episode_only":
        chosen = select_diverse(cross, top_k, max_per_episode, max_overlap, assign_rank=False)
    elif mix == "include_multi_episode":
        reserved = select_diverse(cross, min(multi_episode_slots, top_k), max_per_episode, max_overlap, assign_rank=False)
        chosen = select_diverse(cands, top_k, max_per_episode, max_overlap, assign_rank=False, preselected=reserved)
    else:
        chosen = select_diverse(cands, top_k, max_per_episode, max_overlap, assign_rank=False)
    chosen.sort(key=lambda c: -c.score)
    for i, c in enumerate(chosen, 1):
        c.rank = i
    return chosen
