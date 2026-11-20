"""Candidate promo generation (20–60 s), three structures:

1. ``continuous``    — one unbroken stretch of an episode, cut on line boundaries.
2. ``multi_segment`` — a short hook beat + a later stretch of the SAME episode that ends on a cliffhanger.
3. ``cross_episode`` — a hook beat from episode N + a stretch from episode N+1 or N+2 ending on a cliffhanger.

Multi-segment candidates are NOT random concatenations: the hook beat is chosen from the episode's strongest
openings, the closing stretch from its strongest endings, they must play in story order, leave a real gap
(otherwise the continuous version already covers it) and share at least one character.
Every candidate's total duration is the sum of its segment durations and must lie in [MIN_TOTAL, MAX_TOTAL].
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.cliffhanger import detect_cliffhanger
from src.context import ShowContext
from src.hook import detect_hook
from src.models import PromoCandidate, Segment

MIN_TOTAL = 20.0
MAX_TOTAL = 60.0


@dataclass
class GenConfig:
    min_total: float = MIN_TOTAL
    max_total: float = MAX_TOTAL
    hook_beats_per_episode: int = 6
    endings_per_episode: int = 6
    hook_beat_min: float = 4.0
    hook_beat_max: float = 15.0
    min_gap: float = 5.0           # multi-segment gap; smaller gaps are just a continuous clip
    max_episode_jump: int = 2      # cross-episode: N -> N+1 or N+2
    b_starts_per_pair: int = 4     # start options tried for the closing stretch


def make_segment(ctx: ShowContext, eid: str, i: int, j: int) -> Segment:
    lines = ctx.content[eid][i:j + 1]
    ep = ctx.episode(eid)
    return Segment(episode_id=eid, episode_number=ep.number, start=lines[0].start,
                   end=max(u.end for u in lines), lines=lines)


def _pid(structure: str, segs: list) -> str:
    raw = structure + "|" + "|".join(f"{s.episode_id}:{s.start:.2f}-{s.end:.2f}" for s in segs)
    return structure[:4] + "-" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def continuous_candidates(ctx: ShowContext, eid: str, cfg: GenConfig) -> list:
    lines = ctx.content[eid]
    out = []
    for i in range(len(lines)):
        start = lines[i].start
        end = start
        for j in range(i, len(lines)):
            end = max(end, lines[j].end)
            dur = end - start
            if dur > cfg.max_total:
                break
            if dur >= cfg.min_total and lines[j].end >= end - 1e-6:  # cut right after line j finishes
                seg = make_segment(ctx, eid, i, j)
                out.append(PromoCandidate(promo_id=_pid("continuous", [seg]), show=ctx.show,
                                          structure="continuous", segments=[seg]))
    return out


def hook_beats(ctx: ShowContext, eid: str, cfg: GenConfig) -> list:
    """Short opening beats (4–15 s) ranked by hook strength. Returns [(score, Segment)]."""
    lines = ctx.content[eid]
    beats = []
    for i in range(len(lines)):
        best = None
        for j in range(i, len(lines)):
            seg = make_segment(ctx, eid, i, j)
            if seg.duration > cfg.hook_beat_max:
                break
            if seg.duration >= cfg.hook_beat_min:
                best = seg
                if seg.duration >= 6.0:
                    break
        if best is not None:
            beats.append((detect_hook(best, ctx).score, best))
    beats.sort(key=lambda x: -x[0])
    chosen, used = [], set()
    for s, seg in beats:
        k = seg.lines[0].index
        if k in used:
            continue
        chosen.append((s, seg))
        used.update(u.index for u in seg.lines)
        if len(chosen) >= cfg.hook_beats_per_episode:
            break
    return chosen


def ending_lines(ctx: ShowContext, eid: str, cfg: GenConfig) -> list:
    """Indices of lines that make the strongest cut points, ranked by cliffhanger strength."""
    lines = ctx.content[eid]
    scored = []
    for j, u in enumerate(lines):
        i = j
        while i > 0 and lines[j].end - lines[i - 1].start <= 6.0:
            i -= 1
        scored.append((detect_cliffhanger(make_segment(ctx, eid, i, j), ctx).score, j))
    scored.sort(key=lambda x: -x[0])
    return [j for _, j in scored[: cfg.endings_per_episode]]


def _chars(seg: Segment, ctx: ShowContext) -> set:
    out = set()
    for u in seg.lines:
        if u.kind == "dialogue":
            out |= {t.lower() for t in u.character.split() if len(t) > 2}
        out |= ctx.mentions_names(u.text)
    return out


def _closing_stretches(ctx: ShowContext, eid: str, j: int, budget_min: float, budget_max: float,
                       earliest: float, cfg: GenConfig) -> list:
    lines = ctx.content[eid]
    end = lines[j].end
    opts = []
    for i in range(j, -1, -1):
        if lines[i].start < earliest:
            break
        seg = make_segment(ctx, eid, i, j)
        if seg.end > end + 1e-6:
            continue
        if seg.duration > budget_max:
            break
        if seg.duration >= budget_min:
            opts.append(seg)
    if len(opts) > cfg.b_starts_per_pair:  # spread the start options (short, medium, long)
        step = (len(opts) - 1) / (cfg.b_starts_per_pair - 1)
        opts = [opts[round(k * step)] for k in range(cfg.b_starts_per_pair)]
    return opts


def multi_candidates(ctx: ShowContext, beats: dict, endings: dict, cfg: GenConfig, stats: dict) -> list:
    out = []
    eids = [e.episode_id for e in ctx.episodes]
    for a_pos, a_eid in enumerate(eids):
        for _, a in beats.get(a_eid, []):
            for jump in range(0, cfg.max_episode_jump + 1):
                if a_pos + jump >= len(eids):
                    break
                b_eid = eids[a_pos + jump]
                structure = "multi_segment" if jump == 0 else "cross_episode"
                for j in endings.get(b_eid, []):
                    earliest = a.end + cfg.min_gap if jump == 0 else 0.0
                    lo = max(5.0, cfg.min_total - a.duration)
                    hi = cfg.max_total - a.duration
                    for b in _closing_stretches(ctx, b_eid, j, lo, hi, earliest, cfg):
                        if jump == 0 and b.start < a.end + cfg.min_gap:
                            continue
                        if not (_chars(a, ctx) & _chars(b, ctx)):
                            stats["rejected_unrelated_pairs"] = stats.get("rejected_unrelated_pairs", 0) + 1
                            continue
                        segs = [a, b]
                        out.append(PromoCandidate(promo_id=_pid(structure, segs), show=ctx.show,
                                                  structure=structure, segments=segs))
    return out
